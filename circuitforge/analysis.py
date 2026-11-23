"""High-level analyses: DC operating point and full netlist runs.

This module is the entry point used by the CLI: it interprets the directives
found in a netlist (``.op``, ``.tran``, ``.plot``, ``.export``) and returns
ready-to-present results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import mna
from .netlist import (
    Circuit,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    NetlistError,
    OpDirective,
    TranDirective,
    PlotDirective,
    ExportDirective,
)
from .nonlinear import DEFAULT_VT, NRStats, newton_raphson, solve_dc
from .transient import TranResult, run_transient


@dataclass
class OpResult:
    """Result of a DC operating-point analysis."""

    node_voltages: Dict[int, float]
    branch_currents: Dict[str, float]
    stats: NRStats
    condition: Optional[float]
    warnings: List[str] = field(default_factory=list)
    matrix_size: Tuple[int, int] = (0, 0)
    diode_voltages: Dict[str, float] = field(default_factory=dict)


def _dc_branch_currents(
    circuit: Circuit,
    x: np.ndarray,
    node_rows: Dict[int, int],
    vs_rows: Dict[str, int],
    t: float,
    vt: float,
) -> Dict[str, float]:
    def vnode(n: int) -> float:
        return float(x[node_rows[n]]) if n != 0 else 0.0

    currents: Dict[str, float] = {}
    for c in circuit.components:
        if isinstance(c, Resistor):
            currents[c.name] = (vnode(c.n1) - vnode(c.n2)) * c.conductance
        elif isinstance(c, VoltageSource):
            currents[c.name] = float(x[vs_rows[c.name]])
        elif isinstance(c, CurrentSource):
            currents[c.name] = c.waveform.value(t)
        elif isinstance(c, Inductor):
            # DC: ideal inductor is a short; its current is a voltage-like
            # unknown in the DC system.
            currents[c.name] = float(x[vs_rows[c.name]])
        elif isinstance(c, Capacitor):
            currents[c.name] = 0.0  # open circuit at DC
        elif isinstance(c, Diode):
            vd = vnode(c.anode) - vnode(c.cathode)
            currents[c.name] = c.current(vd, vt)
    return currents


def estimate_condition(A) -> float:
    if A.size == 0:
        return 1.0
    return mna.estimate_condition(A)


def run_op(
    circuit: Circuit,
    *,
    t: float = 0.0,
    vt: float = DEFAULT_VT,
    warn_cond: float = 1e12,
    **nr_kwargs,
) -> OpResult:
    """Run a DC operating-point analysis."""
    result = solve_dc(circuit, t=t, vt=vt, **nr_kwargs)
    system = mna.build_dc_op_system(circuit, t)
    cond = estimate_condition(system.A)
    warnings: List[str] = []
    if not result.stats.converged:
        warnings.append(result.stats.message)
    if cond > warn_cond:
        warnings.append(f"ill-conditioned matrix (cond~{cond:.2e})")

    currents = _dc_branch_currents(
        circuit, result.x, result.node_rows, result.vs_rows, t, vt
    )
    node_voltages = {
        n: float(result.x[r]) for n, r in result.node_rows.items()
    }
    return OpResult(
        node_voltages=node_voltages,
        branch_currents=currents,
        stats=result.stats,
        condition=cond,
        warnings=warnings,
        matrix_size=system.A.shape,
        diode_voltages=result.diode_voltages,
    )


@dataclass
class NetlistRun:
    """Everything produced by executing one netlist file."""

    circuit: Circuit
    op: Optional[OpResult] = None
    tran: Optional[TranResult] = None
    plots: List[str] = field(default_factory=list)
    csv_files: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


def parse_variable(var: str) -> Tuple[str, str]:
    """Split 'v(2)' -> ('v', '2') or 'i(R1)' -> ('i', 'R1')."""
    var = var.strip()
    if "(" not in var or not var.endswith(")"):
        raise NetlistError(
            f"invalid plot variable {var!r}; use v(<node>) or i(<element>)"
        )
    kind, _, arg = var.partition("(")
    arg = arg[:-1].strip()
    kind = kind.strip().lower()
    if kind not in ("v", "i"):
        raise NetlistError(
            f"invalid plot variable {var!r}; use v(<node>) or i(<element>)"
        )
    return kind, arg


def run_netlist(
    circuit: Circuit,
    *,
    plot_dir: Optional[str] = None,
    force_op: bool = False,
    force_tran: Optional[TranDirective] = None,
    write_plots: bool = True,
    vt: float = DEFAULT_VT,
) -> NetlistRun:
    """Execute the directives of a parsed circuit.

    ``force_tran`` (a :class:`TranDirective`) overrides ``.tran``; ``force_op``
    runs ``.op`` even without the directive.
    """
    run = NetlistRun(circuit=circuit)

    want_op = force_op or circuit.get_directive(OpDirective) is not None
    if want_op:
        run.op = run_op(circuit, vt=vt)
        run.messages.append(
            f"DC operating point: {run.op.stats.message} "
            f"(matrix {run.op.matrix_size[0]}x{run.op.matrix_size[1]})"
        )

    tran_directive = force_tran or circuit.get_directive(TranDirective)
    if tran_directive is not None:
        run.tran = run_transient(
            circuit,
            dt=tran_directive.dt,
            tstop=tran_directive.tstop,
            method=tran_directive.method,
            start=tran_directive.start,
            vt=vt,
        )
        run.messages.append(
            f"Transient: {len(run.tran.t)} samples, method={run.tran.method}, "
            f"dt={run.tran.dt:g} s"
        )
        if run.tran.warnings:
            for w in run.tran.warnings:
                run.messages.append(f"  warning: {w}")
        else:
            run.messages.append("  no numerical warnings")

        variables = _collect_variables(circuit)
        if variables and write_plots:
            stem = None
            if circuit.source_path:
                stem = os.path.splitext(os.path.basename(circuit.source_path))[0]
            plots = plot_tran(run.tran, variables, plot_dir=plot_dir, stem=stem)
            run.plots.extend(plots)
            run.messages.append(
                "  plots written: " + ", ".join(plots) if plots else "  no plots"
            )

        export = circuit.get_directive(ExportDirective)
        if export is not None:
            path = export_csv(run.tran, variables, export.filename)
            run.csv_files.append(path)
            run.messages.append(f"  CSV written: {path}")
    return run


def _collect_variables(circuit: Circuit) -> List[str]:
    """Variables requested by .plot; falls back to all nodes + all elements."""
    plot = circuit.get_directive(PlotDirective)
    if plot is not None:
        return plot.variables
    vars_ = [f"v({n})" for n in circuit.nodes]
    vars_ += [f"i({c.name})" for c in circuit.components]
    return vars_


def plot_tran(result: TranResult, variables: List[str], plot_dir=None, stem=None):
    from .plotting import plot_transient  # deferred import (matplotlib)

    return plot_transient(result, variables, plot_dir=plot_dir, stem=stem)


def export_csv(result: TranResult, variables: List[str], filename: str) -> str:
    from .export import write_csv

    return write_csv(result, variables, filename)
