"""Transient analysis using companion models (backward Euler / trapezoidal).

Discretisation
--------------
Capacitor, backward Euler (BE)::

    I_C(t) = G_C * V_C(t) - G_C * V_C(t-dt),        G_C = C/dt

Capacitor, trapezoidal (TRAP)::

    I_C(t) = G_C * V_C(t) - G_C * V_C(t-dt) - I_C(t-dt),   G_C = 2*C/dt

Inductor, backward Euler::

    I_L(t) = I_L(t-dt) + G_L * V_L(t),              G_L = dt/L

Inductor, trapezoidal::

    I_L(t) = I_L(t-dt) + G_L * (V_L(t) + V_L(t-dt)), G_L = dt/(2*L)

In every case the element is stamped into the MNA matrix as a conductance
``G`` in parallel with an independent current source ``J``, which is why the
transient system has the same structure as the DC system and needs no extra
unknowns for reactive elements (voltage sources keep their current unknown).

Start-up
--------
Initial conditions at t = 0 are obtained from an operating-point solve in
which every capacitor is replaced by a voltage source equal to its initial
voltage ``v0`` and every inductor by a current source equal to its initial
current ``i0``.  When the trapezoidal method is selected, the *first* step is
always taken with backward Euler (a standard measure that avoids the trapezoid
ringing right after source discontinuities and does not require the capacitor
current at t = 0).
"""

from __future__ import annotations

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
)
from .nonlinear import DEFAULT_VT, NRStats, newton_raphson

# Condition number above which a warning is emitted for a transient step.
_TRAN_COND_WARN = 1.0e14


@dataclass
class TranResult:
    """Result of a transient simulation.

    Attributes
    ----------
    t            : time samples (seconds)
    v            : node number -> voltage samples
    i            : element name -> branch current samples (through n+ -> n-)
    method       : 'be' or 'trap'
    dt, tstop    : step and stop time used
    init_stats   : Newton-Raphson stats of the t = 0 start-up solve
    step_stats   : per-step Newton-Raphson stats
    warnings     : human-readable numerical warnings
    converged    : True if every step converged
    """

    t: np.ndarray
    v: Dict[int, np.ndarray]
    i: Dict[str, np.ndarray]
    method: str
    dt: float
    tstop: float
    init_stats: Optional[NRStats]
    step_stats: List[NRStats]
    warnings: List[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return all(s.converged for s in self.step_stats)

    def node_voltage(self, node: int) -> np.ndarray:
        return self.v[node]

    def branch_current(self, name: str) -> np.ndarray:
        return self.i[name]


def _cap_state(c: Capacitor, v_c_old: float, i_c_old: float, dt: float, method: str) -> dict:
    if method == "be":
        g = c.value / dt
        j = -g * v_c_old
    else:
        g = 2.0 * c.value / dt
        j = -g * v_c_old - i_c_old
    return {"g": g, "j": j, "v_old": v_c_old, "i_old": i_c_old}


def _ind_state(c: Inductor, v_l_old: float, i_l_old: float, dt: float, method: str) -> dict:
    if method == "be":
        g = dt / c.value
        j = i_l_old
    else:
        g = dt / (2.0 * c.value)
        j = i_l_old + g * v_l_old
    return {"g": g, "j": j, "v_old": v_l_old, "i_old": i_l_old}


def _cap_current(st: dict, dv_new: float, method: str) -> float:
    i = st["g"] * (dv_new - st["v_old"])
    if method == "trap":
        i -= st["i_old"]
    return i


def _ind_current(st: dict, dv_new: float, method: str) -> float:
    i = st["i_old"] + st["g"] * dv_new
    if method == "trap":
        i += st["g"] * st["v_old"]
    return i


def _project_to_step(
    x_init: np.ndarray,
    node_rows: Dict[int, int],
    vs_rows: Dict[str, int],
    circuit: Circuit,
) -> np.ndarray:
    """Project a start-up solution (may contain capacitor stand-in currents)
    onto the unknown vector used by transient step systems."""
    n_nodes = len(node_rows)
    vs_names = [c.name for c in circuit.voltage_sources]
    x = np.zeros(n_nodes + len(vs_names))
    x[:n_nodes] = x_init[:n_nodes]
    for i, name in enumerate(vs_names):
        if name in vs_rows:
            x[n_nodes + i] = x_init[vs_rows[name]]
    return x


def _solve_init_op(
    circuit: Circuit, t: float, dt: float, vt: float
) -> Tuple[np.ndarray, Dict[int, int], Dict[str, int], Optional[dict], NRStats, List[str]]:
    """Operating point at t = 0 with capacitors/inductors at their initial
    state.  Falls back to dropping capacitor sources when the circuit has a
    capacitor directly in parallel with an ideal voltage source (a genuinely
    conflicting pair for any MNA start-up; see the technical document)."""
    warnings: List[str] = []
    caps_used = True
    try:
        result = newton_raphson(
            circuit, lambda: mna.build_init_op_system(circuit, t), vt=vt
        )
    except mna.MNAError as exc:
        # Retry without capacitor stand-ins: their voltage is pinned by the
        # parallel source anyway; the capacitor current at t=0 is then unknown,
        # which is acceptable because the first step uses backward Euler.
        try:
            caps_used = False
            result = newton_raphson(
                circuit,
                lambda: _init_op_without_caps(circuit, t),
                vt=vt,
            )
        except mna.MNAError as exc2:
            raise mna.MNAError(
                "cannot find a consistent initial state at t = 0: "
                f"{exc}\nAfter dropping parallel capacitors: {exc2}"
            ) from exc2
        warnings.append(
            "start-up: capacitor(s) in parallel with ideal voltage source(s); "
            "capacitor stand-ins dropped from the t=0 solve (first step is "
            "backward Euler)"
        )
    return result.x, result.node_rows, result.vs_rows, (
        result if caps_used else None
    ), result.stats, warnings


def _init_op_without_caps(circuit: Circuit, t: float) -> mna.MnaSystem:
    names = [c.name for c in circuit.voltage_sources]
    A, z, node_rows, vs_rows = _base_tran_like(circuit, names, t)
    return mna.MnaSystem(A, z, node_rows, vs_rows, [])


def _base_tran_like(circuit: Circuit, vs_names: List[str], t: float):
    A, z, node_rows, vs_rows = mna._base_system(circuit, vs_names)
    sources = []
    for c in circuit.resistors:
        mna._stamp_conductance(A, z, node_rows, c.n1, c.n2, c.conductance)
    for c in circuit.current_sources:
        val = c.waveform.value(t)
        mna._stamp_current_source(A, z, node_rows, c.n1, c.n2, val)
        sources.append((c, val))
    for c in circuit.voltage_sources:
        val = c.waveform.value(t)
        mna._stamp_voltage_source(A, z, node_rows, c.n1, c.n2, val, vs_rows[c.name])
        sources.append((c, val))
    for c in circuit.inductors:
        mna._stamp_current_source(A, z, node_rows, c.n1, c.n2, c.i0)
    return A, z, node_rows, vs_rows


def _element_currents(
    circuit: Circuit,
    x: np.ndarray,
    node_rows: Dict[int, int],
    vs_rows: Dict[str, int],
    t: float,
    states: Dict[str, dict],
    method: str,
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
        elif isinstance(c, Capacitor):
            st = states[c.name]
            currents[c.name] = _cap_current(st, vnode(c.n1) - vnode(c.n2), method)
        elif isinstance(c, Inductor):
            st = states[c.name]
            currents[c.name] = _ind_current(st, vnode(c.n1) - vnode(c.n2), method)
        elif isinstance(c, Diode):
            vd = vnode(c.anode) - vnode(c.cathode)
            currents[c.name] = c.current(vd, vt)
    return currents


def run_transient(
    circuit: Circuit,
    *,
    dt: float,
    tstop: float,
    method: str = "trap",
    start: float = 0.0,
    vt: float = DEFAULT_VT,
    tol: float = 1e-9,
    max_iterations: int = 100,
) -> TranResult:
    """Run a transient simulation.

    Parameters
    ----------
    dt     : fixed time step (seconds)
    tstop  : stop time (seconds)
    method : 'be' (backward Euler) or 'trap' (trapezoidal)
    start  : only record output for t >= start (simulation still starts at 0
             so that the physical initial conditions apply)
    """
    if dt <= 0 or tstop <= 0:
        raise NetlistError("transient: dt and tstop must be positive")
    if method not in ("be", "trap"):
        raise NetlistError("transient: method must be 'be' or 'trap'")
    if start < 0 or start > tstop:
        raise NetlistError("transient: start must satisfy 0 <= start <= tstop")

    warnings: List[str] = []

    # ---- t = 0 operating point -------------------------------------------
    x_cur, node_rows, vs_rows, init_result, init_stats, init_warnings = (
        _solve_init_op(circuit, 0.0, dt, vt)
    )
    warnings.extend(init_warnings)

    # The transient step systems carry voltage unknowns only for voltage
    # sources (no capacitor stand-ins), so re-project the start-up vector.
    x_step = _project_to_step(x_cur, node_rows, vs_rows, circuit)

    nodes = circuit.nodes
    names = [c.name for c in circuit.components]
    n_steps = int(np.floor(tstop / dt + 1e-9))
    times = np.arange(n_steps + 1) * dt
    if times[-1] < tstop - 1e-12:
        times = np.append(times, tstop)

    v_hist: Dict[int, np.ndarray] = {n: np.zeros(len(times)) for n in nodes}
    i_hist: Dict[str, np.ndarray] = {name: np.zeros(len(times)) for name in names}
    step_stats: List[NRStats] = []

    # Initial currents at t = 0 (capacitor currents come from the start-up
    # solve when it used capacitor stand-ins; otherwise they are placeholders
    # that are never used because the first step is backward Euler).
    def vnode(n: int) -> float:
        return float(x_cur[node_rows[n]]) if n != 0 else 0.0

    init_states: Dict[str, dict] = {}
    for c in circuit.capacitors:
        init_states[c.name] = _cap_state(c, c.v0, 0.0, dt, "be")
    for c in circuit.inductors:
        init_states[c.name] = _ind_state(c, 0.0, c.i0, dt, "be")
    i0 = _element_currents(
        circuit, x_cur, node_rows, vs_rows, 0.0, init_states, "be", vt
    )
    # The t = 0 currents are physical initial conditions, not companion
    # extrapolations: inductors carry their declared i0; capacitors carry the
    # current of their stand-in voltage source from the start-up solve.
    for c in circuit.inductors:
        i0[c.name] = c.i0
    for c in circuit.capacitors:
        if init_result is not None and c.name in vs_rows:
            i0[c.name] = float(x_cur[vs_rows[c.name]])
        else:
            i0[c.name] = 0.0  # unknown but unused (first step is backward Euler)
    for n in nodes:
        v_hist[n][0] = vnode(n)
    for name in names:
        i_hist[name][0] = i0.get(name, 0.0)

    # ---- time stepping -----------------------------------------------------
    states: Dict[str, dict] = {}
    for c in circuit.capacitors:
        states[c.name] = {"v_old": vnode(c.n1) - vnode(c.n2), "i_old": i0[c.name]}
    for c in circuit.inductors:
        states[c.name] = {
            "v_old": vnode(c.n1) - vnode(c.n2),
            "i_old": i0[c.name],
        }
    # Warn if a dropped-cap fallback left the capacitor voltage inconsistent.
    if init_result is None:
        for c in circuit.capacitors:
            solved = vnode(c.n1) - vnode(c.n2)
            if abs(solved - c.v0) > 1e-6:
                warnings.append(
                    f"{c.name}: initial voltage set by the parallel source "
                    f"({solved:.6g} V) differs from declared v0 ({c.v0:g} V)"
                )

    for k in range(1, len(times)):
        t_new = times[k]
        dt_k = t_new - times[k - 1]
        method_eff = "be" if (method == "be" or k == 1) else "trap"
        # Refresh companion states using values at the previous time.
        for c in circuit.capacitors:
            dv_old = vnode(c.n1) - vnode(c.n2)
            states[c.name] = _cap_state(
                c, dv_old, i_hist[c.name][k - 1], dt_k, method_eff
            )
        for c in circuit.inductors:
            dv_old = vnode(c.n1) - vnode(c.n2)
            states[c.name] = _ind_state(
                c, dv_old, i_hist[c.name][k - 1], dt_k, method_eff
            )

        result = newton_raphson(
            circuit,
            lambda: mna.build_tran_step_system(
                circuit, t_new, states, method_eff
            ),
            x0=x_step,
            vt=vt,
            atol=tol,
            rtol=tol,
            max_iterations=max_iterations,
        )
        step_stats.append(result.stats)
        if not result.stats.converged:
            warnings.append(f"t={t_new:.6g} s: {result.stats.message}")

        x_cur = result.x
        x_step = result.x
        currents = _element_currents(
            circuit, x_cur, node_rows, vs_rows, t_new, states, method_eff, vt
        )

        # Ill-conditioning warning.
        cond = mna.estimate_condition(
            mna.build_tran_step_system(circuit, t_new, states, method_eff).A
        )
        if cond > _TRAN_COND_WARN:
            warnings.append(
                f"t={t_new:.6g} s: ill-conditioned matrix (cond~{cond:.2e})"
            )

        for n in nodes:
            v_hist[n][k] = vnode(n)
        for name in names:
            i_hist[name][k] = currents.get(name, 0.0)

    # Keep only t >= start for output.
    keep = times >= start - 1e-12
    t_out = times[keep]
    v_out = {n: arr[keep] for n, arr in v_hist.items()}
    i_out = {name: arr[keep] for name, arr in i_hist.items()}

    return TranResult(
        t=t_out,
        v=v_out,
        i=i_out,
        method=method,
        dt=dt,
        tstop=tstop,
        init_stats=init_stats,
        step_stats=step_stats,
        warnings=warnings,
    )
