"""Matplotlib plotting of transient results (node voltages and branch
currents).  The plot is a clean engineering figure: one subplot per quantity
family.  If matplotlib is unavailable the CLI degrades gracefully (text
tables + CSV remain functional)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analysis import parse_variable
from .netlist import NetlistError
from .transient import TranResult


def _select_variables(
    result: TranResult, variables: List[str]
) -> Tuple[List[Tuple[int, str]], List[Tuple[str, str]]]:
    """Split requested variables into (voltage-node, label) and
    (element-name, label) lists, validating they exist in the result."""
    v_items: List[Tuple[int, str]] = []
    i_items: List[Tuple[str, str]] = []
    for var in variables:
        kind, arg = parse_variable(var)
        if kind == "v":
            node = int(arg)
            if node == 0:
                # v(0) is identically zero; allowed but useless
                continue
            if node not in result.v:
                raise NetlistError(f"node {node} not present in the circuit")
            v_items.append((node, var))
        else:
            if arg not in result.i:
                raise NetlistError(
                    f"element {arg!r} not present in the circuit"
                )
            i_items.append((arg, var))
    return v_items, i_items


def plot_transient(
    result: TranResult,
    variables: List[str],
    plot_dir: Optional[str] = None,
    stem: Optional[str] = None,
) -> List[str]:
    """Plot the requested node voltages / branch currents vs time.

    ``stem`` (e.g. the netlist file name) is included in the PNG file name so
    that different circuits do not overwrite each other's plots.

    Returns the list of PNG file paths written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is not installed; plotting is unavailable"
        ) from exc

    v_items, i_items = _select_variables(result, variables)
    n_axes = int(bool(v_items)) + int(bool(i_items))
    if n_axes == 0:
        return []

    fig, axes = plt.subplots(
        n_axes, 1, figsize=(8, 3.2 * n_axes), sharex=True, squeeze=False
    )
    ax_v = axes[0][0] if v_items else None
    ax_i = axes[1][0] if i_items and v_items else axes[0][0]

    t = result.t
    if ax_v is not None:
        for node, label in v_items:
            ax_v.plot(t, result.v[node], label=label)
        ax_v.set_ylabel("Voltage (V)")
        ax_v.legend(loc="best", fontsize=8)
        ax_v.grid(True, alpha=0.3)
        ax_v.set_title(f"CircuitForge transient ({result.method}, dt={result.dt:g} s)")
    if ax_i is not None:
        for name, label in i_items:
            ax_i.plot(t, result.i[name], label=label)
        ax_i.set_ylabel("Current (A)")
        ax_i.set_xlabel("Time (s)")
        ax_i.legend(loc="best", fontsize=8)
        ax_i.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = plot_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    tag = f"tran_{result.method}_{result.dt:g}".replace(".", "p")
    base = f"{stem}_" if stem else ""
    path = os.path.join(out_dir, f"{base}{tag}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [os.path.abspath(path)]
