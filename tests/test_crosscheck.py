"""Cross-check the package's DC results against the independent reference
implementation in ``reference_nodal.py`` on random resistive networks."""

import numpy as np
import pytest

from circuitforge.analysis import run_op
from circuitforge.netlist import parse_netlist
from tests.reference_nodal import nodal_analysis_dc


def _random_divider_netlist(seed):
    rng = np.random.default_rng(seed)
    lines = [f"V1 1 0 DC({10.0})"]
    n1 = 1
    for k in range(2, 7):
        r = float(rng.uniform(100, 5000))
        lines.append(f"R{k - 1} {n1} {k} {r}")
        n1 = k
    lines.append(f"R6 {n1} 0 {float(rng.uniform(100, 5000))}")
    return "\n".join(lines)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_resistive_chains(seed):
    text = _random_divider_netlist(seed)
    c = parse_netlist(text)
    ours = run_op(c)

    # Build the same network for the reference solver.
    resistors = []
    for comp in c.resistors:
        resistors.append((comp.n1, comp.n2, comp.value))
    vsources = [(comp.n1, comp.n2, comp.waveform.value(0.0))
                for comp in c.voltage_sources]
    ref_v, _ = nodal_analysis_dc(c.nodes, resistors, vsources, [])

    for node, v in ref_v.items():
        assert ours.node_voltages[node] == pytest.approx(v, abs=1e-9)


def test_random_network_with_current_sources(seed=11):
    rng = np.random.default_rng(seed)
    lines = [f"V1 1 0 DC({5.0})"]
    edges = [(1, 2), (2, 3), (1, 3), (3, 0), (2, 0), (1, 0)]
    rvals = [float(rng.uniform(200, 3000)) for _ in edges]
    for (a, b), r in zip(edges, rvals):
        lines.append(f"R{a}{b} {a} {b} {r}")
    lines.append("I1 0 2 1e-3")
    c = parse_netlist("\n".join(lines))
    ours = run_op(c)

    resistors = [(a, b, r) for (a, b), r in zip(edges, rvals)]
    vsources = [(1, 0, 5.0)]
    isources = [(0, 2, 1e-3)]
    ref_v, _ = nodal_analysis_dc(c.nodes, resistors, vsources, isources)
    for node, v in ref_v.items():
        assert ours.node_voltages[node] == pytest.approx(v, abs=1e-9)
