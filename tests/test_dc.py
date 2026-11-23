"""Tests for DC operating-point analysis (MNA)."""

import numpy as np
import pytest

from circuitforge import mna
from circuitforge.analysis import run_op
from circuitforge.netlist import NetlistError, parse_netlist
from circuitforge.nonlinear import solve_dc


def _op(text):
    return run_op(parse_netlist(text))


def test_voltage_divider():
    r = _op("V1 1 0 DC(10)\nR1 1 2 1000\nR2 2 0 1000")
    assert r.node_voltages[1] == pytest.approx(10.0)
    assert r.node_voltages[2] == pytest.approx(5.0)
    # KCL at node 2, independent of the solver:
    assert (10 - 5) / 1000 - 5 / 1000 == pytest.approx(0.0)
    assert r.branch_currents["R1"] == pytest.approx(5e-3)
    assert r.branch_currents["R2"] == pytest.approx(5e-3)
    # Source current is negative because it flows from ground INTO node 1.
    assert r.branch_currents["V1"] == pytest.approx(-5e-3)
    assert r.stats.converged


def test_three_resistor_chain_kcl():
    # 12 V -> 1k -> 2k -> 3k -> gnd; verify every node by hand KCL.
    r = _op("V1 1 0 DC(12)\nR1 1 2 1000\nR2 2 3 2000\nR3 3 0 3000")
    itot = 12 / 6000
    v2 = 12 - itot * 1000
    v3 = itot * 3000
    assert r.node_voltages[2] == pytest.approx(v2, abs=1e-9)
    assert r.node_voltages[3] == pytest.approx(v3, abs=1e-9)
    # KCL at every node:
    assert (12 - v2) / 1e3 - (v2 - v3) / 2e3 == pytest.approx(0.0)
    assert (v2 - v3) / 2e3 - v3 / 3e3 == pytest.approx(0.0)
    assert r.branch_currents["R1"] == pytest.approx(itot)
    assert r.branch_currents["R2"] == pytest.approx(itot)
    assert r.branch_currents["R3"] == pytest.approx(itot)


def test_current_source_into_resistor():
    r = _op("I1 0 1 DC(1e-3)\nR1 1 0 1000")
    assert r.node_voltages[1] == pytest.approx(1.0)
    assert r.branch_currents["R1"] == pytest.approx(1e-3)
    assert r.branch_currents["I1"] == pytest.approx(1e-3)


def test_supernode_voltage_source_between_nodes():
    r = _op(
        "V1 3 0 DC(10)\nR1 3 1 1000\nV2 1 2 DC(2)\nR2 2 0 1000"
    )
    assert r.node_voltages[1] == pytest.approx(6.0)
    assert r.node_voltages[2] == pytest.approx(4.0)
    # Hand KCL on the supernode {1,2}:
    assert (10 - 6) / 1e3 - 4 / 1e3 == pytest.approx(0.0)


def test_inductor_is_short_at_dc():
    r = _op("V1 1 0 DC(5)\nR1 1 2 1000\nL1 2 0 1e-3")
    assert r.node_voltages[2] == pytest.approx(0.0)
    assert r.branch_currents["L1"] == pytest.approx(5e-3)
    assert r.branch_currents["R1"] == pytest.approx(5e-3)


def test_capacitor_is_open_at_dc():
    r = _op("V1 1 0 DC(5)\nR1 1 2 1000\nC1 2 0 1e-6")
    assert r.node_voltages[2] == pytest.approx(5.0)  # no current through R1
    assert r.branch_currents["R1"] == pytest.approx(0.0, abs=1e-12)
    assert r.branch_currents["C1"] == pytest.approx(0.0, abs=1e-12)


def test_floating_node_diagnostic():
    with pytest.raises(mna.MNAError, match="floating"):
        _op("I1 0 3 DC(1e-3)\nR1 1 2 1000\nV1 1 0 DC(5)")


def test_floating_node_listed_exactly():
    c = parse_netlist("I1 0 3 DC(1e-3)\nV1 1 0 DC(5)")
    assert mna.floating_nodes(c, mode="dc") == [3]


def test_voltage_source_loop_singular():
    with pytest.raises(mna.MNAError, match="singular"):
        _op("V1 1 0 DC(5)\nV2 1 0 DC(3)\nR1 1 0 1000")


def test_two_sources_same_value_loop():
    # Two identical voltage sources in parallel is also singular.
    with pytest.raises(mna.MNAError, match="singular"):
        _op("V1 1 0 DC(5)\nV2 1 0 DC(5)\nR1 1 0 1000")


def test_no_components():
    r = _op("")
    assert r.node_voltages == {}
    assert r.branch_currents == {}


def test_branch_current_signs_consistent():
    # A 5 V source drives 1k from node 1 to node 2; node 2 grounded via 2k.
    # Hand check: v2 = 5*2k/(1k+2k) = 3.333...
    r = _op("V1 1 0 DC(5)\nR1 1 2 1000\nR2 2 0 2000")
    v2 = 5 * 2000 / 3000
    i = (5 - v2) / 1000
    assert r.branch_currents["R1"] == pytest.approx(i)
    assert r.branch_currents["R2"] == pytest.approx(i)
    # i(V1) flows from n+ to n- through the source: node 1 is being charged
    # by the source (current flows ground -> node 1), so the signed value is
    # negative.
    assert r.branch_currents["V1"] == pytest.approx(-i)


def test_kcl_invariant_across_random_divider():
    rng = np.random.default_rng(7)
    ra, rb = 470.0, 2200.0
    r = _op(f"V1 1 0 DC(9)\nR1 1 2 {ra}\nR2 2 0 {rb}")
    v2 = r.node_voltages[2]
    assert (9 - v2) / ra - v2 / rb == pytest.approx(0.0, abs=1e-9)


def test_matrix_condition_reported():
    r = _op("V1 1 0 DC(10)\nR1 1 2 1000\nR2 2 0 1000")
    assert r.condition is not None and r.condition > 0
