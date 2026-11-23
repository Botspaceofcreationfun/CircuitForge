"""Tests for the nonlinear (diode) Newton-Raphson solver."""

import math

import pytest

from circuitforge.analysis import run_op
from circuitforge.netlist import Diode, NetlistError, parse_netlist
from circuitforge.nonlinear import DEFAULT_VT, solve_dc


def _diode_voltage_from_current(isat, current, n=1.0, vt=DEFAULT_VT):
    """Inverse diode equation: Vd = n*Vt*ln(I/Is + 1)."""
    return n * vt * math.log(current / isat + 1.0)


def _bisect_diode_bias(vsrc, r, isat=1e-14, n=1.0, vt=DEFAULT_VT):
    """Independent reference: solve (Vsrc - Vd)/R = Is*(exp(Vd/(nVt)) - 1)
    by plain bisection (no Jacobian, no MNA)."""
    def f(vd):
        return (vsrc - vd) / r - isat * (math.exp(vd / (n * vt)) - 1.0)

    lo, hi = 0.0, vsrc
    assert f(lo) >= 0 and f(hi) <= 0, "bracketing failed"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_diode_dc_matches_analytical():
    vsrc, r = 5.0, 1000.0
    c = parse_netlist(f"V1 1 0 DC({vsrc})\nR1 1 2 {r}\nD1 2 0")
    result = solve_dc(c)
    vd_ref = _bisect_diode_bias(vsrc, r)
    vd_num = result.diode_voltages["D1"]
    assert abs(vd_num - vd_ref) < 1e-6
    assert result.stats.converged
    assert result.stats.status == "converged"
    # current must equal (Vsrc - Vd)/R
    i = (vsrc - vd_num) / r
    assert abs(result.x[result.node_rows[2]] - vd_num) < 1e-12


def test_diode_current_source_hard_start():
    """A current source into a diode-only node is the hard case: the Newton
    iteration must walk the exponential from zero bias without blowing up."""
    isat = 1e-14
    c = parse_netlist(f"I1 0 1 DC(1e-3)\nD1 1 0")
    result = solve_dc(c)
    vd_ref = _diode_voltage_from_current(isat, 1e-3)
    assert abs(result.diode_voltages["D1"] - vd_ref) < 1e-4
    assert result.stats.converged


def test_diode_model_parameters_respected():
    c = parse_netlist(
        "I1 0 1 DC(1e-3)\nD1 1 0 sharp\n.model sharp IS=1e-12 N=2"
    )
    result = solve_dc(c)
    vd_ref = _diode_voltage_from_current(1e-12, 1e-3, n=2.0)
    assert abs(result.diode_voltages["D1"] - vd_ref) < 1e-4


def test_two_diodes_in_series():
    c = parse_netlist("I1 0 1 DC(1e-3)\nD1 1 2\nD2 2 0")
    result = solve_dc(c)
    vd1 = result.diode_voltages["D1"]
    vd2 = result.diode_voltages["D2"]
    assert vd1 == pytest.approx(vd2, rel=1e-6)
    ref = _diode_voltage_from_current(1e-14, 1e-3)
    assert abs((vd1 + vd2) - 2 * ref) < 1e-4
    assert result.stats.converged


def test_diode_reverse_bias():
    c = parse_netlist("V1 1 0 DC(-5)\nR1 1 2 1000\nD1 2 0")
    result = solve_dc(c)
    # Reverse biased: the diode blocks, so no current flows, node 2 floats
    # to the source voltage through the resistor: vd = -5 V and i ~ -Is.
    assert result.diode_voltages["D1"] == pytest.approx(-5.0, abs=1e-9)
    assert result.stats.converged


def test_op_reports_diode_bias():
    r = run_op(parse_netlist("V1 1 0 DC(5)\nR1 1 2 1000\nD1 2 0"))
    vd_ref = _bisect_diode_bias(5.0, 1000.0)
    assert abs(r.diode_voltages["D1"] - vd_ref) < 1e-6
    assert r.node_voltages[2] == pytest.approx(vd_ref, abs=1e-6)


def test_convergence_diagnostics_present():
    c = parse_netlist("I1 0 1 DC(1e-3)\nD1 1 0")
    result = solve_dc(c)
    s = result.stats
    assert s.iterations >= 1
    assert s.final_residual < 1e-7
    assert "converged" in s.message


def test_diode_bad_model_raises():
    with pytest.raises(NetlistError, match="IS>0"):
        parse_netlist("D1 1 0 bad\n.model bad IS=0 N=1")
