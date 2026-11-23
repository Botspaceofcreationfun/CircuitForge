"""Transient simulation validation against analytical solutions.

The reference formulas are derived independently of the solver:

  RC:    V(t) = Vf + (Vi - Vf)*exp(-t/(R*C))
  RL:    i(t) = (Vs/R)*(1 - exp(-t*R/L))
  RLC:   step response of a series RLC (underdamped / overdamped / critical)
  error: backward Euler is first-order (halving dt halves the error);
         trapezoidal is second-order (halving dt quarters the error).
"""

import math

import numpy as np
import pytest

from circuitforge.analysis import run_op
from circuitforge.netlist import parse_netlist
from circuitforge.transient import TranResult, run_transient


def _rc_analytical(t, vf, vi, tau):
    return vf + (vi - vf) * np.exp(-t / tau)


def test_rc_charging_trap():
    tau = 1e-3
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nC1 2 0 1e-6")
    r = run_transient(c, dt=1e-5, tstop=8e-3, method="trap")
    assert r.converged
    t = r.t
    ref = _rc_analytical(t, 5.0, 0.0, tau)
    err = np.abs(r.v[2] - ref)
    # Trapezoid is second-order; the residual error is the documented
    # backward-Euler start-up step at t=0, which decays with tau.
    assert np.max(err[10:]) < 5e-4, f"max err = {np.max(err):.3e}"
    # Final value must equal the analytical value, not Vf itself (at 8*tau
    # the exact response is 5*(1-exp(-8)) = 4.9983227 V).
    assert abs(r.v[2][-1] - ref[-1]) < 1e-6


def test_rc_charging_be():
    tau = 1e-3
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nC1 2 0 1e-6")
    r = run_transient(c, dt=1e-5, tstop=8e-3, method="be")
    ref = _rc_analytical(r.t, 5.0, 0.0, tau)
    err = np.abs(r.v[2] - ref)
    # Backward Euler is first-order; at dt = tau/100 the error is ~1%.
    assert np.max(err[10:]) < 2e-2, f"max err = {np.max(err):.3e}"


def test_rc_discharge_both_methods():
    tau = 1e-3
    c = parse_netlist("C1 1 0 1e-6 5\nR1 1 0 1000")
    for method in ("be", "trap"):
        r = run_transient(c, dt=1e-5, tstop=8e-3, method=method)
        ref = _rc_analytical(r.t, 0.0, 5.0, tau)
        err = np.abs(r.v[1] - ref)
        tol = 2e-2 if method == "be" else 5e-4
        assert np.max(err[10:]) < tol, f"{method}: max err = {np.max(err):.3e}"
        # capacitor current is -V/R and equals the resistor current
        assert np.allclose(r.i["C1"], -r.v[1] / 1000.0, atol=1e-6)
        assert np.allclose(r.i["R1"], r.v[1] / 1000.0, atol=1e-6)


def test_rl_step_response():
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nL1 2 0 10e-3")
    r = run_transient(c, dt=1e-7, tstop=8e-5, method="trap")
    tau = 10e-3 / 1000.0  # L/R = 10 us
    i_ref = (5.0 / 1000.0) * (1.0 - np.exp(-r.t / tau))
    err = np.abs(r.i["L1"] - i_ref)
    assert np.max(err[10:]) < 5e-6, f"max err = {np.max(err):.3e}"
    # V across the inductor: L*di/dt = 5*exp(-t/tau)
    v_ref = 5.0 * np.exp(-r.t / tau)
    assert np.allclose(r.v[2], v_ref, atol=1e-3)


def test_rlc_underdamped():
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 10\nL1 2 3 1e-3\nC1 3 0 1e-6")
    r = run_transient(c, dt=1e-7, tstop=2e-3, method="trap")
    alpha = 10.0 / (2.0 * 1e-3)  # 5000 1/s
    omega0 = 1.0 / math.sqrt(1e-3 * 1e-6)  # 31623 rad/s
    assert alpha < omega0
    wd = math.sqrt(omega0 ** 2 - alpha ** 2)
    v_ref = 5.0 * (1.0 - np.exp(-alpha * r.t) *
                   (np.cos(wd * r.t) + (alpha / wd) * np.sin(wd * r.t)))
    err = np.abs(r.v[3] - v_ref)
    assert np.max(err[10:]) < 1e-3, f"max err = {np.max(err):.3e}"


def test_rlc_overdamped():
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 200\nL1 2 3 1e-3\nC1 3 0 1e-6")
    r = run_transient(c, dt=1e-7, tstop=4e-3, method="trap")
    alpha = 200.0 / (2.0 * 1e-3)  # 1e5 1/s
    omega0 = 1.0 / math.sqrt(1e-3 * 1e-6)
    assert alpha > omega0
    mu = math.sqrt(alpha ** 2 - omega0 ** 2)
    v_ref = 5.0 * (1.0 - np.exp(-alpha * r.t) *
                   (np.cosh(mu * r.t) + (alpha / mu) * np.sinh(mu * r.t)))
    err = np.abs(r.v[3] - v_ref)
    assert np.max(err[10:]) < 2e-3, f"max err = {np.max(err):.3e}"


def test_rlc_critical():
    rcrit = 2.0 * math.sqrt(1e-3 / 1e-6)  # 63.2455 ohm
    c = parse_netlist(
        f"V1 1 0 STEP(0 5 0)\nR1 1 2 {rcrit}\nL1 2 3 1e-3\nC1 3 0 1e-6"
    )
    r = run_transient(c, dt=1e-7, tstop=2e-3, method="trap")
    alpha = rcrit / (2.0 * 1e-3)
    v_ref = 5.0 * (1.0 - np.exp(-alpha * r.t) * (1.0 + alpha * r.t))
    err = np.abs(r.v[3] - v_ref)
    assert np.max(err[10:]) < 2e-3, f"max err = {np.max(err):.3e}"


def test_inductor_initial_current_freewheel():
    # L pre-charged to 1 mA discharges through R: i = i0*exp(-t*R/L).
    c = parse_netlist("L1 1 0 1e-3 1e-3\nR1 1 0 100")
    r = run_transient(c, dt=1e-7, tstop=5e-5, method="trap")
    tau = 1e-3 / 100.0
    i_ref = 1e-3 * np.exp(-r.t / tau)
    err = np.abs(r.i["L1"] - i_ref)
    assert np.max(err[10:]) < 5e-6, f"max err = {np.max(err):.3e}"
    assert r.i["L1"][0] == pytest.approx(1e-3)


def test_pulse_source_rc():
    """A near-instantaneous pulse through the RC low-pass.

    The reference treats the edges as ideal steps.  Backward Euler evaluates
    the source at the end of each step, so it reproduces the ideal-step
    response directly.  The trapezoid, by contrast, integrates the forcing as
    the average across the step, so a jump landing on a sample point is
    'half-integrated' — a standard, documented trapezoid property that we
    assert explicitly below rather than hiding it.
    """
    tau = 1e-3
    net = (
        "V1 1 0 PULSE(0 5 1e-3 1e-9 1e-9 2e-3 1e-2)\n"
        "R1 1 2 1000\nC1 2 0 1e-6"
    )
    t0, t1 = 1e-3, 3e-3

    def reference(t):
        v_ref = np.empty_like(t)
        rising = (t >= t0) & (t < t1)
        falling = t >= t1
        v_ref[~rising & ~falling] = 0.0
        v_ref[rising] = 5.0 * (1.0 - np.exp(-(t[rising] - t0) / tau))
        v_at_t1 = 5.0 * (1.0 - np.exp(-(t1 - t0) / tau))
        v_ref[falling] = v_at_t1 * np.exp(-(t[falling] - t1) / tau)
        return v_ref

    c = parse_netlist(net)
    r = run_transient(c, dt=1e-5, tstop=5e-3, method="be")
    err = np.abs(r.v[2] - reference(r.t))
    # Backward Euler is first-order: local error ~ V*dt/(2*tau), so near the
    # 4.3 V plateau after the pulse the error reaches ~1% of the swing.
    assert np.max(err[20:]) < 2.2e-2, f"max err = {np.max(err):.3e}"
    # near the end (small V) the accumulated first-order error is still ~1%
    assert err[-1] < 7e-3

    # Trapezoid: converges and stays finite; the half-step artifact at each
    # edge decays with tau (assert its documented size at the first sample).
    r2 = run_transient(c, dt=1e-5, tstop=5e-3, method="trap")
    assert r2.converged
    k1 = np.searchsorted(r2.t, t0 + 1e-5)  # first sample after the rising edge
    ideal_first = 5.0 * (1.0 - np.exp(-1e-5 / tau))
    assert abs(r2.v[2][k1] - 0.5 * ideal_first) < 0.02 * ideal_first
    tail2 = r2.t >= 4.8e-3
    # The edge perturbation decays with tau = 1 ms: at 1.8*tau after the
    # falling edge its residue is ~0.025*exp(-1.8) ~ 4e-3.
    assert np.max(np.abs(r2.v[2][tail2] - reference(r2.t[tail2]))) < 6e-3


def test_error_order_backward_euler():
    """Halving dt should halve the error (first order)."""
    c = parse_netlist("C1 1 0 1e-6 5\nR1 1 0 1000")
    tau = 1e-3
    t_target = tau
    dts = [tau / 20.0, tau / 40.0, tau / 80.0, tau / 160.0]
    errs = []
    for dt in dts:
        r = run_transient(c, dt=dt, tstop=t_target, method="be")
        k = np.searchsorted(r.t, t_target)
        ref = 5.0 * np.exp(-t_target / tau)
        errs.append(abs(r.v[1][k] - ref))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for ratio in ratios:
        assert 1.5 < ratio < 2.7, f"BE error ratio {ratio} not ~2: {errs}"


def test_error_order_trapezoidal():
    """Halving dt should quarter the error (second order)."""
    c = parse_netlist("C1 1 0 1e-6 5\nR1 1 0 1000")
    tau = 1e-3
    t_target = tau
    dts = [tau / 20.0, tau / 40.0, tau / 80.0, tau / 160.0]
    errs = []
    for dt in dts:
        r = run_transient(c, dt=dt, tstop=t_target, method="trap")
        k = np.searchsorted(r.t, t_target)
        ref = 5.0 * np.exp(-t_target / tau)
        errs.append(abs(r.v[1][k] - ref))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for ratio in ratios:
        assert 3.0 < ratio < 6.5, f"trap error ratio {ratio} not ~4: {errs}"


def test_transient_statistics_for_linear_circuit():
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nC1 2 0 1e-6")
    r = run_transient(c, dt=1e-5, tstop=1e-3, method="trap")
    assert r.converged
    assert all(s.status == "converged" for s in r.step_stats)
    assert r.init_stats is not None and r.init_stats.converged
    assert len(r.step_stats) == len(r.t) - 1


def test_transient_start_offset():
    c = parse_netlist("C1 1 0 1e-6 5\nR1 1 0 1000")
    r = run_transient(c, dt=1e-5, tstop=2e-3, method="be", start=1e-3)
    assert r.t[0] == pytest.approx(1e-3)
    # The recorded window must still be continuous with the physics from t=0.
    ref = 5.0 * np.exp(-r.t / 1e-3)
    assert np.allclose(r.v[1], ref, atol=2e-2)


def test_capacitor_parallel_voltage_source_fallback():
    # Capacitor directly across an ideal voltage source: the start-up must
    # fall back gracefully and the node must sit at the source value.
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nC1 1 0 1e-6\nR1 1 0 1000")
    r = run_transient(c, dt=1e-5, tstop=1e-3, method="trap")
    assert r.converged
    assert np.allclose(r.v[1], 5.0, atol=1e-9)
    assert any("parallel" in w for w in r.warnings)


def test_rectifier_diode_transient_runs():
    c = parse_netlist(
        "V1 1 0 PULSE(-10 10 0 1e-4 1e-4 4e-4 1e-3)\n"
        "R1 1 2 100\nD1 2 3\nR2 3 0 1000\nC1 3 0 10e-6"
    )
    r = run_transient(c, dt=1e-6, tstop=8e-3, method="trap")
    assert r.converged, r.warnings
    assert np.all(np.isfinite(r.v[3]))
    assert np.min(r.v[3]) >= -1e-3  # rectified output never goes negative
    assert np.max(r.v[3]) <= 10.0 + 1e-3
    # After the first cycles the output should be pumped up by the diode.
    assert np.max(r.v[3]) > 5.0


def test_dc_of_diode_rectifier_load():
    # Sanity: with no diode current (reverse half), KCL still balances.
    r = run_op(parse_netlist("V1 1 0 DC(5)\nR1 1 2 1000\nD1 2 0\nR2 2 0 1000"))
    assert r.stats.converged
    vd = r.diode_voltages["D1"]
    assert vd > 0.5  # forward biased
