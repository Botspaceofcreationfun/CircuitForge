"""Independent numerical validation of the CircuitForge solver.

For every validation target the reference is computed analytically (or by an
independent method), never by the package's own solver.  Outputs:

  - ``validation_report.txt``  : quantitative results, printed to stdout
  - ``plots/rc_validation.png``, ``plots/error_convergence.png``,
    ``plots/rlc_validation.png``

Run:  python3 scripts/validate.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuitforge.analysis import run_op
from circuitforge.netlist import parse_netlist
from circuitforge.nonlinear import solve_dc
from circuitforge.transient import run_transient

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def max_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def rc_analytical(t, vf, vi, tau):
    return vf + (vi - vf) * np.exp(-np.asarray(t) / tau)


def bisect_diode(vsrc, r, isat=1e-14, n=1.0, vt=25.852e-3):
    lo, hi = 0.0, vsrc
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if (vsrc - mid) / r > isat * (math.exp(mid / (n * vt)) - 1.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fmt_row(*cells, widths=(28, 14, 14, 14, 14)):
    return "  " + "".join(str(c).ljust(w) for c, w in zip(cells, widths))


# ---------------------------------------------------------------------------
# 1. RC charging / discharging vs V(t) = Vf + (Vi-Vf) exp(-t/(R*C))
# ---------------------------------------------------------------------------


def validate_rc():
    lines = ["=" * 78, "1. RC validation against V(t) = Vf + (Vi - Vf)*exp(-t/(R*C))",
             "   R = 1 kOhm, C = 1 uF, tau = 1 ms, tstop = 8 ms"]
    charge_net = "V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nC1 2 0 1e-6"
    discharge_net = "C1 1 0 1e-6 5\nR1 1 0 1000"
    checkpoints = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3]
    for label, net, vf, vi in (
        ("charging  ", charge_net, 5.0, 0.0),
        ("discharging", discharge_net, 0.0, 5.0),
    ):
        for method in ("be", "trap"):
            c = parse_netlist(net)
            r = run_transient(c, dt=1e-5, tstop=8e-3, method=method)
            err = max_err(r.v[1 if vi else 2], rc_analytical(r.t, vf, vi, 1e-3))
            lines.append(fmt_row(f"RC {label} [{method}]",
                                 f"max|err|={err:.3e} V", "converged",
                                 f"steps={len(r.t) - 1}", ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Time-step convergence (numerical error vs dt)
# ---------------------------------------------------------------------------


def validate_order():
    lines = ["=" * 78,
             "2. Numerical error vs time step (RC discharge, evaluated at t = tau)"]
    discharge_net = "C1 1 0 1e-6 5\nR1 1 0 1000"
    tau = 1e-3
    dts = [tau / 20, tau / 40, tau / 80, tau / 160, tau / 320]
    lines.append(fmt_row("dt (s)", "BE err (V)", "trap err (V)", "ratio BE",
                         "ratio trap"))
    be_errs, trap_errs = [], []
    for dt in dts:
        ref = 5.0 * math.exp(-1.0)
        r_be = run_transient(parse_netlist(discharge_net), dt=dt, tstop=tau,
                             method="be")
        r_tp = run_transient(parse_netlist(discharge_net), dt=dt, tstop=tau,
                             method="trap")
        k = np.searchsorted(r_be.t, tau)
        be = abs(r_be.v[1][k] - ref)
        kt = np.searchsorted(r_tp.t, tau)
        tp = abs(r_tp.v[1][kt] - ref)
        be_errs.append(be)
        trap_errs.append(tp)
        lines.append(fmt_row(f"{dt:.3e}", f"{be:.3e}", f"{tp:.3e}", "", ""))
    lines.append(fmt_row("error ratios (previous/next dt)", ""))
    lines.append(fmt_row("  backward Euler (expect ~2, first order)",
                         f"{[f'{be_errs[i]/be_errs[i+1]:.2f}' for i in range(len(be_errs)-1)]}"))
    lines.append(fmt_row("  trapezoidal  (expect ~4, second order)",
                         f"{[f'{trap_errs[i]/trap_errs[i+1]:.2f}' for i in range(len(trap_errs)-1)]}"))
    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.loglog(dts, be_errs, "o-", label="backward Euler (slope ~1)")
        ax.loglog(dts, trap_errs, "s-", label="trapezoidal (slope ~2)")
        # slope-1 and slope-2 reference lines through the first data points
        ax.loglog(dts, be_errs[0] * (dts[0] / np.array(dts)), ":", alpha=0.35,
                  color="C0")
        ax.loglog(dts, trap_errs[0] * (dts[0] / np.array(dts)) ** 2, ":",
                  alpha=0.35, color="C1")
        ax.set_xlabel("time step dt (s)")
        ax.set_ylabel("|error| at t = tau (V)")
        ax.set_title("RC discharge: numerical error vs dt")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "error_convergence.png"), dpi=140)
        plt.close(fig)
    except ImportError:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. RL and RLC vs analytical step responses
# ---------------------------------------------------------------------------


def validate_rl():
    net = "V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nL1 2 0 10e-3"
    r = run_transient(parse_netlist(net), dt=1e-7, tstop=8e-5, method="trap")
    tau = 10e-3 / 1000.0
    i_ref = 5e-3 * (1 - np.exp(-r.t / tau))
    v_ref = 5.0 * np.exp(-r.t / tau)
    lines = ["=" * 78, "3. RL step response (R=1k, L=10mH, tau_L=10 us, dt=100 ns)"]
    lines.append(fmt_row("quantity", "max|err|", "relative", "", ""))
    lines.append(fmt_row("i(L1) vs (Vs/R)(1-exp(-t*R/L))",
                         f"{max_err(r.i['L1'], i_ref):.3e} A", ""))
    lines.append(fmt_row("v(L1) vs Vs*exp(-t*R/L)",
                         f"{max_err(r.v[2], v_ref):.3e} V", ""))
    return "\n".join(lines)


def validate_rlc():
    lines = ["=" * 78, "4. Series RLC step response vs analytical (L=1mH, C=1uF)"]
    omega0 = 1.0 / math.sqrt(1e-3 * 1e-6)
    cases = [
        ("underdamped", "V1 1 0 STEP(0 5 0)\nR1 1 2 10\nL1 2 3 1e-3\nC1 3 0 1e-6",
         10.0, 2e-3),
        ("critical   ", "V1 1 0 STEP(0 5 0)\nR1 1 2 63.24555320336759\n"
                        "L1 2 3 1e-3\nC1 3 0 1e-6", 63.24555320336759, 2e-3),
        ("overdamped ", "V1 1 0 STEP(0 5 0)\nR1 1 2 200\nL1 2 3 1e-3\nC1 3 0 1e-6",
         200.0, 4e-3),
    ]
    for label, net, R, tstop in cases:
        r = run_transient(parse_netlist(net), dt=1e-7, tstop=tstop,
                          method="trap")
        alpha = R / (2e-3)
        t = r.t
        if alpha < omega0:
            wd = math.sqrt(omega0 ** 2 - alpha ** 2)
            ref = 5 * (1 - np.exp(-alpha * t) *
                       (np.cos(wd * t) + (alpha / wd) * np.sin(wd * t)))
        elif alpha > omega0:
            mu = math.sqrt(alpha ** 2 - omega0 ** 2)
            ref = 5 * (1 - np.exp(-alpha * t) *
                       (np.cosh(mu * t) + (alpha / mu) * np.sinh(mu * t)))
        else:
            ref = 5 * (1 - np.exp(-alpha * t) * (1 + alpha * t))
        err = max_err(r.v[3], ref)
        lines.append(fmt_row(f"RLC {label} (R={R:g})",
                             f"max|err|={err:.3e} V", "converged", "", ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Resistive networks: independent KCL / reference nodal solver
# ---------------------------------------------------------------------------


def validate_resistive():
    lines = ["=" * 78,
             "5. Resistive networks vs independent nodal-analysis reference"]
    import sys as _s
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tests.reference_nodal import nodal_analysis_dc  # noqa: E402

    rng = np.random.default_rng(42)
    worst = 0.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        lines.append(fmt_row(f"  chain seed={seed}",
                             f"  R in [100, 5000] ohm, 6 nodes", "", "", ""))
        text = ["V1 1 0 DC(10)"]
        prev = 1
        for k in range(2, 7):
            rr = float(rng.uniform(100, 5000))
            text.append(f"R{k - 1} {prev} {k} {rr}")
            prev = k
        text.append(f"R6 {prev} 0 {float(rng.uniform(100, 5000))}")
        c = parse_netlist("\n".join(text))
        ours = run_op(c)
        res = [(comp.n1, comp.n2, comp.value) for comp in c.resistors]
        vs = [(comp.n1, comp.n2, comp.waveform.value(0.0))
              for comp in c.voltage_sources]
        ref_v, _ = nodal_analysis_dc(c.nodes, res, vs, [])
        d = max(abs(ours.node_voltages[n] - ref_v[n]) for n in ref_v)
        worst = max(worst, d)
    lines.append(fmt_row("worst node-voltage deviation across 5 networks",
                         f"{worst:.3e} V", "", "", ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Diode operating point vs bisection reference
# ---------------------------------------------------------------------------


def validate_diode():
    lines = ["=" * 78, "6. Diode DC bias vs bisection reference (Is=1e-14, N=1)"]
    c = parse_netlist("V1 1 0 DC(5)\nR1 1 2 1000\nD1 2 0")
    res = solve_dc(c)
    vd_ref = bisect_diode(5.0, 1000.0)
    vd = res.diode_voltages["D1"]
    lines.append(fmt_row("vd(D1) solver / reference / deviation",
                         f"{vd:.9f}", f"{vd_ref:.9f}", f"{abs(vd - vd_ref):.2e} V",
                         f"iters={res.stats.iterations}"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    out = []
    for fn in (validate_rc, validate_order, validate_rl, validate_rlc,
               validate_resistive, validate_diode):
        out.append(fn())
    text = "\n\n".join(out) + "\n" + "=" * 78 + "\n"
    print(text)
    report = os.path.join(PLOT_DIR, "..", "validation_report.txt")
    report = os.path.normpath(report)
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w") as fh:
        fh.write(text)
    print(f"report written: {report}")


if __name__ == "__main__":
    main()
