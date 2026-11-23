"""Newton-Raphson solution of nonlinear (diode) circuits.

The diode is the only nonlinear element: :math:`I_d = I_S(\\exp(V_d/(N V_T))-1)`
with :math:`V_T = kT/q`.  At every Newton iteration the diode is replaced by
its tangent-line linearization

.. math:: I_d \\approx g\\,V_d + c, \\qquad g = \\frac{dI_d}{dV_d}, \\quad
          c = I_{d0} - g V_{d0},

which is stamped into the MNA matrix, the linear system is solved, and the
solution is updated.  Convergence is judged on the Newton step size and on the
KCL residual.  Voltage changes across every diode are damped to at most
``voltage_limit`` volts per iteration (a classic measure for exponential
devices) so the iteration cannot blow up on the exponential term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import mna
from .netlist import Circuit, Diode, NetlistError

# Default thermal voltage at 300 K (kT/q).
DEFAULT_VT = Diode.thermal_voltage(300.0)


@dataclass
class NRStats:
    """Convergence diagnostics for one Newton-Raphson run.

    Attributes
    ----------
    converged     : True if the Newton loop met its criteria
    iterations    : number of Newton iterations performed
    final_step    : infinity-norm of the last Newton step (relative)
    final_residual: infinity-norm of the KCL residual (amperes)
    status        : one of 'converged', 'max_iterations', 'diverged',
                    'residual_large'
    message       : human-readable summary
    """

    converged: bool
    iterations: int
    final_step: float
    final_residual: float
    status: str
    message: str


@dataclass
class NonLinearResult:
    x: np.ndarray
    stats: NRStats
    node_rows: Dict[int, int]
    vs_rows: Dict[str, int]
    diode_voltages: Dict[str, float] = field(default_factory=dict)


def _diode_residual_injection(
    circuit: Circuit, x: np.ndarray, node_rows: Dict[int, int], vt: float
) -> np.ndarray:
    """KCL residual contribution of all diodes: r = A0 x + I_diodes - z0.

    Returns a vector aligned with the node-voltage rows (all others zeroed).
    """
    n = len(x)
    res = np.zeros(n)
    for d in circuit.diodes:
        vd = _diode_voltage(d, x, node_rows)
        i = d.current(vd, vt)  # flows anode -> cathode
        if d.anode != 0:
            res[node_rows[d.anode]] += i
        if d.cathode != 0:
            res[node_rows[d.cathode]] -= i
    return res


def _diode_voltage(d: Diode, x: np.ndarray, node_rows: Dict[int, int]) -> float:
    va = x[node_rows[d.anode]] if d.anode != 0 else 0.0
    vc = x[node_rows[d.cathode]] if d.cathode != 0 else 0.0
    return float(va - vc)


def newton_raphson(
    circuit: Circuit,
    base_builder: Callable[[], mna.MnaSystem],
    *,
    x0: Optional[np.ndarray] = None,
    vt: float = DEFAULT_VT,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    res_tol: float = 1e-7,
    max_iterations: int = 100,
    voltage_limit: float = 0.1,
) -> NonLinearResult:
    """Solve a nonlinear circuit by damped Newton-Raphson.

    Parameters
    ----------
    circuit       : the circuit (its diodes provide the nonlinearity)
    base_builder  : zero-argument callable returning the linear (diode-free)
                    MNA system for the current operating point / time step
    x0            : initial guess for the full unknown vector
    vt            : thermal voltage
    atol, rtol    : absolute/relative tolerance on the Newton step
    res_tol       : tolerance on the KCL residual (amperes)
    max_iterations: safety cap
    voltage_limit : max change of any diode voltage per iteration (volts)
    """
    system0 = base_builder()
    A0, z0 = system0.A, system0.z
    node_rows, vs_rows = system0.node_rows, system0.vs_rows
    n = A0.shape[0]

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float).copy()
        if x.shape[0] != n:
            raise NetlistError("initial guess has wrong length for this system")

    # Linear circuits need exactly one solve (Newton converges in one step).
    if not circuit.diodes:
        try:
            x = np.linalg.solve(A0, z0)
        except np.linalg.LinAlgError as exc:
            raise mna.MNAError(mna.diagnose_singular(system0)) from exc
        res = float(np.linalg.norm(A0 @ x - z0, np.inf)) if n else 0.0
        stats = NRStats(
            converged=True,
            iterations=1,
            final_step=float("nan"),
            final_residual=res,
            status="converged",
            message=(
                "linear circuit solved directly "
                f"(|KCL|inf={res:.3e} A)"
            ),
        )
        return NonLinearResult(x, stats, node_rows, vs_rows, {})

    iter_count = 0
    status = "max_iterations"
    final_step = float("inf")
    final_res = float("inf")

    for _ in range(max_iterations):
        # Build the linearized system: fixed part + diode tangents at x.
        A = A0.copy()
        z = z0.copy()
        for d in circuit.diodes:
            vd = _diode_voltage(d, x, node_rows)
            g = d.conductance(vd, vt)
            i0 = d.current(vd, vt)
            c = i0 - g * vd
            mna.stamp_diode_linearization(
                A, z, node_rows, d.anode, d.cathode, g, c
            )
        try:
            x_new = np.linalg.solve(A, z)
        except np.linalg.LinAlgError as exc:
            raise mna.MNAError(
                mna.diagnose_singular(
                    mna.MnaSystem(A, z, node_rows, vs_rows)
                )
            ) from exc

        if not np.all(np.isfinite(x_new)):
            status = "diverged"
            break

        # Damp the step for every diode whose voltage moved too far.
        step = x_new - x
        scale = 1.0
        for d in circuit.diodes:
            dv = _diode_voltage(d, x_new, node_rows) - _diode_voltage(
                d, x, node_rows
            )
            if abs(dv) > voltage_limit:
                scale = min(scale, voltage_limit / abs(dv))
        if scale < 1.0:
            x_new = x + scale * step

        iter_count += 1
        step_norm = float(np.linalg.norm(x_new - x, np.inf))
        final_step = step_norm
        x = x_new

        if step_norm <= atol + rtol * float(np.linalg.norm(x, np.inf)):
            status = "converged"
            break
    else:
        status = "max_iterations"

    # Residual check (KCL balance in amperes).
    res = A0 @ x + _diode_residual_injection(circuit, x, node_rows, vt) - z0
    final_res = float(np.linalg.norm(res[: len(node_rows)], np.inf))

    if status == "converged" and final_res > res_tol:
        status = "residual_large"

    converged = status == "converged"
    message = _format_message(status, iter_count, final_step, final_res)
    stats = NRStats(
        converged=converged,
        iterations=iter_count,
        final_step=final_step,
        final_residual=final_res,
        status=status,
        message=message,
    )
    diode_voltages = {
        d.name: _diode_voltage(d, x, node_rows) for d in circuit.diodes
    }
    return NonLinearResult(x, stats, node_rows, vs_rows, diode_voltages)


def _format_message(status: str, iters: int, step: float, res: float) -> str:
    if status == "converged":
        return (
            f"Newton-Raphson converged in {iters} iterations "
            f"(|dx|inf={step:.3e}, |KCL|inf={res:.3e} A)"
        )
    if status == "max_iterations":
        return (
            f"Newton-Raphson did NOT converge within {iters} iterations "
            f"(|dx|inf={step:.3e}, |KCL|inf={res:.3e} A). "
            "The last iterate is returned; treat results with caution."
        )
    if status == "diverged":
        return (
            "Newton-Raphson diverged (non-finite values). "
            "Check the netlist and the diode model parameters."
        )
    if status == "residual_large":
        return (
            f"Newton step converged but KCL residual is large "
            f"(|KCL|inf={res:.3e} A > tolerance). "
            "The matrix may be ill-conditioned; results are unreliable."
        )
    return status  # pragma: no cover


def solve_dc(
    circuit: Circuit,
    *,
    t: float = 0.0,
    vt: float = DEFAULT_VT,
    **kwargs,
) -> NonLinearResult:
    """DC operating-point analysis (Newton-Raphson; linear circuits converge
    in a single iteration)."""
    return newton_raphson(
        circuit,
        lambda: mna.build_dc_op_system(circuit, t),
        vt=vt,
        **kwargs,
    )
