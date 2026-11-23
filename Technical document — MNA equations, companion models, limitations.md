# CircuitForge — Technical Document

*Architecture, equations, algorithms, assumptions, and limitations.*

---

## 1. Objective and scope

CircuitForge is a from-scratch circuit simulator whose numerical core is
**modified nodal analysis (MNA)**. It solves:

1. **DC operating points** of linear networks and of networks containing an
   idealized exponential diode (Newton-Raphson with damping and convergence
   diagnostics).
2. **Transient responses** of circuits with resistors, capacitors,
   inductors, independent sources and diodes, using companion models with
   backward Euler or trapezoidal integration and a fixed time step.

The project deliberately stops at ideal circuit models. No parasitic
effects, semiconductor devices beyond the diode, or electro-mechanical
coupling are modelled; nothing is claimed about transistor-level behaviour.

## 2. Component and waveform model

| Element | Model |
|---|---|
| Resistor | `I = G·(V(n+) − V(n−))`, `G = 1/R` |
| Capacitor | `I = C·dV/dt` (transient); open at DC |
| Inductor | `V = L·dI/dt` (transient); short at DC |
| Voltage source | `V(n+) − V(n−) = v_s(t)` |
| Current source | forces `i_s(t)` from `n+` to `n−` |
| Diode | `I_d = I_S·(exp(V_d/(N·V_T)) − 1)`, `V_T = kT/q` at 300 K (25.852 mV); default `I_S = 1e-14 A`, `N = 1.0` |

Waveforms: `DC(x)`, `STEP(v0,v1,t0)`, `PULSE(v0,v1,td,tr,tf,pw,per)`.
Ground is node 0; the ground row is never assembled (reference handling).

## 3. Modified nodal analysis — the DC system

Unknown vector `x`:

- `V_1 … V_N` — node voltages of all non-ground nodes;
- `I_k` — one branch-current unknown per independent voltage source (and,
  at DC, per inductor short; at start-up, per capacitor initial source).

The linear system `A·x = z` is assembled by stamping:

| Element | Stamps |
|---|---|
| Resistor `G` between `i,j` | `+G` on `(i,i),(j,j)`; `−G` on `(i,j),(j,i)` |
| Current source `J` flowing `i→j` | `z[i] −= J`, `z[j] += J` |
| Voltage source `Vs` between `a,b`, unknown `I` at column `k` | row `a`: `A[a,k]+=1`; row `b`: `A[b,k]−=1`; constraint row `k`: `A[k,a]+=1`, `A[k,b]−=1`, `z[k]=Vs` |
| Diode linearized at `V_d` with slope `g`, intercept `c` | like a resistor `g` plus `z[a]−=c`, `z[c]+=c` (see §5) |

KCL sign convention: at each node the sum of currents *leaving* the node
equals the current injected by sources (`A·V = z`). Branch currents of
voltage-like unknowns are signed from `n+` to `n−` through the element.

**Capacitors are open** at DC (no stamp), **inductors are shorted** (stamped
as a 0 V source with a current unknown).

## 4. Singularity and floating-node diagnostics

Two complementary checks:

1. **Connectivity check** (used by the diagnostics): build the undirected
   graph of conductive/constraint edges (resistors, voltage sources,
   inductor shorts, diodes; at DC capacitors are open, current sources are
   never edges). Every non-ground node must be reachable from ground.
   Unreachable nodes are reported as *floating* with a corrective message.
2. **Matrix inspection**: on a `LinAlgError`, Gaussian elimination with
   partial pivoting on the node block identifies zero pivots (floating
   nodes); if none exist, the message reports a rank deficiency and the
   usual cause (voltage-source loop / conflicting constraints).

The condition number is always reported, and warnings are emitted above
`1e12` (op) / `1e14` (transient step).

## 5. Nonlinear analysis — Newton-Raphson for the diode

The diode is the only nonlinear element. Around the current iterate `V_d`:

```
I_d(V) ≈ I_d0 + g·(V − V_d0),   g = dI/dV = I_S/(N·V_T)·exp(V_d/(N·V_T))
```

which is stamped as conductance `g` plus intercept `c = I_d0 − g·V_d0`.
Each Newton iteration re-stamps all diodes, solves the linear system, and
updates. **Damping**: the change of every diode voltage is capped at
`voltage_limit = 0.1 V` per iteration (classic measure for exponential
devices — a raw Newton step from zero bias can be ~1e9 V). The exponential
argument is clamped at `x = 40` with a linear continuation to avoid float
overflow (the model is documented as *not* modelling breakdown or the
high-injection regime).

Convergence diagnostics (`NRStats`) are returned and printed rather than
silently discarded: iterations, final Newton step, final KCL residual
(amperes), and one of the statuses `converged`, `max_iterations`,
`diverged`, `residual_large`. Linear circuits solve in exactly one
factorization.

## 6. Transient analysis — companion models

### 6.1 Discretizations

**Backward Euler** (BE), `method=be`:

```
C:  I_C(t) = G_C·V_C(t) − G_C·V_C(t−dt),     G_C = C/dt
L:  I_L(t) = I_L(t−dt) + G_L·V_L(t),          G_L = dt/L
```

**Trapezoidal** (TRAP), `method=trap`:

```
C:  I_C(t) = G_C·V_C(t) − G_C·V_C(t−dt) − I_C(t−dt),   G_C = 2C/dt
L:  I_L(t) = I_L(t−dt) + G_L·(V_L(t) + V_L(t−dt)),     G_L = dt/(2L)
```

Every dynamic element is stamped as a conductance `G` in parallel with an
independent current source `J`, so the transient system has the same shape
as the DC system; inductors need **no extra unknown** (only voltage sources
keep branch-current unknowns). Diode nonlinearity is handled with the same
Newton loop inside each step (continuation from the previous step's
solution).

### 6.2 Start-up and initial conditions

At `t = 0` an operating point is solved with **capacitors replaced by
voltage sources of value `v0`** and **inductors by current sources of value
`i0`** — this yields the consistent initial node voltages *and* the initial
capacitor currents. Two documented edges:

- A capacitor directly in parallel with an ideal voltage source makes the
  start-up system singular (two voltage sources pinning the same pair). The
  solver drops the capacitor stand-in (its voltage is set by the source
  anyway) and continues; a warning is emitted. This matches standard
  simulator behaviour for this degenerate but common topology.
- Pure inductor–current-source loops have no well-posed initial state and
  are reported as an initial-condition error.

**First step is always backward Euler**, even for `method=trap` — the
standard measure that (a) avoids the trapezoid's half-step/ringing right at
source discontinuities, and (b) removes the need for the capacitor current
at `t = 0` in the degenerate topologies above. The remaining TRAP steps are
second-order.

### 6.3 Numerical properties

- **BE is L-stable** (unconditionally stable for the test equation, and
  damped); first-order — halving `dt` halves the error (measured ratio
  ≈ 1.98–2.00).
- **TRAP is A-stable** (no stability limit) and second-order — halving `dt`
  quarters the error (measured ratio ≈ 3.94–3.99).
- TRAP does **not** damp oscillations on the imaginary axis and integrates a
  step discontinuity as the step average, which is why the first sample
  after a pulse edge is about half of the ideal-step response. This is a
  documented property, asserted in the tests and described in the README's
  troubleshooting section; `method=be` is recommended for pulse-heavy
  circuits unless ringing-free second-order integration matters.
- Fixed time step: `dt` is global and constant; the final partial interval
  (when `tstop` is not a multiple of `dt`) uses its true length.
- Per-step ill-conditioning (cond > 1e14) produces a warning.

## 7. Validation methodology

Everything in `scripts/validate.py` and `tests/` compares the solver
against **independent** references:

- RC charging/discharging vs `V(t) = V_f + (V_i − V_f)·exp(−t/(R·C))`;
- RL step response vs `i(t) = (Vs/R)·(1 − exp(−t·R/L))`;
- RLC (under/over/critically damped) step response vs the closed forms
  `5[1 − e^(−αt)(cos ω_d t + (α/ω_d) sin ω_d t)]` and their cosh/sinh and
  critical counterparts;
- resistive networks vs an independent nodal-analysis implementation
  (`tests/reference_nodal.py`) on seeded random networks;
- diode bias vs plain bisection of the transcendental equation;
- error order: log-log slope of error vs `dt` (BE ≈ 1, TRAP ≈ 2);
- KCL checks at every node on hand-computed dividers.

Results are summarized in `docs/verification_report.md` and
`scripts/validation_report.txt`.

## 8. Implementation notes

- Dense `numpy.linalg.solve` (LU with partial pivoting); systems are small
  (≤ a few hundred unknowns), so a sparse direct solver is unnecessary.
- Sign conventions are documented at the top of `circuitforge/mna.py` and
  verified by KCL invariants in the tests.
- Netlist parser reports file/line on every error; component names must be
  unique; node numbers are non-negative integers.
- `analysis.run_netlist` executes the directives of a file; the CLI wraps
  it. Plotting is lazily imported so the core works without matplotlib.

## 9. Assumptions and limitations

1. **Ideal elements only.** No ESR/ESL, no saturation, no parasitic
   coupling, no transistor-level semiconductor behaviour. The diode is the
   exponential law with fixed `V_T` at 300 K; reverse breakdown and series
   resistance are not modelled, and the exponential is linear-extended for
   `V_d > 40·N·V_T` (~1.03 V).
2. **Fixed time step.** No adaptive stepping, no LTE estimator. For stiff
   circuits (fast and slow time constants together) the step must suit the
   fastest mode; BE remains stable but accuracy needs `dt` small enough.
3. **Ideal sources are truly ideal**: a capacitor directly across a voltage
   source yields an impulsive current in continuous time; the discretized
   solver handles it gracefully (start-up fallback, first-step BE) but the
   reported current is not the (undefined) Dirac pulse.
4. **Initial conditions** are given by `v0`/`i0` or are zero; no "UIC"-style
   iterative consistent-IC refinement beyond the described start-up solve.
5. **Newton-Raphson** may need several tens of iterations when diodes start
   from zero bias (the voltage damping walks the exponential); status and
   residual are always reported. `voltage_limit`, tolerances and iteration
   caps are configurable.
6. **Temperature** is fixed at 300 K (`V_T = 25.852 mV`); no `.temp`
   directive.
7. The **connectivity diagnostic** treats diodes as conductive edges; a
   diode-only path has a tiny but non-zero conductance, so such nodes are
   solvable but potentially ill-conditioned (warned).
8. Numerics are double precision; condition warnings above 1e12/1e14 are
   heuristic thresholds.

## 10. Suggested next steps

1. **Adaptive time stepping** with a local-truncation-error estimator
   (e.g. trap–BE comparison) — the highest-value improvement for stiff
   circuits.
2. **Sparse linear algebra** (scipy.sparse LU) for large networks.
3. **More nonlinear devices**: e.g. the Shockley diode with series
   resistance, or an Ebers-Moll BJT — each requires only a new nonlinear
   stamp in the Newton loop.
4. A **graphical netlist editor** (drag-and-drop components) driving the
   same solver core.
5. `tran` output of element voltages and charge conservation checks per
   step (`Q_C` bookkeeping) as an extra invariant.
