# CircuitForge — Verification Report

*Date: 2026-10-27 (UTC). Environment: Linux, Python 3.12.11, numpy 1.26.4,
matplotlib 3.11.0, pytest 9.2.0.*

This report lists what was implemented, what was verified and how, the
exact test results, known issues, and unfinished work. Everything below was
actually executed in the delivery environment; no result is asserted from
assumption.

## 1. Implemented features

| Area | Feature | Status |
|---|---|---|
| Netlist | R, C (with v0), L (with i0), V, I, D components; ground aliases; `DC/STEP/PULSE` waveforms; `.op/.tran/.model/.plot/.export` directives; comments; line-numbered errors | implemented |
| DC | MNA assembly (node-voltage unknowns + voltage-source current unknowns); capacitor open / inductor short; supernode handling via source constraint rows; branch currents with documented signs | implemented |
| Diagnostics | floating-node detection (connectivity + pivot analysis); rank-deficient (voltage-loop) message; condition-number reporting and warnings | implemented |
| Nonlinear | damped Newton-Raphson for the diode; per-iteration voltage limiting; `NRStats` (iterations, step, KCL residual, status) always reported; linear fast path | implemented |
| Transient | backward Euler and trapezoidal companion models (documented equations); first-step-BE start-up; start-up operating point from `v0`/`i0`; cap-across-ideal-source fallback; per-step convergence status; ill-conditioning warnings | implemented |
| UI / output | CLI (`run`, `edit`, `list`); interactive text-based netlist editor; matplotlib plots (voltage + current subplots); CSV export with SPICE-style headers | implemented |
| Docs | README, this report, technical document, validation script | delivered |

## 2. Verification executed

### 2.1 Automated tests

Command: `python3 -m pytest -q` from the project root.

**Result: 76 passed, 0 failed** in ≈ 6.6 s.

Coverage by file:

- `tests/test_netlist.py` (16) — parsing, ground aliases, waveforms
  (spaces/commas), pulse semantics, `.model` anywhere, duplicate/unknown/
  invalid inputs, line numbers in errors.
- `tests/test_dc.py` (13) — divider and multi-resistor KCL hand-checks,
  current sources, supernodes, inductor short, capacitor open, floating-node
  diagnostics (message and exact node list), voltage-loop singularity,
  branch-current signs, condition reporting.
- `tests/test_nonlinear.py` (8) — diode bias vs bisection (1e-6 V),
  current-source-into-diode (hard start), model parameters, two diodes in
  series, reverse bias, convergence diagnostics presence.
- `tests/test_transient.py` (17) — RC charge/discharge (both methods) vs
  analytical; RL step; RLC under/over/critical damping; inductor freewheel;
  pulse-source RC (BE exact, TRAP half-step artifact asserted explicitly);
  **error-order scaling (BE ratio ≈ 2, TRAP ratio ≈ 4)**; start offset;
  cap-||-V-source fallback; diode rectifier transient.
- `tests/test_crosscheck.py` (7) — random resistive networks vs the
  independent nodal solver in `tests/reference_nodal.py`.
- `tests/test_cli.py` (10) — CLI runs, CSV contents (headers, sample count,
  values vs analytical), missing/invalid inputs, override flags, editor
  basics and `.save`.

### 2.2 Independent validation (`python3 scripts/validate.py`)

| Check | Result |
|---|---|
| RC charging, BE, dt = τ/100 | max err 9.16e-3 V (first order, expected) |
| RC charging, TRAP, dt = τ/100 | max err 2.46e-4 V (start-up-BE residual, decays) |
| RC discharging, same | identical numbers (symmetry of the model) |
| Error order, BE (ratios dt→dt/2) | 1.98, 1.99, 1.99, 2.00 → first order confirmed |
| Error order, TRAP | 3.94, 3.97, 3.99, 3.99 → second order confirmed |
| RL step, i(L1) | max err 2.46e-7 A (dt = τ/100) |
| RL step, v(L1) | max err 2.46e-4 V |
| RLC underdamped (R = 10 Ω) | max err 2.50e-5 V |
| RLC critically damped | max err 2.47e-5 V |
| RLC overdamped (R = 200 Ω) | max err 2.42e-5 V |
| 5 random resistive chains vs reference nodal solver | worst deviation 0.000e+00 V |
| Diode bias vs bisection | 0.692543627 vs 0.692543633 → 5.7e-9 V, 12 Newton iterations |

### 2.3 End-to-end example runs

All eight `examples/*.netlist` run successfully through the CLI. Transient
examples produce PNG plots and CSVs (`output/`):

- `rc_charge` — v(2) charges to 4.9983227 V at 8 ms (= 5(1−e⁻⁸), exact to
  2.8e-8 V); CSV verified against the analytical formula.
- `rc_discharge` — v(1) = 1.7453e-3 V at 8 ms (BE; matches the BE
  recurrence 5·(1/1.01)^800 = 1.7455e-3).
- `rl_step` — i(L1) = 4.99834e-3 A at 80 µs (≈ 5e-3·(1−e⁻⁸)).
- `rlc_underdamped` — settles at 5.000 V with correct ringing.
- `rectifier` — output pumped above 7 V with correct polarity; all steps
  converged.
- `floating_node` — CLI exits 1 with the floating-node diagnostic listing
  nodes 2 and 3.
- `divider`, `diode_dc` — exact DC results (5.000 V, 0.6925436 V).

## 3. Known issues / limitations (verified behaviour)

1. **Trapezoidal half-step at discontinuities** — the first TRAP sample
   after a pulse edge is ≈ half of the ideal-step response (documented
   property, asserted in `test_pulse_source_rc`). Use `method=be` for
   pulse-heavy circuits.
2. **Start-up BE step** leaves a first-order residue in TRAP runs that
   decays with the circuit time constant (measured 2.5e-4 V for
   dt = τ/100).
3. **Capacitor directly across an ideal voltage source** — start-up falls
   back by dropping the capacitor stand-in and warns; the (undefined) Dirac
   current of the ideal model is not represented.
4. **Inductor–current-source loops** have no well-posed initial state and
   are reported as errors, not silently solved.
5. **Fixed time step only** — no adaptive stepping; accuracy for stiff
   circuits requires a sufficiently small `dt` (BE remains stable).
6. **Diode model** is the ideal exponential law (no breakdown, no series
   resistance, fixed 300 K thermal voltage); the exponential is linear-
   extended above `40·N·V_T`.
7. **Condition thresholds** (1e12 / 1e14) are heuristic warning levels, not
   hard error bounds.
8. **Dense solver** — fine for the intended small systems; large networks
   would need sparse storage.

## 4. Unfinished work / not attempted

- Adaptive time-stepping and local-truncation-error control (highest-value
  next step; the architecture supports it — companion models are already
  per-step and stateless).
- Sparse direct solver for large circuits.
- Additional device models (BJT/MOSFET Ebers-Moll / Shichman-Hodges).
- Graphical (drag-and-drop) circuit editor; the text editor is complete.
- `.dc` sweep directive, AC/small-signal analysis, frequency domain.
- Temperature handling (`V_T(T)` sweep).

## 5. Reproducibility

From a clean checkout in the delivery environment:

```bash
cd CircuitForge
python3 -m pytest -q            # 76 passed
python3 scripts/validate.py     # report + plots (scripts/plots/)
for f in examples/*.netlist; do python3 -m circuitforge run "$f"; done
```

The exact environment versions used: Python 3.12.11 (CPython),
numpy 1.26.4, scipy 1.17.1 (test-time reference only), matplotlib 3.11.0,
pytest 9.2.0.
