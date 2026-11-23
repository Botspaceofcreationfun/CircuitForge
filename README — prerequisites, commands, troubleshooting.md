# CircuitForge

A small circuit-analysis application built from first principles on
**modified nodal analysis (MNA)** — the numerical core is implemented here,
not wrapped around a commercial simulator.

It supports DC operating-point analysis (with an idealized diode model solved
by damped Newton-Raphson), transient simulation (backward Euler or
trapezoidal companion models), a documented netlist format, an interactive
text-based netlist editor, matplotlib plots, and CSV export.

```
R1 1 2 1000
C1 2 0 1e-6
V1 1 0 STEP(0 5 0)
.tran dt=1e-5 tstop=8e-3 method=trap
.plot v(2) i(R1)
```

## 1. Prerequisites

- Python ≥ 3.9
- numpy ≥ 1.21
- matplotlib ≥ 3.5 (plots; the CLI degrades to text tables + CSV without it)
- pytest ≥ 7 (tests only)

Verified in the delivery environment: Python 3.12.11, numpy 1.26.4,
matplotlib 3.11.0, pytest 9.2.0 (see `docs/verification_report.md`).

## 2. Installation

No build step is required; the package is plain Python.

```bash
cd CircuitForge
pip install -r requirements.txt     # numpy, matplotlib (+ pytest for tests)
python3 -m circuitforge --version   # smoke test
```

Optional: `pip install -e .` installs the `circuitforge` console script.

## 3. Quick start

Run a bundled example:

```bash
cd CircuitForge

# DC operating point of a resistive divider
python3 -m circuitforge run examples/divider.netlist

# RC charging (transient), writes PNG plots and rc_charge.csv
python3 -m circuitforge run examples/rc_charge.netlist

# Every example
python3 -m circuitforge list
for f in examples/*.netlist; do python3 -m circuitforge run "$f"; done
```

Overrides and options:

```bash
python3 -m circuitforge run examples/rc_charge.netlist \
    --dt 1e-6 --tstop 2e-3 --method be --plot-dir out --no-plot
```

Interactive text-based netlist editor:

```bash
python3 -m circuitforge edit            # start with an empty buffer
python3 -m circuitforge edit my.netlist # edit an existing netlist
```

In the editor, type component/directive lines, then `.run` to analyse,
`.plot` to write PNGs, `.export out.csv` to export, `.save f` to keep the
buffer. Type `.help` for the full command list. Netlist directives (`.op`,
`.tran`, `.model`, `.plot`, `.export`) are buffer lines, not editor commands.

## 4. Netlist format

Line oriented. `#` starts a comment line; `;` starts a trailing comment.
Tokens are whitespace separated; function arguments may contain spaces
(e.g. `STEP(0 5 0)` is one token).

| Line | Meaning |
|---|---|
| `R1 1 2 1000` | resistor 1 kΩ between nodes 1 and 2 |
| `C1 2 0 1e-6 [v0]` | capacitor 1 µF, optional initial voltage `v0` |
| `L1 2 0 1e-3 [i0]` | inductor 1 mH, optional initial current `i0` (n+ → n−) |
| `V1 1 0 STEP(0 5 0)` | voltage source, `v(1) − v(0)` |
| `I1 0 1 1e-3` | current source, forces 1 mA from node 0 to node 1 |
| `D1 2 0 [model]` | diode, anode 2 → cathode 0 |
| `.model D IS=1e-14 N=1.0` | diode model parameters |
| `.op` | run DC operating point |
| `.tran dt=.. tstop=.. [start=..] [method=be\|trap]` | transient |
| `.plot v(2) i(R1)` | variables to plot / export |
| `.export out.csv` | write CSV of the last transient |

Ground is node `0` (aliases `gnd`, `GND`, `ground`). All other nodes are
non-negative integers. Duplicate names, invalid values and unknown elements
are rejected with a line number.

**Waveforms**

- `DC(x)` / bare number — constant `x`
- `STEP(v0, v1, t0)` — `v0` for `t < t0`, `v1` for `t >= t0`
- `PULSE(v0, v1, td, tr, tf, pw, per)` — SPICE-style periodic pulse:
  base `v0`, peak `v1`, delay `td`, rise `tr`, on-time `pw`, fall `tf`,
  period `per`.

## 5. Examples

| File | What it demonstrates |
|---|---|
| `divider.netlist` | DC op point, KCL, supernode behaviour |
| `rc_charge.netlist` | RC charging, analytical comparison, CSV/plots |
| `rc_discharge.netlist` | RC discharging with initial capacitor voltage |
| `rl_step.netlist` | RL step response |
| `rlc_underdamped.netlist` | series RLC, underdamped ringing |
| `diode_dc.netlist` | diode bias point (Newton-Raphson) |
| `rectifier.netlist` | half-wave rectifier — nonlinear transient |
| `floating_node.netlist` | intentionally broken — shows the diagnostics |

Run the validation script for a quantitative report against analytical
solutions and error-order measurements:

```bash
python3 scripts/validate.py     # prints + writes scripts/validation_report.txt
```

Run the automated test suite:

```bash
python3 -m pytest -q            # 76 tests, all pass
```

## 6. Using the Python API

```python
from circuitforge.netlist import parse_netlist
from circuitforge.transient import run_transient
from circuitforge.analysis import run_op, export_csv
import numpy as np

circuit = parse_netlist("""
    V1 1 0 STEP(0 5 0)
    R1 1 2 1000
    C1 2 0 1e-6
    .tran dt=1e-5 tstop=4e-3
""")

op = run_op(circuit)                       # DC op point
tran = run_transient(circuit, dt=1e-5, tstop=4e-3, method="trap")
print(tran.v[2])                           # capacitor voltage samples
print(tran.i["R1"], tran.converged, tran.warnings)

export_csv(tran, ["v(2)", "i(R1)"], "out.csv")
```

## 7. Architecture (summary)

```
circuitforge/
  netlist.py     netlist parser + component/waveform data model
  mna.py         MNA matrix assembly, linear solve, singularity diagnostics
  nonlinear.py   damped Newton-Raphson for diode circuits + diagnostics
  transient.py   companion-model transient integration (BE / trapezoidal)
  analysis.py    DC op point; directive execution for a netlist
  plotting.py    matplotlib plots of node voltages / branch currents
  export.py      CSV export
  cli.py         CLI + interactive netlist editor
```

See `docs/technical_document.md` for the full equations (MNA stamp table,
companion models, Newton linearization), numerical properties, and
assumptions.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `singular MNA matrix: node(s) N are floating` | Node N has no conductive path to ground. Add a resistor/voltage-source path or fix the netlist (see `examples/floating_node.netlist`). |
| `singular MNA matrix: rank ... ` | Voltage-source loop or conflicting constraints (two voltage sources in parallel with different values). |
| `cannot find a consistent initial state at t = 0` | Initial-condition conflict (e.g. a capacitor pre-charged to a value that clashes with a parallel ideal source, or an inductor–current-source loop). See the technical document. |
| `Newton-Raphson did NOT converge` | Rare with the built-in voltage damping; check the diode model parameters (`IS > 0`, `N > 0`) and the circuit. The last iterate is returned with a warning, never silently. |
| Trapezoid shows a "half step" right after a sharp pulse edge | Documented trapezoidal behaviour on discontinuities (the forcing is integrated as the step average). Use `method=be` for pulse-heavy circuits, or a smooth rise time. |
| No plots written | matplotlib not installed; text tables and CSV still work. `--no-plot` disables plots explicitly. |
| CSV lands in the current directory | `.export` paths are relative to the process working directory. |

## 9. Limitations (short version)

- Ideal components only: no parasitics, no transistor-level models, no
  nonlinear elements beyond the exponential diode.
- Fixed time step; no adaptive stepping (see the "next steps" list in the
  technical document).
- Initial-condition solve conflicts are diagnosed, not silently resolved.
- The diode model is the simple exponential law (no reverse breakdown,
  no series resistance, no temperature sweep beyond the fixed `Vt`).

Full list: `docs/technical_document.md` §9.
