"""Command-line interface for CircuitForge.

Subcommands
-----------
``run``   analyse a netlist file (op / transient, plots, CSV)
``edit``  interactive text-based netlist editor
``list``  list the bundled example netlists
``--version``  print the version
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Optional

from . import __version__
from .analysis import NetlistRun, parse_variable, run_netlist
from .netlist import NetlistError, TranDirective, parse_netlist
from .transient import TranResult

_EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _fmt_op(run: NetlistRun) -> str:
    lines: List[str] = ["DC operating point"]
    op = run.op
    if op is None:
        return ""
    lines.append("  Node voltages (V):")
    for n in sorted(op.node_voltages):
        lines.append(f"    v({n}) = {op.node_voltages[n]: .6e}")
    lines.append("  Branch currents (A, through n+ -> n-):")
    for name in sorted(op.branch_currents):
        lines.append(f"    i({name}) = {op.branch_currents[name]: .6e}")
    if op.diode_voltages:
        lines.append("  Diode voltages (V):")
        for name in sorted(op.diode_voltages):
            lines.append(f"    vd({name}) = {op.diode_voltages[name]: .6e}")
    lines.append(f"  Solver: {op.stats.message}")
    if op.condition is not None:
        lines.append(f"  Matrix condition number: {op.condition:.3g}")
    for w in op.warnings:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


def _fmt_tran(run: NetlistRun) -> str:
    if run.tran is None:
        return ""
    r = run.tran
    lines = [
        "Transient analysis",
        f"  Method: {r.method} (first step always backward Euler)",
        f"  dt = {r.dt:g} s, tstop = {r.tstop:g} s, samples = {len(r.t)}",
        f"  Convergence: "
        + ("all steps converged" if r.converged else "SOME STEPS DID NOT CONVERGE"),
    ]
    if r.init_stats is not None:
        lines.append(f"  Start-up (t=0): {r.init_stats.message}")
    for w in r.warnings:
        lines.append(f"  WARNING: {w}")
    if r.t.size:
        last = r.t[-1]
        lines.append("  Final values:")
        for n in sorted(r.v):
            lines.append(f"    v({n})({last:g}s) = {r.v[n][-1]: .6e}")
        for name in sorted(r.i):
            lines.append(f"    i({name})({last:g}s) = {r.i[name][-1]: .6e}")
    return "\n".join(lines)


def _run_file(path: str, args: argparse.Namespace) -> NetlistRun:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    circuit = parse_netlist(text, source_path=path)
    force_tran = None
    if args.dt is not None or args.tstop is not None or args.method is not None:
        dt = args.dt if args.dt is not None else 1e-6
        tstop = args.tstop if args.tstop is not None else 1e-3
        method = args.method if args.method is not None else "trap"
        force_tran = TranDirective(dt=dt, tstop=tstop, method=method)
    plot_dir = args.plot_dir or os.path.dirname(os.path.abspath(path))
    return run_netlist(
        circuit,
        plot_dir=plot_dir,
        force_op=args.op,
        force_tran=force_tran,
        write_plots=not args.no_plot,
    )


def cmd_run(args: argparse.Namespace) -> int:
    if not os.path.exists(args.netlist):
        print(f"error: netlist file not found: {args.netlist}", file=sys.stderr)
        return 2
    try:
        run = _run_file(args.netlist, args)
    except (NetlistError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for m in run.messages:
        print(m)
    if run.op is not None:
        print()
        print(_fmt_op(run))
    if run.tran is not None:
        print()
        print(_fmt_tran(run))
    return 0


# ---------------------------------------------------------------------------
# Interactive netlist editor
# ---------------------------------------------------------------------------

_EDITOR_HELP = """\
CircuitForge interactive editor — type netlist lines, one per line:

  R1 1 2 1000            resistor 1k between nodes 1 and 2
  C1 2 0 1e-6            capacitor 1uF node 2 to ground
  L1 2 0 1e-3            inductor 1mH
  V1 1 0 STEP(0 5 0)     voltage source, 5 V step at t = 0
  I1 1 0 1e-3            current source 1 mA (flows 1 -> 0)
  D1 2 0                 diode anode=2 cathode=0
  .model D IS=1e-14 N=1.0   diode model parameters
  .op                    run DC operating-point analysis
  .tran dt=1e-6 tstop=1e-3 method=trap   transient analysis
  .plot v(2) i(R1)       variables to plot / export
  .export out.csv        export results to CSV

Editor commands:
  .run        analyse the current buffer (.op/.tran per directives)
  .plot       re-run and write PNG plots of the last result
  .export f   write CSV of the last transient result
  .save f     save the buffer to a netlist file
  .load f     load a netlist file into the buffer
  .clear      clear the buffer
  .help       this help
  .quit       exit (Ctrl-D also works)
"""


def _edit_session(initial: Optional[str] = None) -> int:
    buffer: List[str] = []
    last_tran: Optional[TranResult] = None
    last_plots: List[str] = []
    if initial:
        buffer = [ln for ln in initial.splitlines() if ln.strip()]
        print("loaded buffer:")
        for ln in buffer:
            print(f"  {ln}")

    while True:
        try:
            line = input("cf> " if not buffer else f"cf[{len(buffer)}]> ")
        except EOFError:
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith("."):
            # Netlist directives (.op, .tran, .model, .plot, .export) belong
            # to the circuit buffer; only editor commands are handled here.
            if line.split()[0].lower() in (
                ".op", ".tran", ".model", ".plot", ".export",
            ):
                buffer.append(line)
                continue
            cmd = line.split()[0].lower()
            rest = line[len(cmd):].strip()
            if cmd in (".quit", ".exit"):
                break
            elif cmd == ".help":
                print(_EDITOR_HELP)
            elif cmd == ".clear":
                buffer.clear()
                last_tran = None
                print("buffer cleared")
            elif cmd == ".save":
                if not rest:
                    print("usage: .save <filename>")
                    continue
                with open(rest, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(buffer) + "\n")
                print(f"saved {len(buffer)} lines to {rest}")
            elif cmd == ".load":
                if not rest:
                    print("usage: .load <filename>")
                    continue
                try:
                    with open(rest, "r", encoding="utf-8") as fh:
                        buffer = [
                            ln for ln in fh.read().splitlines() if ln.strip()
                        ]
                    print(f"loaded {len(buffer)} lines from {rest}")
                except OSError as exc:
                    print(f"error: {exc}")
            elif cmd == ".run":
                try:
                    circuit = parse_netlist("\n".join(buffer), source_path="<editor>")
                    run = run_netlist(
                        circuit,
                        plot_dir=None,
                        write_plots=False,
                    )
                except (NetlistError, ValueError, RuntimeError) as exc:
                    print(f"error: {exc}")
                    continue
                print(_fmt_op(run))
                print(_fmt_tran(run))
                for w in run.messages:
                    print(f"  note: {w}")
                last_tran = run.tran
                last_plots = run.plots
            elif cmd == ".plot":
                if last_tran is None:
                    print("no transient result yet; run .run first (needs .tran)")
                    continue
                try:
                    circuit = parse_netlist("\n".join(buffer), source_path="<editor>")
                    variables = _collect_plot_variables(circuit, last_tran)
                    from .analysis import plot_tran

                    paths = plot_tran(last_tran, variables, plot_dir=".")
                    last_plots = paths
                    for p in paths:
                        print(f"plot written: {p}")
                except (NetlistError, ValueError, RuntimeError) as exc:
                    print(f"error: {exc}")
            elif cmd == ".export":
                if not rest:
                    print("usage: .export <filename.csv>")
                    continue
                if last_tran is None:
                    print("no transient result yet; run .run first (needs .tran)")
                    continue
                try:
                    circuit = parse_netlist("\n".join(buffer), source_path="<editor>")
                    variables = _collect_plot_variables(circuit, last_tran)
                    from .export import write_csv

                    path = write_csv(last_tran, variables, rest)
                    print(f"CSV written: {path}")
                except (NetlistError, ValueError, RuntimeError) as exc:
                    print(f"error: {exc}")
            else:
                print(f"unknown editor command {cmd!r}; type .help")
        else:
            buffer.append(line)

    if buffer:
        print("(buffer not saved; use .save <file> to keep it)")
    return 0


def _collect_plot_variables(circuit, result):
    from .analysis import _collect_variables

    return _collect_variables(circuit)


def cmd_edit(args: argparse.Namespace) -> int:
    initial = None
    if args.file:
        if not os.path.exists(args.file):
            print(f"error: file not found: {args.file}", file=sys.stderr)
            return 2
        with open(args.file, "r", encoding="utf-8") as fh:
            initial = fh.read()
    return _edit_session(initial)


def cmd_list(args: argparse.Namespace) -> int:
    if not os.path.isdir(_EXAMPLE_DIR):
        print("no bundled examples found", file=sys.stderr)
        return 1
    for name in sorted(os.listdir(_EXAMPLE_DIR)):
        if name.endswith(".netlist"):
            path = os.path.join(_EXAMPLE_DIR, name)
            with open(path, "r", encoding="utf-8") as fh:
                first = fh.readline().strip()
            print(f"  {name:<28} {first}")
    print("\nusage: circuitforge run examples/<name>.netlist")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuitforge",
        description="CircuitForge — a small circuit simulator built on "
        "modified nodal analysis (MNA).",
    )
    parser.add_argument("--version", action="version", version=f"CircuitForge {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="analyse a netlist file")
    p_run.add_argument("netlist", help="path to the .netlist file")
    p_run.add_argument("--op", action="store_true", help="force DC operating point")
    p_run.add_argument("--dt", type=float, default=None, help="override time step (s)")
    p_run.add_argument("--tstop", type=float, default=None, help="override stop time (s)")
    p_run.add_argument("--method", choices=["be", "trap"], default=None,
                       help="override integration method")
    p_run.add_argument("--plot-dir", default=None, help="directory for PNG plots")
    p_run.add_argument("--no-plot", action="store_true", help="skip writing plots")
    p_run.set_defaults(func=cmd_run)

    p_edit = sub.add_parser("edit", help="interactive text-based netlist editor")
    p_edit.add_argument("file", nargs="?", default=None, help="optional netlist to load")
    p_edit.set_defaults(func=cmd_edit)

    p_list = sub.add_parser("list", help="list bundled example netlists")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
