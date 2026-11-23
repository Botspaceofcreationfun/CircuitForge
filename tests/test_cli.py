"""End-to-end tests for the CLI, CSV export, and example netlists."""

import csv
import os
import subprocess
import sys

import numpy as np
import pytest

from circuitforge.analysis import export_csv, run_netlist
from circuitforge.netlist import parse_netlist
from circuitforge.transient import run_transient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def _cli(*args, input_text=None, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "circuitforge", *args],
        capture_output=True,
        text=True,
        input=input_text,
        cwd=cwd or ROOT,
        env=env,
    )


def test_cli_run_divider():
    p = _cli("run", os.path.join(EXAMPLES, "divider.netlist"))
    assert p.returncode == 0, p.stderr
    assert "v(2) =  5.000000e+00" in p.stdout


def test_cli_run_rc_charge_and_csv(tmp_path):
    p = _cli("run", os.path.join(EXAMPLES, "rc_charge.netlist"), "--no-plot",
             cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr
    assert "all steps converged" in p.stdout
    # .export directive wrote rc_charge.csv into the cwd
    csv_path = os.path.join(str(tmp_path), "rc_charge.csv")
    assert os.path.exists(csv_path)
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t", "v(2)", "i(R1)"]
    assert len(rows) == 802  # 801 samples + header
    t = [float(r[0]) for r in rows[1:]]
    assert t[0] == 0.0 and t[-1] == pytest.approx(8e-3)
    v2 = [float(r[1]) for r in rows[1:]]
    ref = 5.0 * (1.0 - np.exp(-np.array(t) / 1e-3))
    assert np.max(np.abs(np.array(v2) - ref)) < 5e-4


def test_cli_missing_file():
    p = _cli("run", "does_not_exist.netlist")
    assert p.returncode == 2


def test_cli_invalid_netlist():
    bad = os.path.join(str(__import__("tempfile").mkdtemp()), "bad.netlist")
    with open(bad, "w") as fh:
        fh.write("R1 1 0 -5\n")
    p = _cli("run", bad)
    assert p.returncode == 1
    assert "error" in p.stderr.lower()


def test_cli_floating_node_diagnostic():
    p = _cli("run", os.path.join(EXAMPLES, "floating_node.netlist"))
    assert p.returncode == 1
    assert "floating" in p.stderr


def test_cli_override_options(tmp_path):
    p = _cli("run", os.path.join(EXAMPLES, "rc_charge.netlist"),
             "--dt", "1e-4", "--tstop", "2e-3", "--method", "be",
             "--no-plot", cwd=str(tmp_path))
    assert p.returncode == 0, p.stderr
    assert "dt = 0.0001 s" in p.stdout
    assert "method: be" in p.stdout or "Method: be" in p.stdout


def test_cli_list():
    p = _cli("list")
    assert p.returncode == 0
    assert "divider.netlist" in p.stdout


def test_export_csv_matches_api():
    c = parse_netlist("V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nC1 2 0 1e-6")
    r = run_transient(c, dt=1e-5, tstop=2e-3, method="trap")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        path = fh.name
    export_csv(r, ["v(2)", "i(R1)"], path)
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["t", "v(2)", "i(R1)"]
    assert len(rows) == 202
    os.unlink(path)


def test_run_netlist_directive_execution(tmp_path):
    c = parse_netlist(
        "V1 1 0 STEP(0 5 0)\nR1 1 2 1000\nC1 2 0 1e-6\n"
        ".tran dt=1e-5 tstop=1e-3\n.export out.csv",
        source_path="demo.netlist",
    )
    run = run_netlist(c, plot_dir=str(tmp_path), write_plots=True)
    assert run.tran is not None
    assert len(run.plots) == 1
    assert os.path.exists(run.csv_files[0])


def test_interactive_editor_basic():
    script = (
        "R1 1 0 1000\n"
        "V1 1 0 DC(5)\n"
        ".op\n"
        ".run\n"
        ".quit\n"
    )
    p = _cli("edit", input_text=script)
    assert p.returncode == 0
    assert "v(1) =  5.000000e+00" in p.stdout


def test_interactive_editor_save(tmp_path):
    target = os.path.join(str(tmp_path), "saved.netlist")
    script = (
        "R1 1 0 1000\n"
        f".save {target}\n"
        ".quit\n"
    )
    p = _cli("edit", input_text=script)
    assert p.returncode == 0
    assert os.path.exists(target)
    with open(target) as fh:
        assert fh.read().strip() == "R1 1 0 1000"
