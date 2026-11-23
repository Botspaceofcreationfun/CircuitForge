"""Tests for the netlist parser."""

import pytest

from circuitforge.netlist import (
    Circuit,
    NetlistError,
    DCSource,
    StepSource,
    PulseSource,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    OpDirective,
    TranDirective,
    parse_netlist,
)


def test_parse_basic_rc():
    c = parse_netlist(
        """
        # comment line
        V1 1 0 STEP(0 5 0)   ; trailing comment
        R1 1 2 1000
        C1 2 0 1e-6 0.5
        """
    )
    assert isinstance(c.get_component("V1"), VoltageSource)
    assert isinstance(c.get_component("R1"), Resistor)
    c1 = c.get_component("C1")
    assert isinstance(c1, Capacitor) and c1.v0 == 0.5
    assert c.nodes == [1, 2]


def test_ground_aliases():
    c = parse_netlist("R1 gnd 0 100\nV1 GND 1 DC(5)\nR2 ground 1 200")
    assert c.get_component("R1").n1 == 0 and c.get_component("R1").n2 == 0
    assert c.get_component("V1").n2 == 1
    assert c.get_component("R2").n1 == 0


def test_waveform_spaces_and_commas():
    v1 = parse_netlist("V1 1 0 STEP(0 5 0)").get_component("V1").waveform
    assert isinstance(v1, StepSource)
    v2 = parse_netlist("V2 1 0 STEP(0,5,1e-4)").get_component("V2").waveform
    assert v2.t0 == 1e-4
    assert v2.value(0) == 0 and v2.value(1e-4) == 5


def test_waveform_bare_number_is_dc():
    w = parse_netlist("V1 1 0 5").get_component("V1").waveform
    assert isinstance(w, DCSource) and w.value0 == 5


def test_pulse_semantics():
    # PULSE(0 5 0.1 0.2 0.3 0.4 1.0): tr=0.2 from td=0.1 -> reaches 5 at
    # absolute 0.3; on until 0.7; falls over 0.3 -> 0 at 1.0; idle until
    # period 1.0 repeats.
    w = PulseSource(0, 5, 0.1, 0.2, 0.3, 0.4, 1.0)
    assert w.value(0.05) == 0
    assert abs(w.value(0.2) - 2.5) < 1e-12  # halfway up the rise
    assert w.value(0.3) == pytest.approx(5.0)  # top of the rise (float edge)
    assert w.value(0.5) == 5
    assert abs(w.value(0.85) - 2.5) < 1e-12  # halfway down the fall
    assert abs(w.value(0.95) - (5.0 / 6.0)) < 1e-12  # 5/6 of the way down
    assert w.value(1.1) == 0
    assert w.value(1.5) == 5  # period repeats


def test_model_directive_anywhere():
    c = parse_netlist(
        """
        V1 1 0 DC(5)
        R1 1 2 1000
        D1 2 0 fast
        .model fast IS=1e-12 N=1.5
        """
    )
    d = c.get_component("D1")
    assert isinstance(d, Diode)
    assert d.is_sat == 1e-12 and d.n_ideal == 1.5


def test_model_defaults():
    c = parse_netlist("D1 1 0")
    assert c.get_component("D1").is_sat == 1e-14 and c.get_component("D1").n_ideal == 1.0


def test_unknown_model_raises():
    with pytest.raises(NetlistError, match="not defined"):
        parse_netlist("D1 1 0 mysuperdiode")


def test_unknown_element_type():
    with pytest.raises(NetlistError, match="unknown element type"):
        parse_netlist("X1 1 0 100")


def test_duplicate_names():
    with pytest.raises(NetlistError, match="duplicate"):
        parse_netlist("R1 1 0 100\nR1 2 0 200")


def test_bad_node():
    with pytest.raises(NetlistError, match="node"):
        parse_netlist("R1 a 0 100")


def test_float_node_rejected():
    with pytest.raises(NetlistError, match="node"):
        parse_netlist("R1 1.5 0 100")


def test_negative_resistance():
    with pytest.raises(NetlistError, match="resistance"):
        parse_netlist("R1 1 0 -100")


def test_tran_directive():
    c = parse_netlist(
        ".tran dt=1e-6 tstop=1e-3 method=be start=1e-4"
    )
    t = c.get_directive(TranDirective)
    assert t.dt == 1e-6 and t.tstop == 1e-3 and t.method == "be" and t.start == 1e-4


def test_tran_directive_defaults():
    c = parse_netlist(".tran dt=1e-6 tstop=1e-3")
    assert c.get_directive(TranDirective).method == "trap"


def test_tran_requires_keys():
    with pytest.raises(NetlistError, match="requires dt="):
        parse_netlist(".tran dt=1e-6")


def test_op_directive():
    assert isinstance(parse_netlist(".op").get_directive(OpDirective), OpDirective)


def test_bad_method():
    with pytest.raises(NetlistError, match="method"):
        parse_netlist(".tran dt=1e-6 tstop=1e-3 method=euler")


def test_pulse_bad_period():
    with pytest.raises(NetlistError, match="period"):
        parse_netlist("V1 1 0 PULSE(0 1 0 1e-6 1e-6 1e-3 0)")


def test_comment_only_lines_ok():
    c = parse_netlist("# nothing here\n\n; also nothing\nR1 1 0 1\n")
    assert len(c.components) == 1


def test_line_number_in_error():
    with pytest.raises(NetlistError) as ei:
        parse_netlist("R1 1 0 100\nX9 1 0 1", source_path="f.net")
    assert ":2:" in str(ei.value)
