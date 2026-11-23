"""Netlist parsing and the circuit data model for CircuitForge.

The netlist format is line oriented, one component or directive per line.
Lines starting with ``#`` or containing ``;`` (after the first token) are
comments and are ignored.  Tokens are separated by whitespace (spaces/tabs).

Ground is node ``0`` (aliases: ``0``, ``gnd``, ``GND``, ``ground``).
Every other node must be a non-negative integer.

Component lines
---------------
``R<name> n+ n- value``        resistor, ohms
``C<name> n+ n- value [v0]``   capacitor, farads, optional initial voltage V
``L<name> n+ n- value [i0]``   inductor, henrys, optional initial current A
``V<name> n+ n- waveform``     independent voltage source (n+ minus n-)
``I<name> n+ n- waveform``     independent current source (flows n+ -> n-)
``D<name> anode cathode [model]``  ideal diode, anode -> cathode conduction

Waveforms
---------
``DC(value)``                  constant value
``STEP(v0, v1, t0)``           v0 for t < t0, v1 for t >= t0
``PULSE(v0, v1, td, tr, tf, pw, per)``   SPICE-style pulse
``5``                          bare number is shorthand for DC(5)

PULSE semantics (SPICE convention):
  v = v0                       for t < td
  v = v0 + (v1-v0)*(t-td)/tr   for td <= t < td+tr     (rise)
  v = v1                       for td+tr <= t < td+tr+pw (on)
  v = v1 - (v1-v0)*(t-td-tr-pw)/tf  for ... <= t < td+tr+pw+tf (fall)
  v = v0                       after the fall until per (idle), repeats.

Directives
----------
``.op``                                  perform DC operating-point analysis
``.tran dt=.. tstop=.. [start=..] [method=be|trap]``   transient analysis
``.model D IS=.. N=..``                  diode model parameters (defaults IS=1e-14, N=1.0)
``.plot v(n1) v(n2) ... i(R1) ...``      variables plotted / written to CSV
``.export <filename.csv>``               write time + requested variables to CSV
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NetlistError(ValueError):
    """Raised when a netlist cannot be parsed or is inconsistent."""


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------


class Waveform:
    """Base class for time-varying source waveforms.

    Subclasses implement ``value(t)`` returning the source value at time t.
    """

    kind = "base"

    def value(self, t: float) -> float:
        raise NotImplementedError


@dataclass
class DCSource(Waveform):
    """Constant (time-invariant) source value."""

    value0: float
    kind = "dc"

    def value(self, t: float) -> float:
        return self.value0

    def __str__(self) -> str:
        return f"DC({self.value0:g})"


@dataclass
class StepSource(Waveform):
    """Step from v0 to v1 at time t0 (v0 for t < t0, v1 for t >= t0)."""

    v0: float
    v1: float
    t0: float
    kind = "step"

    def value(self, t: float) -> float:
        return self.v1 if t >= self.t0 else self.v0

    def __str__(self) -> str:
        return f"STEP({self.v0:g},{self.v1:g},{self.t0:g})"


@dataclass
class PulseSource(Waveform):
    """SPICE-style periodic pulse.

    Parameters: v0 (base), v1 (peak), td (delay), tr (rise time),
    tf (fall time), pw (pulse width at v1), per (period).
    """

    v0: float
    v1: float
    td: float
    tr: float
    tf: float
    pw: float
    per: float
    kind = "pulse"

    def value(self, t: float) -> float:
        if self.per <= 0:
            raise NetlistError("PULSE period must be > 0")
        t = (t - self.td) % self.per
        if t < 0:
            return self.v0
        rise = self.tr
        on = self.tr + self.pw
        fall = self.tr + self.pw + self.tf
        if t < rise:
            if rise <= 0:
                return self.v1
            return self.v0 + (self.v1 - self.v0) * (t / rise)
        if t < on:
            return self.v1
        if t < fall:
            if self.tf <= 0:
                return self.v0
            return self.v1 - (self.v1 - self.v0) * ((t - on) / self.tf)
        return self.v0

    def __str__(self) -> str:
        return (
            f"PULSE({self.v0:g},{self.v1:g},{self.td:g},"
            f"{self.tr:g},{self.tf:g},{self.pw:g},{self.per:g})"
        )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclass
class Resistor:
    name: str
    n1: int
    n2: int
    value: float  # ohms

    @property
    def conductance(self) -> float:
        if self.value <= 0:
            raise NetlistError(f"{self.name}: resistor value must be > 0")
        return 1.0 / self.value


@dataclass
class Capacitor:
    name: str
    n1: int
    n2: int
    value: float  # farads
    v0: float = 0.0  # initial voltage across (v(n1) - v(n2)) at t = 0


@dataclass
class Inductor:
    name: str
    n1: int
    n2: int
    value: float  # henrys
    i0: float = 0.0  # initial current flowing n1 -> n2 at t = 0


@dataclass
class VoltageSource:
    name: str
    n1: int
    n2: int
    waveform: Waveform


@dataclass
class CurrentSource:
    name: str
    n1: int
    n2: int
    waveform: Waveform


@dataclass
class Diode:
    """Ideal exponential diode: I = IS*(exp(Vd/(N*Vt)) - 1), Vt = kT/q."""

    name: str
    anode: int
    cathode: int
    model_name: str = "default"

    # Diode model parameters (filled from .model directives or defaults).
    is_sat: float = 1e-14  # saturation current, A
    n_ideal: float = 1.0  # ideality factor

    @staticmethod
    def thermal_voltage(temperature_k: float = 300.0) -> float:
        """Vt = kT/q at the given temperature."""
        return 1.380649e-23 * temperature_k / 1.602176634e-19

    def current(self, vd: float, vt: float) -> float:
        """Diode current for voltage vd = v(anode) - v(cathode)."""
        x = vd / (self.n_ideal * vt)
        # Clamp the exponential argument to avoid float overflow while keeping
        # the linear continuation valid for the device model (documented).
        if x > 40.0:
            return self.is_sat * (math.exp(40.0) * (1.0 + (x - 40.0)) - 1.0)
        return self.is_sat * (math.exp(x) - 1.0)

    def conductance(self, vd: float, vt: float) -> float:
        """dI/dVd at the given diode voltage."""
        x = vd / (self.n_ideal * vt)
        if x > 40.0:
            return self.is_sat * math.exp(40.0) / (self.n_ideal * vt)
        return self.is_sat * math.exp(x) / (self.n_ideal * vt)


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


@dataclass
class OpDirective:
    """Run a DC operating-point analysis."""


@dataclass
class TranDirective:
    dt: float
    tstop: float
    start: float = 0.0
    method: str = "trap"  # 'be' (backward Euler) or 'trap' (trapezoidal)


@dataclass
class ModelDirective:
    """Diode model parameters: .model <name> IS=.. N=.."""

    model_name: str
    is_sat: float
    n_ideal: float


@dataclass
class PlotDirective:
    """Variables to plot / export: e.g. v(2), i(R1), i(V1)."""

    variables: List[str]


@dataclass
class ExportDirective:
    filename: str


# ---------------------------------------------------------------------------
# Circuit container
# ---------------------------------------------------------------------------


@dataclass
class Circuit:
    """Everything parsed from one netlist file."""

    source_path: Optional[str] = None
    components: List[object] = field(default_factory=list)
    directives: List[object] = field(default_factory=list)
    diode_models: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------
    @property
    def resistors(self) -> List[Resistor]:
        return [c for c in self.components if isinstance(c, Resistor)]

    @property
    def capacitors(self) -> List[Capacitor]:
        return [c for c in self.components if isinstance(c, Capacitor)]

    @property
    def inductors(self) -> List[Inductor]:
        return [c for c in self.components if isinstance(c, Inductor)]

    @property
    def voltage_sources(self) -> List[VoltageSource]:
        return [c for c in self.components if isinstance(c, VoltageSource)]

    @property
    def current_sources(self) -> List[CurrentSource]:
        return [c for c in self.components if isinstance(c, CurrentSource)]

    @property
    def diodes(self) -> List[Diode]:
        return [c for c in self.components if isinstance(c, Diode)]

    @property
    def nodes(self) -> List[int]:
        """All non-ground node numbers referenced by any component."""
        seen: set = set()
        for c in self.components:
            for n in _component_nodes(c):
                if n != 0:
                    seen.add(n)
        return sorted(seen)

    def get_component(self, name: str) -> object:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"no component named {name!r}")

    def get_directive(self, cls) -> Optional[object]:
        for d in self.directives:
            if isinstance(d, cls):
                return d
        return None

    def validate(self) -> None:
        """Consistency checks: positive values, valid node ids, no dups."""
        names: set = set()
        for c in self.components:
            if c.name in names:
                raise NetlistError(f"duplicate component name {c.name!r}")
            names.add(c.name)
            for n in _component_nodes(c):
                if n < 0:
                    raise NetlistError(
                        f"{c.name}: node must be a non-negative integer, got {n}"
                    )
            if isinstance(c, Resistor) and c.value <= 0:
                raise NetlistError(f"{c.name}: resistance must be > 0")
            if isinstance(c, Capacitor) and c.value <= 0:
                raise NetlistError(f"{c.name}: capacitance must be > 0")
            if isinstance(c, Inductor) and c.value <= 0:
                raise NetlistError(f"{c.name}: inductance must be > 0")
            if isinstance(c, (VoltageSource, CurrentSource)):
                pass  # waveform validated at parse time
            if isinstance(c, Diode):
                if c.is_sat <= 0 or c.n_ideal <= 0:
                    raise NetlistError(
                        f"{c.name}: diode model needs IS>0 and N>0"
                    )
        for c in self.capacitors:
            if c.v0 < 0 and c.v0 != 0:
                pass  # negative initial voltages are physically fine
        # ground must exist as node 0 implicitly; nothing else to check here.


def _component_nodes(c) -> List[int]:
    if isinstance(c, Resistor):
        return [c.n1, c.n2]
    if isinstance(c, Capacitor):
        return [c.n1, c.n2]
    if isinstance(c, Inductor):
        return [c.n1, c.n2]
    if isinstance(c, VoltageSource):
        return [c.n1, c.n2]
    if isinstance(c, CurrentSource):
        return [c.n1, c.n2]
    if isinstance(c, Diode):
        return [c.anode, c.cathode]
    raise TypeError(f"unknown component type {type(c)}")


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$"
)


def parse_number(token: str, what: str) -> float:
    t = token.strip().rstrip()
    if t in ("", "-", "+", ".", "e", "E"):
        raise NetlistError(f"invalid number for {what}: {token!r}")
    try:
        return float(t)
    except ValueError as exc:
        raise NetlistError(f"invalid number for {what}: {token!r}") from exc


# ---------------------------------------------------------------------------
# Waveform parsing
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"^([A-Za-z]+)\s*\((.*)\)$", re.S)


def parse_waveform(token: str, what: str) -> Waveform:
    """Parse ``DC(x)``, ``STEP(a,b,t0)``, ``PULSE(...)`` or a bare number."""
    token = token.strip()
    m = _FUNC_RE.match(token)
    if not m:
        return DCSource(parse_number(token, what))
    func, args = m.group(1).upper(), m.group(2)
    parts = [p.strip() for p in re.split(r"[,\s]+", args) if p.strip() != ""]
    try:
        if func == "DC":
            if len(parts) != 1:
                raise NetlistError(f"{what}: DC() takes one argument")
            return DCSource(parse_number(parts[0], what))
        if func == "STEP":
            if len(parts) != 3:
                raise NetlistError(f"{what}: STEP() takes three arguments")
            return StepSource(
                parse_number(parts[0], what),
                parse_number(parts[1], what),
                parse_number(parts[2], what),
            )
        if func == "PULSE":
            if len(parts) != 7:
                raise NetlistError(
                    f"{what}: PULSE() takes seven arguments "
                    "(v0,v1,td,tr,tf,pw,per)"
                )
            nums = [parse_number(p, what) for p in parts]
            if nums[6] <= 0:
                raise NetlistError(f"{what}: PULSE period must be > 0")
            return PulseSource(*nums)
    except NetlistError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise NetlistError(f"{what}: malformed waveform {token!r}") from exc
    raise NetlistError(f"{what}: unknown waveform function {func!r}")


# ---------------------------------------------------------------------------
# Netlist parser
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Remove comments: a line starting with ``#``, or a ``;`` anywhere."""
    line = line.strip()
    if line.startswith("#"):
        return ""
    idx = line.find(";")
    if idx != -1:
        line = line[:idx]
    return line.strip()


def _tokenize(line: str) -> List[str]:
    """Split a netlist line into tokens, keeping parenthesised function
    arguments (e.g. ``STEP(0 5 0)``) together as a single token."""
    tokens: List[str] = []
    i, n = 0, len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            break
        depth = 0
        start = i
        while i < n:
            ch = line[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
            if depth == 0 and (i >= n or line[i].isspace()):
                break
        tokens.append(line[start:i])
    return tokens


def parse_netlist(text: str, source_path: Optional[str] = None) -> Circuit:
    """Parse netlist text into a :class:`Circuit`."""
    circuit = Circuit(source_path=source_path)
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line:
            continue
        tokens = _tokenize(line)
        head = tokens[0]
        try:
            if head.startswith("."):
                _parse_directive(circuit, head, tokens[1:], lineno)
            else:
                _parse_component(circuit, head, tokens[1:], lineno)
        except NetlistError as exc:
            raise NetlistError(f"{source_path or '<netlist>'}:{lineno}: {exc}") from exc
    _apply_diode_models(circuit)
    circuit.validate()
    return circuit


def _parse_directive(circuit: Circuit, head: str, args: List[str], lineno: int) -> None:
    if head == ".op":
        if args:
            raise NetlistError(".op takes no arguments")
        circuit.directives.append(OpDirective())
        return
    if head == ".tran":
        dt = tstop = None
        start = 0.0
        method = "trap"
        for a in args:
            if "=" in a:
                key, _, val = a.partition("=")
                key = key.strip().lower()
                val = val.strip()
                if key == "dt":
                    dt = parse_number(val, ".tran dt")
                elif key == "tstop":
                    tstop = parse_number(val, ".tran tstop")
                elif key == "start":
                    start = parse_number(val, ".tran start")
                elif key == "method":
                    if val not in ("be", "trap"):
                        raise NetlistError(
                            ".tran method must be 'be' or 'trap'"
                        )
                    method = val
                else:
                    raise NetlistError(f".tran: unknown option {key!r}")
            else:
                raise NetlistError(".tran options must be key=value")
        if dt is None or tstop is None:
            raise NetlistError(".tran requires dt=.. and tstop=..")
        if dt <= 0 or tstop <= 0:
            raise NetlistError(".tran dt and tstop must be > 0")
        if start < 0:
            raise NetlistError(".tran start must be >= 0")
        circuit.directives.append(TranDirective(dt, tstop, start, method))
        return
    if head == ".model":
        if len(args) < 2:
            raise NetlistError(".model needs a name and parameters")
        name = args[0]
        is_sat = 1e-14
        n_ideal = 1.0
        for a in args[1:]:
            if "=" not in a:
                raise NetlistError(f".model {name}: parameters must be key=value")
            key, _, val = a.partition("=")
            key = key.strip().upper()
            if key == "IS":
                is_sat = parse_number(val.strip(), ".model IS")
            elif key == "N":
                n_ideal = parse_number(val.strip(), ".model N")
            else:
                raise NetlistError(f".model {name}: unknown parameter {key!r}")
        if is_sat <= 0 or n_ideal <= 0:
            raise NetlistError(".model requires IS>0 and N>0")
        circuit.diode_models[name] = (is_sat, n_ideal)
        return
    if head == ".plot":
        if not args:
            raise NetlistError(".plot needs at least one variable")
        circuit.directives.append(PlotDirective(args))
        return
    if head == ".export":
        if len(args) != 1:
            raise NetlistError(".export takes exactly one filename")
        circuit.directives.append(ExportDirective(args[0]))
        return
    raise NetlistError(f"unknown directive {head!r}")


_ELEM_TYPES = {
    "R": Resistor,
    "C": Capacitor,
    "L": Inductor,
    "V": VoltageSource,
    "I": CurrentSource,
    "D": Diode,
}


def _parse_component(circuit: Circuit, head: str, args: List[str], lineno: int) -> None:
    kind = head[0].upper()
    name = head
    if kind not in _ELEM_TYPES:
        raise NetlistError(
            f"unknown element type {head[0]!r} "
            "(expected R, C, L, V, I or D)"
        )

    if kind == "R":
        if len(args) != 3:
            raise NetlistError("R needs: n+ n- value")
        c = Resistor(name, _node(args[0]), _node(args[1]),
                     parse_number(args[2], name))
    elif kind == "C":
        if len(args) not in (3, 4):
            raise NetlistError("C needs: n+ n- value [v0]")
        v0 = parse_number(args[3], name) if len(args) == 4 else 0.0
        c = Capacitor(name, _node(args[0]), _node(args[1]),
                      parse_number(args[2], name), v0)
    elif kind == "L":
        if len(args) not in (3, 4):
            raise NetlistError("L needs: n+ n- value [i0]")
        i0 = parse_number(args[3], name) if len(args) == 4 else 0.0
        c = Inductor(name, _node(args[0]), _node(args[1]),
                     parse_number(args[2], name), i0)
    elif kind == "V":
        if len(args) != 3:
            raise NetlistError("V needs: n+ n- waveform")
        c = VoltageSource(name, _node(args[0]), _node(args[1]),
                          parse_waveform(args[2], name))
    elif kind == "I":
        if len(args) != 3:
            raise NetlistError("I needs: n+ n- waveform")
        c = CurrentSource(name, _node(args[0]), _node(args[1]),
                          parse_waveform(args[2], name))
    else:  # D
        if len(args) not in (2, 3):
            raise NetlistError("D needs: anode cathode [model]")
        c = Diode(name, _node(args[0]), _node(args[1]),
                  args[2] if len(args) == 3 else "default")
    circuit.components.append(c)


def _apply_diode_models(circuit: Circuit) -> None:
    """Bind .model parameters to diodes (resolved after the full parse so the
    .model line may appear anywhere in the netlist, as in SPICE)."""
    for d in circuit.diodes:
        if d.model_name in circuit.diode_models:
            is_sat, n_ideal = circuit.diode_models[d.model_name]
            d.is_sat, d.n_ideal = is_sat, n_ideal
        elif d.model_name == "default":
            d.is_sat, d.n_ideal = 1e-14, 1.0
        else:
            raise NetlistError(f"diode {d.name}: model {d.model_name!r} not defined")


def _node(token: str) -> int:
    t = token.strip().lower()
    if t in ("0", "gnd", "ground"):
        return 0
    try:
        v = float(t)
    except ValueError:
        raise NetlistError(
            f"invalid node {token!r} (must be an integer >= 0, 0 = ground)"
        ) from None
    if not v.is_integer() or v < 0:
        raise NetlistError(
            f"invalid node {token!r} (must be an integer >= 0, 0 = ground)"
        )
    return int(v)
