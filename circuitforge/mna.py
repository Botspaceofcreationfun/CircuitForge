"""Modified nodal analysis (MNA): system assembly and linear solves.

Layout of the MNA system
------------------------
The unknowns are, in order:
  x[0 .. N-1]     node voltages of every non-ground node (see ``node_rows``)
  x[N .. N+M-1]   branch currents of "voltage-like" unknowns

A voltage-like unknown is created for every independent voltage source and,
depending on the analysis, for inductor shorts (DC) and capacitor initial
sources (transient start-up).  Its current is defined as flowing from the
element's n+ (or anode) terminal to its n- (or cathode) terminal through the
element.

Stamp table (all values in SI units; ground node 0 is never a row)
-----------------------------------------------------------------
Resistor G between i and j:      +G on (i,i),(j,j), -G on (i,j),(j,i)
Current source J flowing i -> j: z[i] -= J,  z[j] += J
Voltage source Vs between a, b (unknown current I):
  row a: A[a, k] += 1 ; row b: A[b, k] -= 1
  constraint row k: A[k, a] += 1 ; A[k, b] -= 1 ; z[k] = Vs
Diode linearized at vd with slope g and intercept c = I0 - g*vd
  (branch current from anode a to cathode c is g*(Va-Vc) + c):
  A[a,a] += g, A[c,c] += g, A[a,c] -= g, A[c,a] -= g, z[a] -= c, z[c] += c

KCL sign convention: at every node, the sum of currents *leaving* the node
through all branches equals the current *injected* by current sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .netlist import (
    Circuit,
    Resistor,
    Capacitor,
    Inductor,
    VoltageSource,
    CurrentSource,
    Diode,
    NetlistError,
)

# Condition number above which the matrix is reported as ill-conditioned.
_ILL_COND_THRESHOLD = 1.0e12


class MNAError(RuntimeError):
    """Raised when the assembled MNA system cannot be solved."""


@dataclass
class MnaSystem:
    """One assembled linear MNA system (diodes NOT included).

    Attributes
    ----------
    A, z           : coefficient matrix and right-hand side
    node_rows      : node number -> row index of its voltage unknown
    vs_rows        : element name -> row index of its current unknown
    source_at_t    : (element, value) list of evaluated independent sources
    condition      : estimated condition number of A (or None)
    """

    A: np.ndarray
    z: np.ndarray
    node_rows: Dict[int, int]
    vs_rows: Dict[str, int]
    source_at_t: List[Tuple[object, float]] = field(default_factory=list)
    condition: Optional[float] = None

    @property
    def num_nodes(self) -> int:
        return len(self.node_rows)

    @property
    def num_vs(self) -> int:
        return len(self.vs_rows)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _stamp_conductance(A, z, node_rows, i: int, j: int, g: float) -> None:
    if i != 0:
        r = node_rows[i]
        A[r, r] += g
    if j != 0:
        r = node_rows[j]
        A[r, r] += g
    if i != 0 and j != 0:
        ri, rj = node_rows[i], node_rows[j]
        A[ri, rj] -= g
        A[rj, ri] -= g


def _stamp_current_source(A, z, node_rows, i: int, j: int, current: float) -> None:
    """Current source forcing `current` amperes to flow from i to j."""
    if i != 0:
        z[node_rows[i]] -= current
    if j != 0:
        z[node_rows[j]] += current


def _stamp_voltage_source(A, z, node_rows, a: int, b: int, vs: float, row: int) -> None:
    """Voltage source Vs = v(a) - v(b) with unknown current at `row`."""
    if a != 0:
        A[node_rows[a], row] += 1.0
    if b != 0:
        A[node_rows[b], row] -= 1.0
    if a != 0:
        A[row, node_rows[a]] += 1.0
    if b != 0:
        A[row, node_rows[b]] -= 1.0
    z[row] = vs


def stamp_diode_linearization(
    A, z, node_rows, anode: int, cathode: int, g: float, c: float
) -> None:
    """Stamp a diode linearized as I_branch = g*(Va - Vc) + c (anode -> cathode)."""
    if anode != 0:
        r = node_rows[anode]
        A[r, r] += g
        z[r] -= c
    if cathode != 0:
        r = node_rows[cathode]
        A[r, r] += g
        z[r] += c
    if anode != 0 and cathode != 0:
        ra, rc = node_rows[anode], node_rows[cathode]
        A[ra, rc] -= g
        A[rc, ra] -= g


def _vs_unknown_names(circuit: Circuit, caps: bool, inds_as_shorts: bool) -> List[str]:
    names = [c.name for c in circuit.voltage_sources]
    if inds_as_shorts:
        names += [c.name for c in circuit.inductors]
    if caps:
        names += [c.name for c in circuit.capacitors]
    return names


def _base_system(circuit: Circuit, vs_names: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[str, int]]:
    nodes = circuit.nodes
    node_rows = {n: i for i, n in enumerate(nodes)}
    n_total = len(nodes) + len(vs_names)
    A = np.zeros((n_total, n_total))
    z = np.zeros(n_total)
    vs_rows = {name: len(nodes) + i for i, name in enumerate(vs_names)}
    return A, z, node_rows, vs_rows


# ---------------------------------------------------------------------------
# System builders (one per analysis stage; diodes handled by the NR loop)
# ---------------------------------------------------------------------------


def build_dc_op_system(circuit: Circuit, t: float = 0.0) -> MnaSystem:
    """DC operating point: capacitors open, inductors shorted (0 V sources)."""
    vs_names = _vs_unknown_names(circuit, caps=False, inds_as_shorts=True)
    A, z, node_rows, vs_rows = _base_system(circuit, vs_names)
    sources = []
    for c in circuit.resistors:
        _stamp_conductance(A, z, node_rows, c.n1, c.n2, c.conductance)
    for c in circuit.current_sources:
        val = c.waveform.value(t)
        _stamp_current_source(A, z, node_rows, c.n1, c.n2, val)
        sources.append((c, val))
    for c in circuit.voltage_sources:
        val = c.waveform.value(t)
        _stamp_voltage_source(A, z, node_rows, c.n1, c.n2, val, vs_rows[c.name])
        sources.append((c, val))
    for c in circuit.inductors:
        # Ideal inductor is a short at DC: 0 V between its terminals.
        _stamp_voltage_source(A, z, node_rows, c.n1, c.n2, 0.0, vs_rows[c.name])
    return MnaSystem(A, z, node_rows, vs_rows, sources)


def build_init_op_system(circuit: Circuit, t: float = 0.0) -> MnaSystem:
    """Transient start-up point at t = 0: capacitors hold v0 (voltage sources),
    inductors carry i0 (current sources).  Ideal because dynamics start here."""
    vs_names = _vs_unknown_names(circuit, caps=True, inds_as_shorts=False)
    A, z, node_rows, vs_rows = _base_system(circuit, vs_names)
    sources = []
    for c in circuit.resistors:
        _stamp_conductance(A, z, node_rows, c.n1, c.n2, c.conductance)
    for c in circuit.current_sources:
        val = c.waveform.value(t)
        _stamp_current_source(A, z, node_rows, c.n1, c.n2, val)
        sources.append((c, val))
    for c in circuit.voltage_sources:
        val = c.waveform.value(t)
        _stamp_voltage_source(A, z, node_rows, c.n1, c.n2, val, vs_rows[c.name])
        sources.append((c, val))
    for c in circuit.capacitors:
        _stamp_voltage_source(A, z, node_rows, c.n1, c.n2, c.v0, vs_rows[c.name])
    for c in circuit.inductors:
        _stamp_current_source(A, z, node_rows, c.n1, c.n2, c.i0)
    return MnaSystem(A, z, node_rows, vs_rows, sources)


def build_tran_step_system(
    circuit: Circuit,
    t: float,
    states: Dict[str, dict],
    method: str,
) -> MnaSystem:
    """Transient step at time t using companion models.

    ``states`` maps element name -> state dict holding the quantities needed
    to form the companion sources (filled by :mod:`circuitforge.transient`).
    """
    vs_names = _vs_unknown_names(circuit, caps=False, inds_as_shorts=False)
    A, z, node_rows, vs_rows = _base_system(circuit, vs_names)
    sources = []
    for c in circuit.resistors:
        _stamp_conductance(A, z, node_rows, c.n1, c.n2, c.conductance)
    for c in circuit.current_sources:
        val = c.waveform.value(t)
        _stamp_current_source(A, z, node_rows, c.n1, c.n2, val)
        sources.append((c, val))
    for c in circuit.voltage_sources:
        val = c.waveform.value(t)
        _stamp_voltage_source(A, z, node_rows, c.n1, c.n2, val, vs_rows[c.name])
        sources.append((c, val))
    for c in circuit.capacitors:
        st = states[c.name]
        _stamp_conductance(A, z, node_rows, c.n1, c.n2, st["g"])
        j = st["j"]  # companion current source, oriented n+ -> n-
        _stamp_current_source(A, z, node_rows, c.n1, c.n2, j)
    for c in circuit.inductors:
        st = states[c.name]
        _stamp_conductance(A, z, node_rows, c.n1, c.n2, st["g"])
        _stamp_current_source(A, z, node_rows, c.n1, c.n2, st["j"])
    return MnaSystem(A, z, node_rows, vs_rows, sources)


# ---------------------------------------------------------------------------
# Linear solve and singularity diagnostics
# ---------------------------------------------------------------------------


def estimate_condition(A: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(A))
    except Exception:  # pragma: no cover - defensive
        return float("inf")


def solve_linear(system: MnaSystem, check_condition: bool = True) -> np.ndarray:
    """Solve A x = z; raises :class:`MNAError` with a diagnostic on failure."""
    try:
        x = np.linalg.solve(system.A, system.z)
    except np.linalg.LinAlgError as exc:
        raise MNAError(diagnose_singular(system)) from exc
    if check_condition:
        system.condition = estimate_condition(system.A)
    return x


def _conductive_edges(circuit: Circuit, mode: str) -> List[Tuple[int, int]]:
    """Undirected graph edges that give a node a conductive/constraint path.

    mode 'dc'  : resistors, voltage sources (constraints), inductor shorts,
                 diodes (conduct when linearized); capacitors are open.
    mode 'init': same as 'dc' but capacitors act as sources (edges) and
                 inductors as current sources (no edge).
    mode 'tran': resistors, capacitors, inductors (companions), voltage
                 sources, diodes.
    """
    edges: List[Tuple[int, int]] = []
    for c in circuit.components:
        if isinstance(c, Resistor):
            edges.append((c.n1, c.n2))
        elif isinstance(c, VoltageSource):
            edges.append((c.n1, c.n2))
        elif isinstance(c, Diode):
            edges.append((c.anode, c.cathode))
        elif isinstance(c, Inductor):
            if mode != "init":
                edges.append((c.n1, c.n2))
        elif isinstance(c, Capacitor):
            if mode in ("init", "tran"):
                edges.append((c.n1, c.n2))
    return edges


def floating_nodes(circuit: Circuit, mode: str = "dc") -> List[int]:
    """Return every non-ground node with no conductive path to ground.

    A node is considered connected if it is reachable from ground (0) in the
    graph of ``_conductive_edges``.
    """
    edges = _conductive_edges(circuit, mode)
    adj: Dict[int, List[int]] = {0: []}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen = {0}
    stack = [0]
    while stack:
        n = stack.pop()
        for m in adj.get(n, []):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return [n for n in circuit.nodes if n not in seen]


def diagnose_singular(system: MnaSystem) -> str:
    """Human-readable explanation of why the MNA matrix is singular."""
    floating = floating_nodes_from_system(system)
    if floating:
        names = ", ".join(str(n) for n in floating)
        return (
            "singular MNA matrix: node(s) "
            f"{names} are floating (no conductive path to ground). "
            "Add a resistor/voltage-source path to ground or fix the netlist."
        )
    rank = np.linalg.matrix_rank(system.A)
    n = system.A.shape[0]
    return (
        f"singular MNA matrix: rank {rank} < {n}. "
        "This usually means a voltage-source loop or conflicting constraints "
        "(e.g. two voltage sources in parallel with different values, or a "
        "current source in series with an open branch)."
    )


def floating_nodes_from_system(system: MnaSystem) -> List[int]:
    """Floating nodes read directly from a singular system matrix.

    After Gaussian elimination with partial pivoting on the node-voltage
    block, a floating node shows up as a (near-)zero diagonal pivot.  This
    inspects the actual matrix, so it works for every analysis mode.
    """
    n_nodes = system.num_nodes
    if n_nodes == 0:
        return []
    row_node = {row: node for node, row in system.node_rows.items()}
    working = system.A[:n_nodes, :n_nodes].astype(float).copy()
    order = list(range(n_nodes))  # order[k] = original row index at position k
    tol = 1e-14
    for k in range(n_nodes):
        piv = int(np.argmax(np.abs(working[k:, k]))) + k
        if piv != k:
            working[[k, piv]] = working[[piv, k]]
            order[k], order[piv] = order[piv], order[k]
        piv_val = working[k, k]
        if abs(piv_val) <= tol * (1.0 + float(np.max(np.abs(working[k, :])))):
            continue  # no valid pivot: node k is floating; rows below left as-is
        for r in range(k + 1, n_nodes):
            factor = working[r, k] / piv_val
            if factor != 0.0:
                working[r, k:] -= factor * working[k, k:]
    floating = []
    for k in range(n_nodes):
        scale = 1.0 + float(np.max(np.abs(working[k, :]))) if working[k, :].size else 1.0
        if abs(working[k, k]) <= tol * scale:
            floating.append(row_node[order[k]])
    return sorted(floating)
