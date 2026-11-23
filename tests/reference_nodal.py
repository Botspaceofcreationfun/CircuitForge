"""Deterministic reference implementation for cross-checking the MNA solver.

This module implements the same circuits with *different* numerical methods
than the package: dense Gaussian elimination for the linear systems and an
independent nodal-analysis assembly written from the stamping rules, used only
in tests.  It is deliberately NOT part of the shipped package API.
"""

import numpy as np


def nodal_analysis_dc(nodes, resistors, vsources, isources):
    """Solve a resistive DC network by plain nodal analysis.

    Parameters (nodes are 1-based; ground is 0):
      nodes     : list of non-ground node ids
      resistors : list of (n1, n2, R)
      vsources  : list of (n1, n2, V)   (treated as ideal; uses MNA rows)
      isources  : list of (n1, n2, I)   (current flows n1 -> n2)
    Returns (node_voltages dict, branch_currents dict).
    """
    node_rows = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    n_vs = len(vsources)
    n = n_nodes + n_vs
    A = np.zeros((n, n))
    z = np.zeros(n)

    def rrow(nid):
        return node_rows[nid] if nid != 0 else None

    for n1, n2, r in resistors:
        g = 1.0 / r
        i, j = rrow(n1), rrow(n2)
        if i is not None:
            A[i, i] += g
        if j is not None:
            A[j, j] += g
        if i is not None and j is not None:
            A[i, j] -= g
            A[j, i] -= g
    for n1, n2, i in isources:
        ri, rj = rrow(n1), rrow(n2)
        if ri is not None:
            z[ri] -= i
        if rj is not None:
            z[rj] += i
    for k, (n1, n2, v) in enumerate(vsources):
        col = n_nodes + k
        ri, rj = rrow(n1), rrow(n2)
        if ri is not None:
            A[ri, col] += 1.0
        if rj is not None:
            A[rj, col] -= 1.0
        if ri is not None:
            A[col, ri] += 1.0
        if rj is not None:
            A[col, rj] -= 1.0
        z[col] = v

    x = np.linalg.solve(A, z)
    v = {n: float(x[node_rows[n]]) for n in nodes}
    i = {}
    for n1, n2, r in resistors:
        v1 = v.get(n1, 0.0)
        v2 = v.get(n2, 0.0)
        i[(n1, n2, r)] = (v1 - v2) / r
    for k, (n1, n2, vv) in enumerate(vsources):
        i[("V", k)] = float(x[n_nodes + k])
    return v, i
