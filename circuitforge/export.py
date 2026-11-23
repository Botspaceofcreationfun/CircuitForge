"""CSV export of transient results.

The file contains one column for time (``t``, seconds) plus one column per
requested variable, using the SPICE-like names ``v(<node>)`` and
``i(<element>)``.
"""

from __future__ import annotations

import csv
from typing import List

from .analysis import parse_variable
from .transient import TranResult


def write_csv(result: TranResult, variables: List[str], filename: str) -> str:
    """Write time + requested variables to ``filename``; returns the path."""
    v_items, i_items = [], []
    for var in variables:
        kind, arg = parse_variable(var)
        if kind == "v":
            node = int(arg)
            if node not in result.v:
                raise ValueError(f"node {node} not present in the circuit")
            v_items.append((node, var))
        else:
            if arg not in result.i:
                raise ValueError(f"element {arg!r} not present in the circuit")
            i_items.append((arg, var))

    header = ["t"] + [var for _, var in v_items + i_items]
    with open(filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for k, tk in enumerate(result.t):
            row = [tk]
            for node, _ in v_items:
                row.append(float(result.v[node][k]))
            for name, _ in i_items:
                row.append(float(result.i[name][k]))
            writer.writerow(row)
    return filename
