"""CircuitForge — a small circuit simulator built from first principles on
modified nodal analysis (MNA).

Modules
-------
netlist      : documented netlist format and parser
mna          : MNA matrix assembly, linear solves, singularity diagnostics
nonlinear    : Newton-Raphson solver for diode circuits
transient    : companion-model transient integration (BE / trapezoidal)
analysis     : DC operating point and full netlist runs
plotting     : matplotlib plots of node voltages / branch currents
export       : CSV export
cli          : command-line interface and interactive netlist editor
"""

__version__ = "1.0.0"
