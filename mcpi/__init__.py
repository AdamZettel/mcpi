"""
mcpi — Modified Chebyshev-Picard Iteration in Python.

A fast Python implementation of the Cheb-Picard ODE integration method
introduced by Bai & Junkins (2010-2011) and extended by Macomber, Kim,
Probe, Woollands and others.  The inner kernel uses numba JIT, the
multistage adaptive driver is identical in spirit to scipy's adaptive
RK methods, and the public API mimics scipy.integrate.solve_ivp where
practical.

Typical usage:

    import numpy as np
    from numba import njit
    import mcpi

    @njit
    def lorenz(t, u, out):
        out[0] = 10 * (u[1] - u[0])
        out[1] = u[0] * (28 - u[2]) - u[1]
        out[2] = u[0] * u[1] - (8/3) * u[2]

    sol = mcpi.solve(lorenz, t_span=(0, 5), y0=[1.0, 1.0, 1.0], tol=1e-13)
    print(sol(2.5))   # state at t = 2.5
"""
from . solver import solve, Solution
from . _chebyshev import chebyshev_setup, barycentric_eval

__version__ = "0.1.0"
__all__ = ["solve", "Solution", "chebyshev_setup", "barycentric_eval"]
