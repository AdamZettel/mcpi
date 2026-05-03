"""
Basic tests for mcpi.solve.

Run with:  pytest tests/
"""
import numpy as np
from numba import njit
import pytest

import mcpi


@njit
def linear(t, u, out):
    """u' = -u  =>  u(t) = u0 * exp(-t)"""
    out[0] = -u[0]


@njit
def sin_u(t, u, out):
    """u' = sin(u),  u(t) = 2 arctan(tan(u0/2) exp(t))"""
    out[0] = np.sin(u[0])


@njit
def harmonic(t, u, out):
    """u'' + u = 0  =>  u(t) = cos(t),  v(t) = -sin(t)"""
    out[0] = u[1]
    out[1] = -u[0]


@njit
def lorenz(t, u, out):
    out[0] = 10*(u[1] - u[0])
    out[1] = u[0]*(28 - u[2]) - u[1]
    out[2] = u[0]*u[1] - (8/3)*u[2]


def test_linear_decay_machine_eps():
    """u' = -u should hit machine epsilon on a moderate interval."""
    sol = mcpi.solve(linear, t_span=(0, 5), y0=[1.0], tol=1e-13, n_cheb=20)
    assert sol.success
    t_test = np.linspace(0, 5, 101)
    u_pred = sol(t_test)
    u_exact = np.exp(-t_test)
    assert np.max(np.abs(u_pred - u_exact)) < 1e-13


def test_sin_u_machine_eps():
    """u' = sin(u) on [0, 10] with closed-form reference."""
    u0 = 0.5
    sol = mcpi.solve(sin_u, t_span=(0, 10), y0=[u0], tol=1e-13, n_cheb=20)
    assert sol.success
    t_test = np.linspace(0, 10, 101)
    u_pred = sol(t_test)
    u_exact = 2 * np.arctan(np.tan(u0/2) * np.exp(t_test))
    assert np.max(np.abs(u_pred - u_exact)) < 1e-12


def test_harmonic_oscillator():
    """u'' + u = 0 on [0, 4π] preserves energy."""
    sol = mcpi.solve(harmonic, t_span=(0, 4*np.pi), y0=[1.0, 0.0],
                     tol=1e-13, n_cheb=20)
    assert sol.success
    t_test = np.linspace(0, 4*np.pi, 201)
    y = sol(t_test)
    energy = y[0]**2 + y[1]**2
    assert np.max(np.abs(energy - 1.0)) < 1e-11


def test_lorenz_runs():
    """Lorenz on [0, 5] should integrate without errors."""
    sol = mcpi.solve(lorenz, t_span=(0, 5), y0=[1.0, 1.0, 1.0],
                     tol=1e-12, n_cheb=20)
    assert sol.success
    assert sol.n_segments > 10
    state = sol(2.5)
    assert state.shape == (3,)
    assert not np.any(np.isnan(state))


def test_call_returns_correct_shape():
    """Scalar t -> 1D array (d,); 1D t -> 2D array (d, n)."""
    sol = mcpi.solve(harmonic, t_span=(0, 1), y0=[1.0, 0.0], tol=1e-10)
    # Scalar input
    s1 = sol(0.5)
    assert s1.shape == (2,)
    # 1D input
    sN = sol(np.linspace(0, 1, 10))
    assert sN.shape == (2, 10)


def test_endpoints_match_ic():
    """sol(t_start) should equal y0 to machine precision."""
    y0 = np.array([2.0, -3.0])
    sol = mcpi.solve(harmonic, t_span=(0, 5), y0=y0, tol=1e-12)
    assert np.max(np.abs(sol(0.0) - y0)) < 1e-13


def test_solve_negative_span_raises():
    with pytest.raises(ValueError):
        mcpi.solve(linear, t_span=(1, 0), y0=[1.0])


def test_solve_zero_span_raises():
    with pytest.raises(ValueError):
        mcpi.solve(linear, t_span=(1, 1), y0=[1.0])


def test_kernel_cache_speeds_up_repeat_calls():
    """Second call with same RHS should be faster than first (no recompile)."""
    import time
    # Warmup once
    mcpi.solve(linear, t_span=(0, 1), y0=[1.0], tol=1e-9, n_cheb=8, max_iter=5)
    # Measure
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        mcpi.solve(linear, t_span=(0, 1), y0=[1.0], tol=1e-9, n_cheb=8, max_iter=5)
        times.append(time.perf_counter() - t0)
    # Repeat calls should each take well under 100 ms (no JIT overhead)
    assert min(times) < 0.05


if __name__ == '__main__':
    import sys
    pytest.main([__file__, '-v'] + sys.argv[1:])
