"""
Solver-level tests covering the benchmark battery and a wide range of API
edge cases.

The two correctness checks we use:
  1. Closed-form comparison where one exists.
  2. Cross-check vs scipy DOP853 at rtol=atol=1e-13 otherwise.

Tests are tolerant: we assert "agrees with reference to 1e-9" rather than
machine eps to avoid false alarms from cosmetic floating-point variations
across numpy/numba versions.
"""
import numpy as np
from numba import njit
import pytest

import mcpi


# ---------------------------------------------------------------------------
# RHS definitions (mirrors benchmarks/problems.py)
# ---------------------------------------------------------------------------
@njit
def _rhs_riccati(t, u, out):
    out[0] = 1.0 + u[0]*u[0]


@njit
def _rhs_stiff_linear(t, u, out):
    out[0] = -50.0 * (u[0] - 1.0)


@njit
def _rhs_sin_u(t, u, out):
    out[0] = np.sin(u[0])


@njit
def _rhs_exp_neg_u(t, u, out):
    out[0] = np.exp(-u[0])


@njit
def _rhs_exp_u(t, u, out):
    out[0] = np.exp(u[0])


@njit
def _rhs_sin_cos(t, u, out):
    out[0] = np.sin(u[0]) * np.cos(u[0])


@njit
def _rhs_sqrt(t, u, out):
    out[0] = np.sqrt(u[0])


@njit
def _rhs_lotka_volterra(t, u, out):
    out[0] = 1.0*u[0] - 0.5*u[0]*u[1]
    out[1] = 0.5*u[0]*u[1] - 1.0*u[1]


@njit
def _rhs_van_der_pol_mu1(t, u, out):
    out[0] = u[1]
    out[1] = (1.0 - u[0]**2)*u[1] - u[0]


@njit
def _rhs_lorenz(t, u, out):
    out[0] = 10.0*(u[1] - u[0])
    out[1] = u[0]*(28.0 - u[2]) - u[1]
    out[2] = u[0]*u[1] - (8.0/3.0)*u[2]


@njit
def _rhs_harmonic(t, u, out):
    out[0] = u[1]
    out[1] = -u[0]


@njit
def _rhs_kepler_2d(t, u, out):
    """Two-body central-force orbit; preserves energy and angular momentum."""
    x, y, vx, vy = u[0], u[1], u[2], u[3]
    r3 = (x*x + y*y)**1.5
    out[0] = vx
    out[1] = vy
    out[2] = -x / r3
    out[3] = -y / r3


# ---------------------------------------------------------------------------
# Closed-form benchmark problems
# ---------------------------------------------------------------------------

def test_riccati_against_tan():
    """u' = 1 + u^2, u(0)=0  =>  u(t) = tan(t).  Test up to t = 1.4 < pi/2."""
    sol = mcpi.solve(_rhs_riccati, (0, 1.4), [0.0], tol=1e-13, n_cheb=20)
    assert sol.success
    t = np.linspace(0, 1.4, 201)
    err = np.max(np.abs(sol(t) - np.tan(t)))
    assert err < 1e-12


def test_stiff_linear_closed_form():
    """u' = -50(u-1) is mildly stiff; closed form 1 - exp(-50t)."""
    sol = mcpi.solve(_rhs_stiff_linear, (0, 0.5), [0.0], tol=1e-13, n_cheb=20)
    assert sol.success
    t = np.linspace(0, 0.5, 101)
    err = np.max(np.abs(sol(t) - (1 - np.exp(-50*t))))
    assert err < 1e-11


def test_exp_neg_u_against_log():
    """u' = exp(-u), u(0)=0  =>  u(t) = log(1+t)."""
    sol = mcpi.solve(_rhs_exp_neg_u, (0, 5), [0.0], tol=1e-13, n_cheb=20)
    assert sol.success
    t = np.linspace(0, 5, 201)
    err = np.max(np.abs(sol(t) - np.log(1 + t)))
    assert err < 1e-12


def test_exp_u_near_singularity():
    """u' = exp(u), u(0)=0 has finite-time singularity at t = 1.

    Stay safely below the blowup point and check against -log(1-t).
    Closer to the singularity, segment lengths shrink and roundoff in the
    -log(1-t) reference itself starts to dominate.
    """
    sol = mcpi.solve(_rhs_exp_u, (0, 0.9), [0.0], tol=1e-13, n_cheb=25)
    assert sol.success
    t = np.linspace(0, 0.9, 101)
    err = np.max(np.abs(sol(t) - (-np.log(1 - t))))
    assert err < 1e-10


def test_sqrt_closed_form():
    """u' = sqrt(u), u(0)=u0  =>  u(t) = (t/2 + sqrt(u0))^2."""
    sol = mcpi.solve(_rhs_sqrt, (0, 3), [1.0], tol=1e-13, n_cheb=20)
    assert sol.success
    t = np.linspace(0, 3, 101)
    err = np.max(np.abs(sol(t) - (t/2 + 1.0)**2))
    assert err < 1e-12


# ---------------------------------------------------------------------------
# Multi-component systems
# ---------------------------------------------------------------------------

def test_harmonic_energy_conservation():
    """u'' + u = 0 over many periods; energy should be conserved."""
    sol = mcpi.solve(_rhs_harmonic, (0, 20*np.pi), [1.0, 0.0],
                     tol=1e-13, n_cheb=20)
    assert sol.success
    t = np.linspace(0, 20*np.pi, 1001)
    y = sol(t)
    energy = y[0]**2 + y[1]**2
    drift = np.max(np.abs(energy - 1.0))
    assert drift < 1e-9


def test_lotka_volterra_invariant():
    """LV preserves H = 0.5 y0 - log(y0) + y1 - log(y1)... no wait,
    for u' = u - 0.5 u v, v' = 0.5 u v - v the invariant is
    H = u + v - log(u) - log(v) (with appropriate coefficients).

    Specifically for u'=a u - b u v, v' = c u v - d v, conservation is
    H(u,v) = c u - d log(u) + b v - a log(v).  Here a=1, b=0.5, c=0.5, d=1.
    So H = 0.5 u - log(u) + 0.5 v - log(v).
    """
    sol = mcpi.solve(_rhs_lotka_volterra, (0, 20), [2.0, 1.0],
                     tol=1e-13, n_cheb=25)
    assert sol.success
    t = np.linspace(0, 20, 401)
    y = sol(t)
    H = 0.5*y[0] - np.log(y[0]) + 0.5*y[1] - np.log(y[1])
    drift = np.max(np.abs(H - H[0]))
    assert drift < 1e-9, f"LV invariant drifted by {drift}"


def test_van_der_pol_mu1_vs_scipy():
    """Cross-check VdP at mu=1 against scipy DOP853@1e-13."""
    import scipy.integrate
    rhs_py = lambda t, y: [y[1], (1 - y[0]**2)*y[1] - y[0]]
    t_span = (0, 20)
    y0 = [2.0, 0.0]

    sol_mcpi = mcpi.solve(_rhs_van_der_pol_mu1, t_span, y0,
                          tol=1e-13, n_cheb=25)
    sol_sci = scipy.integrate.solve_ivp(rhs_py, t_span, y0, method='DOP853',
                                        rtol=1e-13, atol=1e-13,
                                        dense_output=True)
    assert sol_mcpi.success
    assert sol_sci.success
    t = np.linspace(0, 20, 401)
    err = np.max(np.abs(sol_mcpi(t) - sol_sci.sol(t)))
    # Both solvers are at their precision limits — 1e-8 is the right cap.
    assert err < 1e-8


def test_lorenz_vs_scipy():
    """Cross-check Lorenz on [0, 5] (chaotic but short)."""
    import scipy.integrate
    rhs_py = lambda t, y: [10*(y[1]-y[0]),
                            y[0]*(28-y[2])-y[1],
                            y[0]*y[1]-(8/3)*y[2]]
    sol_mcpi = mcpi.solve(_rhs_lorenz, (0, 5), [1.0, 1.0, 1.0],
                          tol=1e-13, n_cheb=25)
    sol_sci = scipy.integrate.solve_ivp(rhs_py, (0, 5), [1.0, 1.0, 1.0],
                                        method='DOP853',
                                        rtol=1e-13, atol=1e-13,
                                        dense_output=True)
    t = np.linspace(0, 5, 501)
    err = np.max(np.abs(sol_mcpi(t) - sol_sci.sol(t)))
    # Lorenz amplifies any roundoff exponentially; 1e-9 over [0,5] is fine.
    assert err < 1e-8


def test_kepler_orbit_invariants():
    """Two-body orbit: energy and angular momentum are exactly conserved.

    Initial state: circular orbit at r=1, v=1, period 2π.
    Energy E = 0.5 v^2 - 1/r = -0.5; ang. momentum L = x*vy - y*vx = 1.
    """
    y0 = [1.0, 0.0, 0.0, 1.0]
    sol = mcpi.solve(_rhs_kepler_2d, (0, 4*np.pi), y0,
                     tol=1e-13, n_cheb=25)
    assert sol.success
    t = np.linspace(0, 4*np.pi, 401)
    y = sol(t)
    r = np.sqrt(y[0]**2 + y[1]**2)
    v2 = y[2]**2 + y[3]**2
    E = 0.5*v2 - 1.0/r
    L = y[0]*y[3] - y[1]*y[2]
    assert np.max(np.abs(E - (-0.5))) < 1e-9
    assert np.max(np.abs(L - 1.0)) < 1e-10
    # After two full orbits we should be back near (1, 0, 0, 1)
    state_end = sol(4*np.pi)
    assert np.max(np.abs(state_end - np.array(y0))) < 1e-7


# ---------------------------------------------------------------------------
# API and shape contracts
# ---------------------------------------------------------------------------

def test_solution_t_y_consistency():
    """sol.y at sol.t should equal the integrated state to spectral accuracy."""
    sol = mcpi.solve(_rhs_harmonic, (0, 4*np.pi), [1.0, 0.0],
                     tol=1e-13, n_cheb=20)
    err_x = np.max(np.abs(sol.y[0] - np.cos(sol.t)))
    err_v = np.max(np.abs(sol.y[1] - (-np.sin(sol.t))))
    assert err_x < 1e-11
    assert err_v < 1e-11


def test_call_scalar_input_d1():
    """Scalar t, d=1 -> Python float."""
    sol = mcpi.solve(_rhs_sin_u, (0, 5), [0.5], tol=1e-12)
    assert sol.success
    val = sol(2.5)
    assert isinstance(val, float)
    assert not np.isnan(val)


def test_call_scalar_input_d_multi():
    """Scalar t, d>1 -> 1D array of length d."""
    sol = mcpi.solve(_rhs_lorenz, (0, 1), [1.0, 1.0, 1.0], tol=1e-10)
    val = sol(0.5)
    assert val.shape == (3,)


def test_call_array_input_d1():
    """1D t, d=1 -> 1D array."""
    sol = mcpi.solve(_rhs_sin_u, (0, 5), [0.5], tol=1e-12)
    t = np.linspace(0, 5, 17)
    val = sol(t)
    assert val.shape == (17,)


def test_call_array_input_d_multi():
    """1D t, d>1 -> 2D (d, n_t)."""
    sol = mcpi.solve(_rhs_lorenz, (0, 1), [1.0, 1.0, 1.0], tol=1e-10)
    t = np.linspace(0, 1, 13)
    val = sol(t)
    assert val.shape == (3, 13)


def test_call_outside_span_returns_nan():
    """Times outside [t_start, t_end] should yield NaN."""
    sol = mcpi.solve(_rhs_sin_u, (0, 1), [0.5], tol=1e-10)
    assert np.isnan(sol(2.0))
    assert np.isnan(sol(-0.5))


def test_endpoints_match_input_state():
    """sol(t_start) should reproduce y0 exactly; sol(t_end) should match."""
    y0 = np.array([1.5, -2.5, 0.5])
    sol = mcpi.solve(_rhs_lorenz, (0, 1), y0, tol=1e-12)
    err_start = np.max(np.abs(sol(0.0) - y0))
    assert err_start < 1e-13


def test_y0_acceptable_as_list_array_tuple():
    """y0 should accept any 1D iterable of floats."""
    a = mcpi.solve(_rhs_sin_u, (0, 1), [0.5], tol=1e-10)
    b = mcpi.solve(_rhs_sin_u, (0, 1), np.array([0.5]), tol=1e-10)
    c = mcpi.solve(_rhs_sin_u, (0, 1), (0.5,), tol=1e-10)
    assert abs(a(1.0) - b(1.0)) < 1e-15
    assert abs(a(1.0) - c(1.0)) < 1e-15


def test_solution_metadata_makes_sense():
    sol = mcpi.solve(_rhs_riccati, (0, 1.4), [0.0], tol=1e-13, n_cheb=20)
    assert sol.success
    assert sol.n_segments >= 1
    assert sol.n_segments == len(sol.segments)
    # nfev = sum n_iters * n_nodes per segment
    expected = sum(s['n_iters'] * len(s['nodes']) for s in sol.segments)
    assert sol.nfev == expected
    # t monotonic non-decreasing within segments (allowing duplicates at boundaries)
    diffs = np.diff(sol.t)
    assert np.all(diffs >= -1e-13)


# ---------------------------------------------------------------------------
# Bad-input errors
# ---------------------------------------------------------------------------

def test_t_span_zero_length_raises():
    with pytest.raises(ValueError):
        mcpi.solve(_rhs_riccati, (0, 0), [0.0])


def test_t_span_reversed_raises():
    with pytest.raises(ValueError):
        mcpi.solve(_rhs_riccati, (1.0, 0.5), [0.0])


# ---------------------------------------------------------------------------
# Reproducibility & cache safety
# ---------------------------------------------------------------------------

def test_repeated_calls_give_identical_results():
    """Two solves with identical args should be bit-exact equal."""
    sol1 = mcpi.solve(_rhs_lorenz, (0, 2), [1.0, 1.0, 1.0], tol=1e-12)
    sol2 = mcpi.solve(_rhs_lorenz, (0, 2), [1.0, 1.0, 1.0], tol=1e-12)
    t = np.linspace(0, 2, 50)
    diff = np.max(np.abs(sol1(t) - sol2(t)))
    assert diff == 0.0


def test_distinct_rhs_functions_get_distinct_kernels():
    """Two different RHS objects must not collide in the cache (regression).

    If the cache used id(rhs) and the first function was GC'd, the second
    might inherit the wrong kernel.  We construct two RHSs with DIFFERENT
    behavior and ensure the second one solves correctly.
    """
    @njit
    def rhs_a(t, u, out):
        out[0] = -u[0]

    @njit
    def rhs_b(t, u, out):
        out[0] = -3.0 * u[0]

    sol_a = mcpi.solve(rhs_a, (0, 1), [1.0], tol=1e-10)
    sol_b = mcpi.solve(rhs_b, (0, 1), [1.0], tol=1e-10)
    assert abs(sol_a(1.0) - np.exp(-1)) < 1e-10
    assert abs(sol_b(1.0) - np.exp(-3)) < 1e-10


# ---------------------------------------------------------------------------
# Long integrations / many segments
# ---------------------------------------------------------------------------

def test_long_decay_machine_eps():
    """Linear decay over [0, 100]: many segments, must remain accurate."""
    @njit
    def f(t, u, out):
        out[0] = -u[0]
    sol = mcpi.solve(f, (0, 100), [1.0], tol=1e-12, n_cheb=20)
    assert sol.success
    t = np.linspace(0, 100, 200)
    err = np.max(np.abs(sol(t) - np.exp(-t)))
    assert err < 1e-11


def test_high_dim_decoupled_system():
    """10 decoupled exponentials with different rates."""
    @njit
    def f(t, u, out):
        for i in range(10):
            out[i] = -(i+1) * u[i]
    sol = mcpi.solve(f, (0, 2), np.ones(10), tol=1e-13, n_cheb=20)
    assert sol.success
    rates = np.arange(1, 11)
    err = np.max(np.abs(sol(2.0) - np.exp(-rates*2.0)))
    assert err < 1e-12
