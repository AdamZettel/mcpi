"""
Benchmark problem battery for MCPI vs scipy comparison.

Each problem is defined by:
  - name: short identifier
  - desc: human-readable description
  - dim: state dimension
  - rhs: numba @njit'd RHS function (t, u, out) -> None, in-place
  - rhs_scipy: equivalent pure-Python function (t, y) -> list, for scipy
  - y0: initial condition
  - t_span: (t_start, t_end)
  - reference: dict with one of
      'closed_form': callable(t, y0) -> exact y(t)  (preferred)
      'mpmath_rhs': mpmath-compatible RHS for high-precision odefun
      'scipy_only': True if no closed-form and mpmath is too slow
  - category: 'polynomial', 'transcendental', 'rational', etc.
  - regime: 'mild', 'stiff', 'oscillatory', 'near-singularity', 'chaotic'
"""
import numpy as np
from numba import njit
import mpmath
from mpmath import mp, mpf


# ---------------------------------------------------------------------
# Scalar problems
# ---------------------------------------------------------------------
@njit(cache=False)
def _rhs_riccati(t, u, out):           out[0] = 1.0 + u[0]*u[0]
@njit(cache=False)
def _rhs_stiff_linear(t, u, out):      out[0] = -50.0 * (u[0] - 1.0)
@njit(cache=False)
def _rhs_cubic_forced(t, u, out):      out[0] = u[0] - u[0]**3 + np.sin(t)
@njit(cache=False)
def _rhs_sin_u(t, u, out):             out[0] = np.sin(u[0])
@njit(cache=False)
def _rhs_exp_neg_u(t, u, out):         out[0] = np.exp(-u[0])
@njit(cache=False)
def _rhs_exp_u(t, u, out):             out[0] = np.exp(u[0])
@njit(cache=False)
def _rhs_rational(t, u, out):          out[0] = u[0] / (1.0 + u[0]**2)
@njit(cache=False)
def _rhs_log(t, u, out):               out[0] = np.log(1.0 + u[0])
@njit(cache=False)
def _rhs_sin_cos(t, u, out):           out[0] = np.sin(u[0]) * np.cos(u[0])
@njit(cache=False)
def _rhs_exp_sin(t, u, out):           out[0] = np.exp(-u[0]) * np.sin(u[0])
@njit(cache=False)
def _rhs_sqrt(t, u, out):              out[0] = np.sqrt(u[0])
@njit(cache=False)
def _rhs_stiff_oscillating(t, u, out): out[0] = -50.0 * (u[0] - np.cos(t))


# ---------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------
@njit(cache=False)
def _rhs_lotka_volterra(t, u, out):
    out[0] = 1.0*u[0] - 0.5*u[0]*u[1]
    out[1] = 0.5*u[0]*u[1] - 1.0*u[1]


@njit(cache=False)
def _rhs_brusselator(t, u, out):
    out[0] = 1.0 + u[0]**2*u[1] - 4.0*u[0]
    out[1] = 3.0*u[0] - u[0]**2*u[1]


@njit(cache=False)
def _rhs_van_der_pol_mu1(t, u, out):
    out[0] = u[1]
    out[1] = (1.0 - u[0]**2)*u[1] - u[0]


@njit(cache=False)
def _rhs_van_der_pol_mu10(t, u, out):
    out[0] = u[1]
    out[1] = 10.0*(1.0 - u[0]**2)*u[1] - u[0]


@njit(cache=False)
def _rhs_lorenz(t, u, out):
    out[0] = 10.0*(u[1] - u[0])
    out[1] = u[0]*(28.0 - u[2]) - u[1]
    out[2] = u[0]*u[1] - (8.0/3.0)*u[2]


@njit(cache=False)
def _rhs_robertson(t, u, out):
    out[0] = -0.04*u[0] + 1e4*u[1]*u[2]
    out[1] = 0.04*u[0] - 1e4*u[1]*u[2] - 3e7*u[1]**2
    out[2] = 3e7*u[1]**2


# Closed-form solutions where they exist
_CLOSED = {
    'riccati':            lambda t, y0: np.tan(t + np.arctan(y0[0])),
    'stiff_linear':       lambda t, y0: 1 - (1 - y0[0])*np.exp(-50*t),
    'sin_u':              lambda t, y0: 2*np.arctan(np.tan(y0[0]/2)*np.exp(t)),
    'exp_neg_u':          lambda t, y0: np.log(t + np.exp(y0[0])),
    'exp_u':              lambda t, y0: -np.log(np.exp(-y0[0]) - t),
    'sin_cos':            lambda t, y0: np.arctan(np.tan(y0[0])*np.exp(t)),
    'sqrt':               lambda t, y0: (t/2 + np.sqrt(y0[0]))**2,
}


def _scipy_rhs(name):
    """Pure-Python (no numba) RHS for scipy and mpmath calls."""
    R = {
        'riccati':            lambda t, y: [1 + y[0]**2],
        'stiff_linear':       lambda t, y: [-50*(y[0]-1)],
        'cubic_forced':       lambda t, y: [y[0] - y[0]**3 + np.sin(t)],
        'sin_u':              lambda t, y: [np.sin(y[0])],
        'exp_neg_u':          lambda t, y: [np.exp(-y[0])],
        'exp_u':              lambda t, y: [np.exp(y[0])],
        'rational':           lambda t, y: [y[0]/(1+y[0]**2)],
        'log':                lambda t, y: [np.log(1+y[0])],
        'sin_cos':            lambda t, y: [np.sin(y[0])*np.cos(y[0])],
        'exp_sin':            lambda t, y: [np.exp(-y[0])*np.sin(y[0])],
        'sqrt':               lambda t, y: [np.sqrt(y[0])],
        'stiff_oscillating':  lambda t, y: [-50*(y[0]-np.cos(t))],
        'lotka_volterra':     lambda t, y: [y[0]-0.5*y[0]*y[1], 0.5*y[0]*y[1]-y[1]],
        'brusselator':        lambda t, y: [1+y[0]**2*y[1]-4*y[0], 3*y[0]-y[0]**2*y[1]],
        'van_der_pol_mu1':    lambda t, y: [y[1], (1-y[0]**2)*y[1]-y[0]],
        'van_der_pol_mu10':   lambda t, y: [y[1], 10*(1-y[0]**2)*y[1]-y[0]],
        'lorenz':             lambda t, y: [10*(y[1]-y[0]), y[0]*(28-y[2])-y[1],
                                              y[0]*y[1]-(8/3)*y[2]],
        'robertson':          lambda t, y: [-0.04*y[0]+1e4*y[1]*y[2],
                                              0.04*y[0]-1e4*y[1]*y[2]-3e7*y[1]**2,
                                              3e7*y[1]**2],
    }
    return R[name]


def _mpmath_rhs(name):
    """mpmath-compatible RHS for odefun reference computation."""
    R = {
        'cubic_forced':       lambda t, y: [y[0] - y[0]**3 + mpmath.sin(t)],
        'rational':           lambda t, y: [y[0]/(1+y[0]**2)],
        'log':                lambda t, y: [mpmath.log(1+y[0])],
        'exp_sin':            lambda t, y: [mpmath.exp(-y[0])*mpmath.sin(y[0])],
        'stiff_oscillating':  lambda t, y: [-50*(y[0]-mpmath.cos(t))],
        'lotka_volterra':     lambda t, y: [y[0]-mpf('0.5')*y[0]*y[1], mpf('0.5')*y[0]*y[1]-y[1]],
        'brusselator':        lambda t, y: [1+y[0]**2*y[1]-4*y[0], 3*y[0]-y[0]**2*y[1]],
        'van_der_pol_mu1':    lambda t, y: [y[1], (1-y[0]**2)*y[1]-y[0]],
        'van_der_pol_mu10':   lambda t, y: [y[1], 10*(1-y[0]**2)*y[1]-y[0]],
        'lorenz':             lambda t, y: [10*(y[1]-y[0]), y[0]*(28-y[2])-y[1],
                                              y[0]*y[1]-(mpf(8)/mpf(3))*y[2]],
    }
    return R.get(name)


PROBLEMS = [
    # name,                dim, rhs (numba),               y0,                 t_span,       category,           regime
    ('riccati',            1, _rhs_riccati,           [0.0],             (0.0, 1.4),   'polynomial',       'near-singularity'),
    ('stiff_linear',       1, _rhs_stiff_linear,      [0.0],             (0.0, 0.5),   'polynomial',       'stiff'),
    ('cubic_forced',       1, _rhs_cubic_forced,      [0.5],             (0.0, 5.0),   'polynomial+t',     'mild'),
    ('sin_u',              1, _rhs_sin_u,             [0.5],             (0.0, 10.0),  'transcendental',   'mild'),
    ('exp_neg_u',          1, _rhs_exp_neg_u,         [0.0],             (0.0, 5.0),   'transcendental',   'mild'),
    ('exp_u',              1, _rhs_exp_u,             [0.0],             (0.0, 0.95),  'transcendental',   'near-singularity'),
    ('rational',           1, _rhs_rational,          [0.3],             (0.0, 5.0),   'rational',         'mild'),
    ('log',                1, _rhs_log,               [0.5],             (0.0, 8.0),   'transcendental',   'mild'),
    ('sin_cos',            1, _rhs_sin_cos,           [0.3],             (0.0, 5.0),   'transcendental',   'mild'),
    ('exp_sin',            1, _rhs_exp_sin,           [1.0],             (0.0, 8.0),   'transcendental',   'mild'),
    ('sqrt',               1, _rhs_sqrt,              [1.0],             (0.0, 3.0),   'fractional-power', 'mild'),
    ('stiff_oscillating',  1, _rhs_stiff_oscillating, [0.0],             (0.0, 2.0),   'polynomial+t',     'stiff-oscillating'),
    ('lotka_volterra',     2, _rhs_lotka_volterra,    [2.0, 1.0],        (0.0, 20.0),  'polynomial-system','oscillator'),
    ('brusselator',        2, _rhs_brusselator,       [1.5, 3.0],        (0.0, 15.0),  'polynomial-system','oscillator'),
    ('van_der_pol_mu1',    2, _rhs_van_der_pol_mu1,   [2.0, 0.0],        (0.0, 20.0),  'polynomial-system','oscillator'),
    ('van_der_pol_mu10',   2, _rhs_van_der_pol_mu10,  [2.0, 0.0],        (0.0, 20.0),  'polynomial-system','mildly-stiff'),
    ('lorenz',             3, _rhs_lorenz,            [1.0, 1.0, 1.0],   (0.0, 5.0),   'polynomial-system','chaotic'),
    ('robertson',          3, _rhs_robertson,         [1.0, 0.0, 0.0],   (0.0, 100.0), 'polynomial-system','extreme-stiff'),
]


def get_reference(name, y0, t_span, t_eval, mpmath_dps=40):
    """Compute high-precision reference solution at t_eval times.

    Strategy:
      1. closed-form if available -> exact
      2. mpmath at high dps if rhs is mpmath-compatible -> ~10^-30 reference
      3. scipy DOP853 at rtol=atol=1e-13 -> ~10^-12 reference (last resort)
    """
    if name in _CLOSED:
        return _CLOSED[name](t_eval, np.asarray(y0)), 'closed-form'
    mp_rhs = _mpmath_rhs(name)
    if mp_rhs is not None:
        try:
            mp.dps = mpmath_dps
            f = mpmath.odefun(mp_rhs, mpf(0), [mpf(v) for v in y0])
            d = len(y0)
            U = np.zeros((d, len(t_eval)))
            for i, ti in enumerate(t_eval):
                vals = f(mpf(float(ti)))
                for k in range(d):
                    U[k, i] = float(vals[k])
            return U if d > 1 else U[0], f'mpmath@{mpmath_dps}'
        except Exception:
            pass
    # scipy fallback
    import scipy.integrate
    sol = scipy.integrate.solve_ivp(
        _scipy_rhs(name), t_span, y0, method='DOP853',
        rtol=1e-13, atol=1e-13, dense_output=True,
    )
    U = sol.sol(t_eval)
    return U if len(y0) > 1 else U[0], 'DOP853@1e-13'
