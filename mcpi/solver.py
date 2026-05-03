"""
Public API for MCPI.

The top-level `solve` function takes a JIT'd RHS, time interval, initial
condition, and tolerance, and returns a Solution object that can be evaluated
at any point in the integration interval via barycentric interpolation
between Cheb-Lobatto nodes.

Multistage adaptive segment sizing is identical in spirit to scipy's adaptive
RK methods: we try a step, accept it if Picard converged, and either grow
(if convergence was easy) or halve (if convergence failed).
"""
import numpy as np
from numba import njit
from . _chebyshev import chebyshev_setup, barycentric_weights
from . _kernel import make_kernel


# Cache of JIT'd kernels keyed on (state_dim, rhs_function).  The first call
# with a new (d, rhs) pair triggers a numba compile (~1 sec); subsequent calls
# with the same RHS reuse the cached compiled kernel.
#
# We key on the function object itself (not id(rhs)) because Python recycles
# id values after garbage collection: a freshly-defined RHS could collide with
# a stale entry and silently get the wrong compiled kernel.  Numba dispatchers
# are hashable, so this works.  The dict holds a strong reference to each RHS,
# which prevents id-reuse and is a non-issue in typical use (RHS lifetimes are
# program-scoped); call `_KERNEL_CACHE.clear()` if memory is ever a concern.
_KERNEL_CACHE = {}


def _get_kernel(d, rhs):
    key = (d, rhs)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = make_kernel(d)(rhs)
        _KERNEL_CACHE[key] = k
    return k


class Solution:
    """Result of an MCPI integration.

    Attributes
    ----------
    t : ndarray
        Times at which the state was sampled (one per Cheb node, all
        segments stitched).
    y : ndarray, shape (d, len(t))
        State at each sampled time.
    segments : list of dict
        Per-segment data: t_anchor, t_end, nodes, u_seg (shape (n_nodes, d)),
        n_iters.  Useful for diagnostics.
    success : bool
        True if integration reached the requested end time.
    nfev : int
        Total number of RHS evaluations.
    n_segments : int
        Number of segments used.
    """
    def __init__(self, segments, t_span, success, d):
        self.segments = segments
        self.t_span = t_span
        self.success = success
        self.d = d
        self.n_segments = len(segments)
        self.nfev = sum(s['n_iters'] * (len(s['nodes'])) for s in segments)

        # Build a flat (t, y) sampling at the union of all nodes.
        ts, ys = [], []
        for s in segments:
            # Reverse so times are ascending within the segment.
            order = np.argsort(s['nodes'])
            ts.append(s['nodes'][order] + s['t_anchor'])
            ys.append(s['u_seg'][order])
        self.t = np.concatenate(ts) if ts else np.array([])
        self.y = np.concatenate(ys, axis=0).T if ys else np.zeros((d, 0))

    def __call__(self, t):
        """Barycentric evaluation of the solution at t (scalar or array)."""
        t_in = np.asarray(t, dtype=np.float64)
        scalar_input = (t_in.ndim == 0)
        t = np.atleast_1d(t_in)
        out = np.full((self.d, len(t)), np.nan)
        for s in self.segments:
            mask = (t >= s['t_anchor'] - 1e-12) & (t <= s['t_end'] + 1e-12)
            if not np.any(mask):
                continue
            tau = t[mask] - s['t_anchor']
            nodes = s['nodes']
            vals = s['u_seg'].T              # shape (d, n_nodes)
            N = len(nodes) - 1
            w = barycentric_weights(N)
            for ti, tau_i in zip(np.where(mask)[0], tau):
                d_arr = tau_i - nodes
                mm = np.abs(d_arr) < 1e-14
                if np.any(mm):
                    out[:, ti] = vals[:, int(np.argmax(mm))]
                else:
                    wd = w / d_arr
                    out[:, ti] = (vals @ wd) / wd.sum()
        # Shape contract:
        #   scalar t in -> 1D (d,)  for d > 1, scalar for d == 1
        #   1D t in     -> 2D (d, n_t) for d > 1, 1D (n_t,) for d == 1
        if scalar_input:
            if self.d == 1:
                return float(out[0, 0])
            return out[:, 0]
        if self.d == 1:
            return out[0]
        return out


def solve(rhs, t_span, y0, tol=1e-12, n_cheb=20, max_iter=None,
          t_initial=None, max_segments=200000):
    """Integrate an ODE system using Modified Chebyshev-Picard Iteration.

    Parameters
    ----------
    rhs : callable
        @njit'd function with signature  rhs(t, u, out) -> None  that writes
        f(t, u) into the preallocated `out` buffer.  Both u and out are 1-D
        numpy arrays of length d.
    t_span : (float, float)
        (t_start, t_end).  Integration proceeds from t_start to t_end.
    y0 : array-like
        Initial state; len(y0) determines the system dimension d.
    tol : float, default 1e-12
        Convergence tolerance for Picard iteration within each segment.
        Practical floor is roughly 1e-14 due to roundoff.
    n_cheb : int, default 20
        Polynomial degree per segment; the segment uses n_cheb+1 collocation
        nodes.  Higher gives spectral convergence but more work per iteration.
    max_iter : int, optional
        Maximum Picard iterations per segment.  Default min(n_cheb, 30).
    t_initial : float, optional
        Initial trial step length.  Default min(0.4, (t_end-t_start)/4).
    max_segments : int, default 200000
        Safety cap on number of segments.

    Returns
    -------
    Solution
        Object with .t, .y, .segments, .success, .nfev, plus call interface
        sol(t) for barycentric evaluation at arbitrary times.
    """
    t_start, t_end = float(t_span[0]), float(t_span[1])
    if t_end <= t_start:
        raise ValueError("t_span must satisfy t_end > t_start")
    y0 = np.asarray(y0, dtype=np.float64)
    d = y0.shape[0]
    if max_iter is None:
        max_iter = min(int(n_cheb), 30)
    if t_initial is None:
        t_initial = min(0.4, (t_end - t_start) / 4.0)

    kernel = _get_kernel(d, rhs)

    t_anchor = t_start
    u_anchor = y0.copy()
    segments = []
    t_try = t_initial
    success = False

    for _ in range(max_segments):
        if t_anchor >= t_end - 1e-12:
            success = True
            break
        t_try = min(t_try, t_end - t_anchor)

        conv = False
        for _attempt in range(60):
            nodes, _D, J = chebyshev_setup(n_cheb, 0.0, t_try)
            u_seg, conv, n_iters = kernel(
                J, nodes, t_anchor, u_anchor, max_iter, tol)
            if conv:
                break
            t_try *= 0.5
            if t_try < 1e-13:
                break

        if not conv:
            break

        # Right-endpoint value: with our chebyshev_setup, nodes[0] = t_try.
        u_anchor = u_seg[0].copy()
        segments.append({
            't_anchor': t_anchor,
            't_end': t_anchor + t_try,
            'nodes': nodes,
            'u_seg': u_seg.copy(),
            'n_iters': n_iters,
        })
        t_anchor += t_try
        # Mild growth on success (capped by remaining interval next loop).
        t_try *= 1.4

    return Solution(segments, (t_start, t_end), success, d)
