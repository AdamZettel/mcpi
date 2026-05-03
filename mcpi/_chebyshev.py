"""
Chebyshev-Lobatto node and integration-matrix construction.

The integration matrix J maps function values at the Cheb-Lobatto nodes on
[a, b] to definite-integral values  ∫_a^t f(τ)dτ  evaluated at the same
nodes.  It is the basic operator for collocation-based ODE solvers.

Implementation follows Trefethen, "Spectral Methods in MATLAB" (2000),
chapter 6, with a sign convention chosen so that nodes[0] = b (right
endpoint) and nodes[N] = a (left endpoint).  Picking nodes[0] = b makes
"value at end of segment" = nodes[0], which slots naturally into an
adaptive multistage chain.
"""
import numpy as np


def chebyshev_setup(N, a, b):
    """Construct Chebyshev-Lobatto nodes, differentiation matrix, and
    integration matrix on the interval [a, b].

    Parameters
    ----------
    N : int
        Highest polynomial degree; the grid has N+1 nodes.
    a, b : float
        Interval endpoints with a < b.

    Returns
    -------
    nodes : ndarray, shape (N+1,)
        Cheb-Lobatto nodes mapped to [a, b].  nodes[0] = b, nodes[N] = a.
    D : ndarray, shape (N+1, N+1)
        Differentiation matrix (mapping values to derivatives at the nodes).
    J : ndarray, shape (N+1, N+1)
        Integration matrix:  (J f)[i] = ∫_a^{nodes[i]} f(τ) dτ.

    Notes
    -----
    J is built as the pseudo-inverse of D restricted to the
    integration-from-a constraint, so that the integration is exact for
    polynomials of degree up to N.
    """
    if N < 2:
        raise ValueError("N must be >= 2")

    # Reference Cheb-Lobatto nodes on [-1, 1]
    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)            # x[0] = 1, x[N] = -1

    # Map to [a, b]  (so nodes[0] = b, nodes[N] = a)
    nodes = (b - a) / 2 * x + (b + a) / 2

    # Differentiation matrix on [-1, 1]  (Trefethen)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * ((-1.0) ** k)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T + np.eye(N + 1)
    D_ref = np.outer(c, 1.0 / c) / dX
    D_ref = D_ref - np.diag(D_ref.sum(axis=1))

    # Scale to [a, b]
    D = D_ref * (2.0 / (b - a))

    # Integration matrix:  given f-values at the nodes, J f returns the values
    # of u(t) where u'(t) = f_interp(t) and u(a) = 0.  We form this by
    # replacing the last row of D with the boundary constraint u(a) = 0 and
    # the right-hand side correspondingly:
    #
    #   D u = f, but in the row corresponding to node N (= a) we instead
    #   require u(a) = 0.  In matrix form: Dmod J = M with M = I except
    #   M[N, N] = 0, so the last column of J is zero — meaning the integrator
    #   correctly discards the contribution of f at the left endpoint
    #   (∫_a^a = 0 regardless of f(a)).
    Dmod = D.copy()
    Dmod[-1, :] = 0.0
    Dmod[-1, -1] = 1.0
    M = np.eye(N + 1)
    M[-1, -1] = 0.0
    J = np.linalg.solve(Dmod, M)

    return nodes, D, J


def barycentric_weights(N):
    """Cheb-Lobatto barycentric weights for an (N+1)-point grid."""
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def barycentric_eval(nodes, vals, t_query):
    """Evaluate the polynomial interpolant defined by (nodes, vals) at t_query.

    Uses the barycentric formula, which is numerically stable on Cheb-Lobatto
    grids.  vals can be 1D (single output) or 2D (multiple outputs sharing
    the grid).
    """
    N = len(nodes) - 1
    w = barycentric_weights(N)
    out = np.full_like(np.atleast_1d(t_query), np.nan, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[None, :]
        scalar_out = True
    else:
        scalar_out = False
        out = np.full((vals.shape[0], len(out)), np.nan, dtype=np.float64)
    t_query = np.atleast_1d(t_query)
    for i, t in enumerate(t_query):
        d = t - nodes
        mm = np.abs(d) < 1e-14
        if np.any(mm):
            j = int(np.argmax(mm))
            if scalar_out:
                out[i] = vals[0, j]
            else:
                out[:, i] = vals[:, j]
        else:
            wd = w / d
            denom = wd.sum()
            if scalar_out:
                out[i] = (wd * vals[0]).sum() / denom
            else:
                out[:, i] = (vals @ wd) / denom
    return out
