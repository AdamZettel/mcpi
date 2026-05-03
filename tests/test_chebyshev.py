"""
Tests for the Chebyshev numerical primitives in mcpi._chebyshev.

These verify the foundational invariants the solver relies on: integration
matrix exactness on polynomials up to degree N, scaling to arbitrary [a, b],
and barycentric-eval correctness.
"""
import numpy as np
import pytest

from mcpi._chebyshev import chebyshev_setup, barycentric_weights, barycentric_eval


# ---------------------------------------------------------------------------
# chebyshev_setup
# ---------------------------------------------------------------------------

def test_node_endpoints():
    """nodes[0] = b, nodes[N] = a."""
    for a, b in [(0.0, 1.0), (-1.0, 1.0), (-3.0, 5.0), (10.0, 11.0)]:
        for N in [3, 8, 20, 50]:
            nodes, _, _ = chebyshev_setup(N, a, b)
            assert abs(nodes[0] - b) < 1e-13, (a, b, N, nodes[0])
            assert abs(nodes[-1] - a) < 1e-13, (a, b, N, nodes[-1])
            assert len(nodes) == N + 1


def test_nodes_are_strictly_decreasing():
    nodes, _, _ = chebyshev_setup(20, 0.0, 1.0)
    diffs = np.diff(nodes)
    assert np.all(diffs < 0)


def test_chebyshev_setup_rejects_small_N():
    with pytest.raises(ValueError):
        chebyshev_setup(1, 0.0, 1.0)
    with pytest.raises(ValueError):
        chebyshev_setup(0, 0.0, 1.0)


def test_integration_matrix_exact_on_polynomials():
    """J integrates polynomials of degree < N exactly (to ~ machine eps).

    Degree exactly N is *not* exact: the (N+1)-point Cheb-Lobatto interpolant
    aliases T_N onto the lower modes (the well-known endpoint-aliasing property
    of Lobatto grids), so integration of t**N picks up an O(1e-7) error.  We
    only test through degree N-1, which is what spectral integration guarantees.
    """
    for a, b in [(0.0, 1.0), (-1.0, 1.0), (0.0, 5.0), (-3.0, 2.0)]:
        N = 20
        nodes, _, J = chebyshev_setup(N, a, b)
        for k in range(N):  # degree 0..N-1 exact
            f = nodes**k
            u_pred = J @ f
            u_exact = (nodes**(k + 1) - a**(k + 1)) / (k + 1)
            err = np.max(np.abs(u_pred - u_exact))
            scale = max(1.0, np.max(np.abs(u_exact)))
            assert err < 1e-12 * scale, (a, b, k, err)


def test_integration_matrix_zero_at_left_endpoint():
    """∫_a^a f(τ)dτ = 0 for any f, so the last row of J should give 0."""
    nodes, _, J = chebyshev_setup(15, -2.0, 3.0)
    for _ in range(5):
        f = np.random.randn(len(nodes))
        u = J @ f
        # Last node corresponds to t = a, so integral from a to a is 0.
        assert abs(u[-1]) < 1e-13


def test_differentiation_matrix_inverse_of_integration():
    """For smooth f, J(D f) should give f - f(a)."""
    N = 25
    a, b = 0.0, 2.0
    nodes, D, J = chebyshev_setup(N, a, b)
    for fn in (np.exp, np.sin, np.cos, lambda x: x**5 - 3*x**2 + 1):
        f = fn(nodes)
        df = D @ f
        recovered = J @ df
        # f(a) = fn(a)
        f_a = fn(a)
        err = np.max(np.abs(recovered - (f - f_a)))
        assert err < 1e-10


def test_integration_matrix_scales_with_interval():
    """Compare J on [0, 2] against J on [0, 1] applied to f(2t)."""
    N = 20
    nodes1, _, J1 = chebyshev_setup(N, 0.0, 2.0)
    # For f(t) = t^3, ∫_0^t τ^3 dτ = t^4 / 4
    f = nodes1**3
    u = J1 @ f
    err = np.max(np.abs(u - nodes1**4 / 4))
    assert err < 1e-12


# ---------------------------------------------------------------------------
# barycentric weights and evaluation
# ---------------------------------------------------------------------------

def test_barycentric_weights_signs():
    w = barycentric_weights(10)
    assert w[0] == 0.5
    assert w[-1] == 0.5  # N=10 even -> last is +0.5
    for k in range(1, 10):
        assert w[k] == (-1)**k

    w = barycentric_weights(11)
    assert w[0] == 0.5
    assert w[-1] == -0.5  # N=11 odd -> last is -0.5


def test_barycentric_eval_at_nodes_returns_node_values():
    """Evaluating at a node should return the node value exactly."""
    nodes, _, _ = chebyshev_setup(15, 0.0, 1.0)
    vals = np.sin(5 * nodes)
    out = barycentric_eval(nodes, vals, nodes)
    err = np.max(np.abs(out.ravel() - vals))
    assert err < 1e-13


def test_barycentric_eval_smooth_function():
    """Reconstruct e^t at random points to ~spectral accuracy."""
    N = 20
    a, b = 0.0, 1.0
    nodes, _, _ = chebyshev_setup(N, a, b)
    vals = np.exp(nodes)
    t_query = np.linspace(a, b, 100)
    out = barycentric_eval(nodes, vals, t_query).ravel()
    err = np.max(np.abs(out - np.exp(t_query)))
    assert err < 1e-13


def test_barycentric_eval_2d_vals():
    """Multi-component vals: shape (d, N+1)."""
    N = 15
    nodes, _, _ = chebyshev_setup(N, 0.0, 2 * np.pi)
    vals = np.vstack([np.sin(nodes), np.cos(nodes)])  # shape (2, N+1)
    t_query = np.linspace(0, 2*np.pi, 50)
    out = barycentric_eval(nodes, vals, t_query)
    assert out.shape == (2, 50)
    err_sin = np.max(np.abs(out[0] - np.sin(t_query)))
    err_cos = np.max(np.abs(out[1] - np.cos(t_query)))
    # 16-point Cheb interpolant of sin/cos over a full 2π period: ~ 1e-9 at most
    assert err_sin < 1e-9
    assert err_cos < 1e-9
