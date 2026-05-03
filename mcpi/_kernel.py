"""
The MCPI inner kernel: one segment of Cheb-Picard iteration.

Given the integration matrix J, the Cheb-Lobatto nodes, an anchor time and
state, and a JIT'd RHS function, iteratively refine the state at all nodes
via

    u_{k+1}[node] = ic + sum_j  J[node, j] * f(t_anchor + nodes[j], u_k[j])

until ||u_{k+1} - u_k||_inf falls below tol_conv or max_iter is reached.

The kernel is created by `make_kernel(d)(rhs)`, which closes over the
state dimension d and the user's @njit'd RHS.  Closing over the RHS lets
numba inline it into the kernel so the entire inner loop compiles down
to tight numerical code with no Python overhead.
"""
from numba import njit
import numpy as np


def make_kernel(d):
    """Create a JIT'd MCPI segment kernel for a d-dimensional state.

    Parameters
    ----------
    d : int
        Number of state components.

    Returns
    -------
    function
        A factory that takes a JIT'd RHS function and returns the kernel.

    The RHS must have signature  rhs(t: float, u: float64[:], out: float64[:])
    and write f(t, u) into the preallocated `out` buffer.  All three arguments
    are 1-D contiguous numpy arrays of length d (except t, which is scalar).
    """
    def construct(rhs_jit):
        @njit(cache=False)
        def kernel(J, nodes, t_anchor, ic, max_iter, tol_conv):
            n_nodes = J.shape[0]

            # Initial guess: constant = ic at every collocation node.
            u = np.zeros((n_nodes, d))
            for j in range(n_nodes):
                for i in range(d):
                    u[j, i] = ic[i]

            f_vals = np.zeros((n_nodes, d))
            f_buf = np.zeros(d)
            u_new = np.zeros((n_nodes, d))

            # Reference scale for divergence detection.
            init_norm = 1e-30
            for i in range(d):
                if abs(ic[i]) > init_norm:
                    init_norm = abs(ic[i])

            for n in range(max_iter):
                # Evaluate RHS at every node with the current iterate.
                for j in range(n_nodes):
                    rhs_jit(t_anchor + nodes[j], u[j], f_buf)
                    for i in range(d):
                        f_vals[j, i] = f_buf[i]

                # u_new = ic + J @ f_vals  (componentwise per state).
                max_diff = 0.0
                for j in range(n_nodes):
                    for i in range(d):
                        acc = ic[i]
                        for k in range(n_nodes):
                            acc += J[j, k] * f_vals[k, i]
                        u_new[j, i] = acc
                        diff = abs(acc - u[j, i])
                        if diff > max_diff:
                            max_diff = diff

                # Copy u_new into u for next iteration.
                for j in range(n_nodes):
                    for i in range(d):
                        u[j, i] = u_new[j, i]

                if max_diff < tol_conv:
                    return u, True, n + 1
                # Divergence guard: if the iterate blows up, give up early
                # so the multistage driver can shrink the step.
                if max_diff > 1e10 * (init_norm + 1.0):
                    return u, False, n + 1
            return u, False, max_iter
        return kernel
    return construct
