"""
Quick-start: solve  u' = sin(u), u(0) = 0.5  on  [0, 10]
and compare against the closed form  u(t) = 2 arctan(tan(u0/2) e^t).
"""
import numpy as np
from numba import njit
import mcpi


@njit
def f(t, u, out):
    out[0] = np.sin(u[0])


def main():
    u0 = 0.5
    T = 10.0
    sol = mcpi.solve(f, t_span=(0, T), y0=[u0], tol=1e-13)

    print(f"Success:        {sol.success}")
    print(f"Segments:       {sol.n_segments}")
    print(f"RHS calls:      {sol.nfev}")

    t = np.linspace(0, T, 11)
    u_pred = sol(t)
    u_true = 2 * np.arctan(np.tan(u0/2) * np.exp(t))

    print(f"\n  {'t':>5}  {'sol(t)':>22}  {'exact':>22}  {'|err|':>10}")
    for ti, p, x in zip(t, u_pred, u_true):
        print(f"  {ti:5.1f}  {p:22.16f}  {x:22.16f}  {abs(p-x):10.2e}")


if __name__ == '__main__':
    main()
