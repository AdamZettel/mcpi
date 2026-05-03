"""
Lorenz attractor — classic chaotic ODE benchmark.

This example demonstrates:
  - Defining a numba @njit RHS
  - Integrating to high precision
  - Using sol(t) for arbitrary-time evaluation
  - The 3D phase portrait
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import mcpi


@njit
def lorenz(t, u, out):
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    out[0] = sigma * (u[1] - u[0])
    out[1] = u[0] * (rho - u[2]) - u[1]
    out[2] = u[0] * u[1] - beta * u[2]


def main():
    sol = mcpi.solve(
        lorenz,
        t_span=(0, 25),
        y0=[1.0, 1.0, 1.0],
        tol=1e-13,
        n_cheb=25,
    )
    print(f"Integration success: {sol.success}")
    print(f"Segments used:       {sol.n_segments}")
    print(f"RHS evaluations:     {sol.nfev}")
    print(f"State at t=25:       {sol(25.0)}")

    # Dense evaluation for plotting
    t = np.linspace(0, 25, 5001)
    y = sol(t)

    fig = plt.figure(figsize=(11, 5))
    # Time series
    ax1 = fig.add_subplot(1, 2, 1)
    for k, label in enumerate(('x', 'y', 'z')):
        ax1.plot(t, y[k], label=label, lw=0.8)
    ax1.set_xlabel('t')
    ax1.set_ylabel('state')
    ax1.set_title('Lorenz time series')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 3D phase portrait
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(y[0], y[1], y[2], lw=0.5, color='C3')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
    ax2.set_title('Lorenz attractor')

    plt.tight_layout()
    plt.savefig('lorenz_example.png', dpi=120, bbox_inches='tight')
    print("Saved: lorenz_example.png")


if __name__ == '__main__':
    main()
