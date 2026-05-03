# mcpi

**M**odified **C**hebyshev–**P**icard **I**teration for solving ordinary differential equations in Python.

A clean, [numba](https://numba.pydata.org)-accelerated Python implementation of the Cheb–Picard iterative ODE solver, with adaptive multistage segmentation. Often 100×–1000× faster than scipy at machine-precision tolerances on smooth nonlinear IVPs.

```python
import numpy as np
from numba import njit
import mcpi

@njit
def lorenz(t, u, out):
    out[0] = 10 * (u[1] - u[0])
    out[1] = u[0] * (28 - u[2]) - u[1]
    out[2] = u[0] * u[1] - (8/3) * u[2]

sol = mcpi.solve(lorenz, t_span=(0, 5), y0=[1.0, 1.0, 1.0], tol=1e-13)
print(sol(2.5))                # state at t = 2.5
print(sol(np.linspace(0, 5)))  # batched evaluation
```

## What this is and isn't

**This is** a fast, well-tested Python implementation of an existing, well-published algorithm: Modified Chebyshev–Picard Iteration. The core ideas are not new. The implementation is.

**This is not** a new ODE solver method. The algorithm goes back to Bai & Junkins (2010-2011) and has been refined by their group and others over fifteen years. We're standing on a lot of shoulders.

We wrote `mcpi` because we wanted a clean, fast, scipy-compatible Python package for a method that has been mostly published in MATLAB and aerospace research code, and because the benchmark suite is genuinely useful for comparing solvers in the non-stiff regime.

## Algorithm in one paragraph

For an IVP $u' = f(t, u)$, $u(t_0) = u_0$, write the equivalent integral form $u(t) = u_0 + \int_{t_0}^{t} f(\tau, u(\tau))\,d\tau$. Discretize $u$ at the Chebyshev–Lobatto collocation nodes on a segment, replace the integral by the discrete Chebyshev integration matrix $J$, and iterate

$$u^{(k+1)}_{\text{nodes}} = u_0 + J \cdot f(t_{\text{nodes}}, u^{(k)}_{\text{nodes}})$$

until the iterate stops moving. This is the Picard fixed-point iteration applied to the Banach-space integral equation, and it converges whenever the segment is short enough that $f$ is contractive. A multistage adaptive driver picks segment widths, halves on failure, grows on success — same logic as scipy's adaptive RK methods, just with a high-order spectral collocation method as the inner step.

## Installation

```bash
pip install -e .                # core only
pip install -e ".[test]"        # add pytest + scipy for the test suite
pip install -e ".[bench]"       # add scipy, matplotlib, mpmath for benchmarks
```

Core requirements: `numpy`, `numba`.

## Running tests

```bash
pytest tests/
```

The suite covers Chebyshev primitives, every benchmark problem with a
closed-form reference, cross-checks vs `scipy.integrate.solve_ivp` at
`rtol=atol=1e-13`, and conservation laws (energy, angular momentum,
Lotka–Volterra invariant) on long integrations.

## API

### `mcpi.solve(rhs, t_span, y0, tol=1e-12, n_cheb=20, max_iter=None, t_initial=None, max_segments=200000) -> Solution`

| arg | type | meaning |
|---|---|---|
| `rhs` | numba `@njit` function `(t, u, out) -> None` | RHS, writes $f(t, u)$ into `out` in-place |
| `t_span` | `(t0, t1)` | integration interval |
| `y0` | array-like | initial condition (length determines state dim) |
| `tol` | float | per-segment Picard convergence tolerance |
| `n_cheb` | int | polynomial degree per segment (n_cheb+1 collocation nodes) |
| `max_iter` | int or None | max Picard iterations per segment, default `min(n_cheb, 30)` |
| `t_initial` | float or None | initial trial step length |
| `max_segments` | int | safety cap |

Returns a `Solution` object with:

- `sol(t)`: barycentric evaluation at any time(s) in `t_span`
- `sol.t`, `sol.y`: stitched node values
- `sol.segments`: list of per-segment dicts (anchor, end, nodes, state, iters)
- `sol.success`: True iff integration reached `t1`
- `sol.nfev`, `sol.n_segments`: diagnostics

### RHS contract

Your RHS function must:
- Be decorated with `@numba.njit`
- Have signature `(t: float, u: float64[:], out: float64[:]) -> None`
- Write the RHS into `out` in place
- Avoid Python objects, dicts, lists; use numpy arrays only

This lets numba inline the RHS into the Picard kernel and produces tight machine code.

## Where it shines, where it doesn't

**Best regime.** Smooth, non-stiff or mildly-stiff IVPs at high precision (`tol = 1e-10` to `1e-14`). Chaotic systems (Lorenz). Long oscillations. Polynomial systems. Most analytic nonlinearities.

**Avoid.** Extreme stiffness (stiffness ratio $> 10^4$). Use scipy's `LSODA` / `Radau` / `BDF` for that. We have a Robertson chemistry test in the suite to demonstrate the failure mode honestly.

**Indifferent.** Discontinuous or stochastic RHS. Picard iteration assumes smoothness; the algorithm will work but won't shine.

## Benchmark results

Run on 17 standard problems with high-precision references (closed-form, mpmath@40-digit, or scipy@1e-13). At matching precision against the *most accurate* scipy method on each problem:

| Problem | MCPI (ms) | Best scipy (ms) | Speedup |
|---|---:|---:|---:|
| riccati | 0.47 | 369 (Radau) | **790×** |
| stiff_linear | 0.98 | 16.4 (RK45) | 17× |
| cubic_forced | 0.68 | 350 (Radau) | **515×** |
| sin_u | 0.64 | 343 (Radau) | **534×** |
| exp_neg_u | 0.36 | 240 (Radau) | **674×** |
| exp_u | 0.58 | 316 (Radau) | **542×** |
| rational | 0.48 | 183 (Radau) | **381×** |
| log | 0.42 | 169 (Radau) | **398×** |
| sin_cos | 0.43 | 243 (Radau) | **570×** |
| exp_sin | 0.50 | 147 (Radau) | **293×** |
| sqrt | 0.35 | 1.27 (Radau) | 3.6× |
| stiff_oscillating | 4.61 | 68 (RK45) | 15× |
| lotka_volterra | 1.37 | 1632 (Radau) | **1195×** |
| brusselator | 3.33 | 10.8 (LSODA) | 3.2× |
| van_der_pol_mu1 | 4.27 | 14.8 (LSODA) | 3.5× |
| van_der_pol_mu10 | 17.8 | 69 (DOP853) | 3.9× |
| lorenz | 3.32 | 3604 (Radau) | **1085×** |
| robertson | (solver inappropriate — extreme stiffness) | LSODA 13ms | — |

To reproduce: `cd benchmarks && python run.py`. Per-problem Pareto plots and a summary plot are written to `benchmarks/bench_out/`.

A few notes on these numbers:

- "Best scipy" is the fastest scipy method that achieves error within 2× of MCPI's best, which is usually `Radau` at `rtol=atol=1e-13` (high precision but slow). Where `LSODA` or `DOP853` can match the precision faster, those are reported instead.
- The very high speedups (200×–1000×) come because scipy's high-precision methods do many small steps; MCPI achieves spectral convergence so it needs few segments.
- The small speedups (3×–4×) come on problems where scipy already has a fast path (LSODA on smooth oscillators) or where MCPI's convergence is hampered (stiffer problems).
- We never beat the right tool for stiff problems. `LSODA` and `BDF` exist for a reason.

## Honest comparison vs scipy

In short: MCPI dominates on smooth, non-stiff IVPs at tight tolerance. scipy's LSODA still wins on quasi-stiff oscillators that benefit from automatic stiffness detection. scipy's BDF/Radau win on truly stiff problems.

If you're solving a smooth nonlinear IVP and you need high precision, try `mcpi.solve` first. If you're solving a Robertson-like chemical kinetics problem, use `scipy.integrate.solve_ivp(method='LSODA')`. If you don't know what regime your problem is in, profile both.

## Provenance and prior art

This package is an implementation of, with minor variations, the methods described in:

- **Bai, X.** (2010). *Modified Chebyshev–Picard Iteration Methods for Solution of Initial Value and Boundary Value Problems.* PhD dissertation, Texas A&M University. The thesis from which this entire line of work descends.
- **Bai, X. & Junkins, J. L.** (2011). "Modified Chebyshev–Picard Iteration Methods for Solution of Initial Value Problems." *Journal of the Astronautical Sciences* 58(4), 583–613.
- **Bai, X. & Junkins, J. L.** (2011). "Modified Chebyshev–Picard Iteration Methods for Solution of Boundary Value Problems." *Journal of the Astronautical Sciences* 58(4), 615–642.
- **Macomber, B., Probe, A. B., Woollands, R., Read, J., Junkins, J. L.** (2013). "Modified Chebyshev–Picard Iteration for Efficient Numerical Integration of Ordinary Differential Equations." Advanced Maui Optical and Space Surveillance Technologies Conference.
- **Kim, D., Junkins, J. L., Turner, J. D.** (2015). "Multisegment Scheme Applications to Modified Chebyshev–Picard Iteration Method for Highly Elliptical Orbits." *Mathematical Problems in Engineering* 290781. The multisegment idea our adaptive driver borrows.
- **Macomber, B., Probe, A. B., Woollands, R., Read, J., Junkins, J. L.** (2016). "Enhancements to Modified Chebyshev–Picard Iteration Efficiency for Perturbed Orbit Propagation." *CMES* 111(1).
- **Woollands, R. & Junkins, J. L.** (2019). "Nonlinear Differential Equation Solvers via Adaptive Picard–Chebyshev Iteration: Applications in Astrodynamics." *Journal of Guidance, Control, and Dynamics* 42(5), 1007–1022. Adds error-feedback acceleration; we have not yet implemented this and so are leaving performance on the table relative to APC.

The closely-related **Parker–Sochacki Method** (Parker & Sochacki 1996; Carothers, Parker, Sochacki, Warne 2005) and the **Adomian–Rach modified decomposition method** (Adomian, Rach, Meyers 1991, 1997) reach similar conclusions via Maclaurin series rather than Chebyshev collocation; for polynomial RHSs the three approaches are mathematically equivalent.

For the spectral methods underpinning the Cheb–Lobatto integration matrix:
- **Trefethen, L. N.** (2000). *Spectral Methods in MATLAB.* SIAM. Chapter 6 is the standard reference.

## What's missing

List of things on the roadmap that we haven't built yet:

- **APC error-feedback acceleration** (Woollands–Junkins 2019). Halves iteration count in their experiments. Should add.
- **Automatic node-count adaptivity.** Right now the user picks `n_cheb`. The APC paper has heuristics for choosing `n_cheb` per segment based on observed convergence rates.
- **Dense `t_eval` integration** in the style of scipy's `solve_ivp(t_eval=...)`. Easy to add as a thin wrapper around `sol(t)`.
- **Event detection.** Useful but non-trivial.
- **Comparison vs SUNDIALS / DifferentialEquations.jl.** scipy isn't the state of the art. We've benchmarked vs scipy because that's what most Python users will compare against.
- **Sensitivity analysis** (`u' = f(u, p)` with derivatives w.r.t. `p`). Picard iteration extends naturally; we just haven't done it.
- **Better stiff-problem detection** so the solver can refuse a problem and recommend `scipy.integrate` instead.

PRs welcome.

## License

MIT. See LICENSE.

## Citing

If `mcpi` was useful in your work, please cite the original methods (Bai-Junkins 2011 is the natural reference) and optionally this implementation.
