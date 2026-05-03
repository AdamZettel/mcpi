"""
Generate the polished plots used in the project README.

Outputs (all PNG, 200 DPI, transparent-friendly backgrounds):
    assets/headline.png      -- speedup bar chart, MCPI vs best scipy method
    assets/lorenz.png        -- time series + 3D attractor with color-graded time
    assets/convergence.png   -- spectral convergence on a smooth IVP

Run:
    python scripts/make_readme_plots.py

These plots are decoupled from the live benchmark output (benchmarks/run.py),
which is intended for developer inspection.  This script renders publication-
quality figures using authoritative speedup data from the benchmark suite.
"""
import os
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers 3D projection
from matplotlib.colors import LinearSegmentedColormap, Normalize

import mcpi


# ---------------------------------------------------------------------------
# Global plot style — clean, modern, sans-serif
# ---------------------------------------------------------------------------
rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#3d3d3d',
    'axes.labelcolor': '#222222',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'grid.color': '#dddddd',
    'grid.alpha': 0.6,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 9,
})


HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, '..', 'assets'))
os.makedirs(ASSETS, exist_ok=True)


# ===========================================================================
# 1. Headline: speedup bar chart vs best scipy method
# ===========================================================================
# Numbers come from the canonical benchmark run (see README table and
# benchmarks/run.py).  Hardcoded here so the plot is reproducible without
# rerunning the suite, and so README and chart stay consistent.
HEADLINE_DATA = [
    # (problem,         mcpi_ms,  scipy_ms,  scipy_method)
    ('riccati',           0.47,    369.0,    'Radau'),
    ('stiff_linear',      0.98,     16.4,    'RK45'),
    ('cubic_forced',      0.68,    350.0,    'Radau'),
    ('sin_u',             0.64,    343.0,    'Radau'),
    ('exp_neg_u',         0.36,    240.0,    'Radau'),
    ('exp_u',             0.58,    316.0,    'Radau'),
    ('rational',          0.48,    183.0,    'Radau'),
    ('log',               0.42,    169.0,    'Radau'),
    ('sin_cos',           0.43,    243.0,    'Radau'),
    ('exp_sin',           0.50,    147.0,    'Radau'),
    ('sqrt',              0.35,      1.27,   'Radau'),
    ('stiff_oscillating', 4.61,     68.0,    'RK45'),
    ('lotka_volterra',    1.37,   1632.0,    'Radau'),
    ('brusselator',       3.33,     10.8,    'LSODA'),
    ('van_der_pol_mu1',   4.27,     14.8,    'LSODA'),
    ('van_der_pol_mu10', 17.8,      69.0,    'DOP853'),
    ('lorenz',            3.32,   3604.0,    'Radau'),
]


def plot_headline(out_path):
    fig, ax = plt.subplots(figsize=(11.5, 8.0))

    # Sort ascending so the most dramatic speedup ends up at top
    data = sorted(HEADLINE_DATA, key=lambda r: r[2] / r[1])
    names = [r[0] for r in data]
    speedups = np.array([r[2] / r[1] for r in data])
    methods = [r[3] for r in data]

    log_sp = np.log10(speedups)
    y = np.arange(len(data))

    # Diverging-style coloring: subtle yellow at low speedup, deep green at high
    cmap = LinearSegmentedColormap.from_list(
        'mcpi_green',
        [(0.0, '#d4a017'),     # mustard at low end
         (0.3, '#a4b020'),     # olive transition
         (0.55, '#6aa84f'),    # mid green
         (1.0, '#1a6e3a')],    # deep forest at top
    )
    norm = Normalize(vmin=0.0, vmax=max(log_sp.max(), 3.2))
    colors = cmap(norm(log_sp))

    bars = ax.barh(y, log_sp, height=0.72, color=colors, edgecolor='#222222',
                   linewidth=0.6)

    # Reference lines: 1× (parity), 10×, 100×, 1000×
    for ref, lbl in [(0.0, '1×'), (1.0, '10×'), (2.0, '100×'), (3.0, '1000×')]:
        ax.axvline(ref, color='#888888', lw=0.6, ls='--', alpha=0.6, zorder=0)
        ax.text(ref, len(data) - 0.3, lbl, ha='center', va='bottom',
                fontsize=8.5, color='#666666')

    # Annotate each bar
    for i, (bar, s, m) in enumerate(zip(bars, speedups, methods)):
        w = bar.get_width()
        if s >= 100:
            label = f'  {s:.0f}× vs {m}'
        elif s >= 10:
            label = f'  {s:.1f}× vs {m}'
        else:
            label = f'  {s:.1f}× vs {m}'
        ax.text(w, i, label, va='center', ha='left',
                fontsize=9.5, color='#222222', weight='medium')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(r'$\log_{10}$(scipy wall time / MCPI wall time)',
                  fontsize=11, labelpad=8)
    ax.set_title('MCPI vs scipy:  3 × – 1200 × faster at machine-precision tolerances',
                 fontsize=13, pad=14, color='#1a3a5e')
    ax.set_xlim(-0.2, max(log_sp.max() + 1.0, 4.0))
    ax.grid(axis='x', alpha=0.35, zorder=0)

    # Subtle subtitle
    fig.text(0.5, 0.015,
             'Compared against the fastest scipy method that matches MCPI precision (within 2×). '
             '17 standard non-stiff IVPs.',
             ha='center', fontsize=9, color='#666666', style='italic')

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out_path)
    plt.close()
    print(f'  wrote {out_path}')


# ===========================================================================
# 2. Lorenz attractor with color-graded time
# ===========================================================================
@njit
def _lorenz_rhs(t, u, out):
    out[0] = 10.0 * (u[1] - u[0])
    out[1] = u[0] * (28.0 - u[2]) - u[1]
    out[2] = u[0] * u[1] - (8.0 / 3.0) * u[2]


def plot_lorenz(out_path):
    sol = mcpi.solve(_lorenz_rhs, t_span=(0, 30), y0=[1.0, 1.0, 1.0],
                     tol=1e-13, n_cheb=25)
    t = np.linspace(0, 30, 8001)
    y = sol(t)

    fig = plt.figure(figsize=(14.0, 6.2), facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.18)

    # ---- Time series ----
    ax1 = fig.add_subplot(gs[0, 0])
    palette = ['#1f6feb', '#d4571c', '#1a6e3a']
    labels = ['x', 'y', 'z']
    for k, (c, lbl) in enumerate(zip(palette, labels)):
        ax1.plot(t, y[k], color=c, lw=0.75, alpha=0.92, label=lbl)
    ax1.set_xlabel('t', fontsize=11)
    ax1.set_ylabel('state', fontsize=11)
    ax1.set_title('Time series  (tol = 1e-13, 30 time units)',
                  fontsize=12, pad=10, color='#1a3a5e')
    ax1.legend(loc='upper right', fontsize=10, ncol=3)
    ax1.grid(alpha=0.3)

    # Annotate solver stats
    stats = (f'segments  : {sol.n_segments}\n'
             f'RHS calls : {sol.nfev:,}\n'
             f'success   : {sol.success}')
    ax1.text(0.015, 0.97, stats, transform=ax1.transAxes,
             va='top', ha='left', fontsize=9, family='monospace',
             color='#333333',
             bbox=dict(boxstyle='round,pad=0.5', fc='#fafbfc', ec='#bbbbbb',
                       lw=0.6, alpha=0.95))

    # ---- 3D phase portrait, time-coloured ----
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    # Use Line3DCollection for crisp, GPU-friendly colouring
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    pts = y.T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=t.min(), vmax=t.max())
    lc = Line3DCollection(segs, cmap=cmap, norm=norm,
                           linewidths=0.55, alpha=0.9)
    lc.set_array(t[:-1])
    ax2.add_collection3d(lc)

    # Mark start and end
    ax2.scatter([y[0, 0]], [y[1, 0]], [y[2, 0]],
                color='#1f6feb', s=55, edgecolor='black', linewidth=0.7,
                label=f't = 0', zorder=10)
    ax2.scatter([y[0, -1]], [y[1, -1]], [y[2, -1]],
                color='#d4571c', s=55, edgecolor='black', linewidth=0.7,
                label=f't = {t[-1]:.0f}', zorder=10)

    ax2.set_xlim(y[0].min() - 1, y[0].max() + 1)
    ax2.set_ylim(y[1].min() - 1, y[1].max() + 1)
    ax2.set_zlim(y[2].min() - 1, y[2].max() + 1)

    ax2.set_xlabel('x', labelpad=2)
    ax2.set_ylabel('y', labelpad=2)
    ax2.set_zlabel('z', labelpad=2)
    ax2.set_title('Phase portrait  (color = time)',
                  fontsize=12, pad=10, color='#1a3a5e')
    ax2.view_init(elev=20, azim=-58)
    # Tone down 3D axis panes
    for pane in (ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane):
        pane.set_facecolor('#fbfbfd')
        pane.set_edgecolor('#dddddd')
    ax2.grid(alpha=0.18)
    ax2.legend(loc='upper left', fontsize=9, frameon=False)

    # Discrete colorbar showing the time gradient
    cbar = fig.colorbar(lc, ax=ax2, shrink=0.7, pad=0.08,
                         ticks=[0, 7.5, 15, 22.5, 30])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor('#888888')
    cbar.ax.tick_params(labelsize=8.5, color='#666666')
    cbar.set_label('t', rotation=0, labelpad=10, fontsize=10, color='#444444')

    plt.savefig(out_path)
    plt.close()
    print(f'  wrote {out_path}')


# ===========================================================================
# 3. Spectral convergence: error vs n_cheb on a smooth problem
# ===========================================================================
@njit
def _decay_rhs(t, u, out):
    out[0] = -u[0]


@njit
def _harmonic_rhs(t, u, out):
    out[0] = u[1]
    out[1] = -u[0]


@njit
def _exp_neg_u_rhs(t, u, out):
    out[0] = np.exp(-u[0])


def plot_convergence(out_path):
    """Error vs. n_cheb at fixed segment length — shows spectral convergence.

    For a smooth IVP, MCPI's per-segment error decays as O(rho^N) for some
    rho < 1 (the Bernstein ellipse parameter), which is exponential / "spectral"
    convergence.  We pick three problems and plot ||sol(t_eval) - exact|| vs N.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.4), facecolor='white')

    Ns = list(range(4, 33, 2))
    problems = [
        ('linear decay',  _decay_rhs,        [1.0],         (0, 1.0),
         lambda t: np.exp(-t),                    '#1f6feb'),
        ('harmonic',      _harmonic_rhs,     [1.0, 0.0],    (0, 1.0),
         lambda t: np.cos(t),                     '#d4571c'),
        ('exp(-u)',       _exp_neg_u_rhs,    [0.0],         (0, 1.0),
         lambda t: np.log(1 + t),                 '#1a6e3a'),
    ]

    t_eval = np.linspace(0, 1.0, 51)
    eps = np.finfo(float).eps

    for label, rhs, y0, t_span, exact_fn, color in problems:
        errors = []
        for N in Ns:
            # Use t_initial = full span so we always solve in one segment
            try:
                sol = mcpi.solve(rhs, t_span, y0,
                                 tol=1e-15, n_cheb=N,
                                 max_iter=80,
                                 t_initial=t_span[1] - t_span[0])
                u_pred = sol(t_eval)
                if u_pred.ndim > 1:
                    u_pred = u_pred[0]
                err = np.max(np.abs(u_pred - exact_fn(t_eval)))
            except Exception:
                err = np.nan
            errors.append(max(err, eps / 2))
        ax.semilogy(Ns, errors, '-o', color=color, label=label,
                    lw=1.6, ms=6, mec='black', mew=0.5)

    # Mark machine epsilon
    ax.axhline(eps, color='#888888', lw=0.8, ls='--', alpha=0.7)
    ax.text(Ns[0] + 0.5, eps * 1.7, r'machine $\epsilon$',
            ha='left', va='bottom', fontsize=9.5, color='#666666',
            style='italic')

    ax.set_xlabel('Polynomial degree per segment ($n_{cheb}$)', fontsize=11)
    ax.set_ylabel(r'$L^\infty$ error  on  $[0, 1]$', fontsize=11)
    ax.set_title('Spectral convergence: error decays exponentially with $n_{cheb}$',
                 fontsize=12, pad=10, color='#1a3a5e')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xticks(Ns[::2])
    # Make sure the y-axis floor includes machine epsilon and a little below
    ax.set_ylim(bottom=eps * 0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  wrote {out_path}')


# ===========================================================================
def main():
    print('Generating README plots...')
    plot_headline(os.path.join(ASSETS, 'headline.png'))
    plot_lorenz(os.path.join(ASSETS, 'lorenz.png'))
    plot_convergence(os.path.join(ASSETS, 'convergence.png'))
    print('done.')


if __name__ == '__main__':
    main()
