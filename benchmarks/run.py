"""
Run the full MCPI benchmark suite vs scipy on the canonical problem battery.

Outputs:
  - Console table per problem (config sweep, best precision per method)
  - JSON dump of all timing/error data
  - Pareto plots per problem and a summary headline plot

Usage:
    python -m benchmarks.run [--quick]
"""
import sys, time, json, argparse, os
import numpy as np
import scipy.integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

import mcpi
from problems import PROBLEMS, get_reference, _scipy_rhs


def time_run(fn, repeat=2):
    walls, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
    return out, min(walls)


def _max_err(u_pred, u_ref, d):
    if d == 1:
        return float(np.max(np.abs(u_pred - u_ref)))
    out = 0.0
    for k in range(d):
        e = float(np.max(np.abs(u_pred[k] - u_ref[k])))
        if e > out:
            out = e
    return out


def bench_one(name, dim, rhs_jit, y0, t_span, t_eval, u_ref,
              n_chebs=(16, 20, 25), tols=(1e-9, 1e-12, 1e-13, 1e-14),
              scipy_methods=('RK45', 'DOP853', 'LSODA', 'Radau', 'BDF'),
              scipy_tols=(1e-3, 1e-6, 1e-9, 1e-12, 1e-13)):
    """Run MCPI and scipy at all configurations on one problem."""
    rows = []

    # MCPI
    for n_cheb in n_chebs:
        for tol in tols:
            try:
                def run():
                    return mcpi.solve(rhs_jit, t_span, y0,
                                       tol=tol, n_cheb=n_cheb,
                                       max_iter=min(int(n_cheb), 30),
                                       t_initial=min(0.4, (t_span[1]-t_span[0])/4),
                                       max_segments=200000)
                sol, wall = time_run(run, repeat=2)
            except Exception:
                continue
            if not sol.success:
                continue
            u_pred = sol(t_eval)
            err = _max_err(u_pred, u_ref, dim)
            rows.append({
                'method': f'MCPI N={n_cheb}', 'method_class': 'MCPI',
                'tol': tol, 'n_steps': sol.n_segments, 'nfev': sol.nfev,
                'err': max(err, 1e-17), 'wall': wall,
            })

    # scipy
    rhs_py = _scipy_rhs(name)
    for method in scipy_methods:
        for tol in scipy_tols:
            try:
                def run():
                    return scipy.integrate.solve_ivp(
                        rhs_py, t_span, y0, method=method,
                        rtol=tol, atol=tol, dense_output=True,
                    )
                sol, wall = time_run(run, repeat=2)
            except Exception:
                continue
            if sol is None or not sol.success:
                continue
            u_pred = sol.sol(t_eval)
            if dim == 1:
                u_pred = u_pred[0]
            err = _max_err(u_pred, u_ref, dim)
            rows.append({
                'method': method, 'method_class': method,
                'tol': tol, 'n_steps': len(sol.t),
                'nfev': sol.nfev,
                'err': max(err, 1e-17), 'wall': wall,
            })
    return rows


def pareto_front(rows):
    pts = sorted(rows, key=lambda r: r['wall'])
    front, best_err = [], float('inf')
    for r in pts:
        if r['err'] < best_err:
            front.append(r); best_err = r['err']
    return front


def plot_problem(name, desc, rows, t_eval, u_ref, dim, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    ax = axes[0]
    if dim == 1:
        ax.plot(t_eval, u_ref, 'k-', lw=1.5)
    else:
        for k in range(dim):
            ax.plot(t_eval, u_ref[k], lw=1.2, label=f'state {k}')
        ax.legend(fontsize=8)
    ax.set_xlabel('t'); ax.set_ylabel('state')
    ax.set_title(desc, fontsize=10)
    ax.grid(alpha=0.3)

    # Pareto
    ax2 = axes[1]
    groups = defaultdict(list)
    for r in rows: groups[r['method']].append(r)
    for k in groups: groups[k].sort(key=lambda r: r['wall'])
    mcpi_keys = sorted([k for k in groups if k.startswith('MCPI')])
    sci_keys  = sorted([k for k in groups if not k.startswith('MCPI')])
    a_col = plt.cm.YlOrRd(np.linspace(0.45, 0.95, max(len(mcpi_keys), 1)))
    s_col = plt.cm.Blues(np.linspace(0.4, 0.95, max(len(sci_keys), 1)))
    for c, n in zip(a_col, mcpi_keys):
        runs = groups[n]
        ax2.loglog([r['wall']*1000 for r in runs], [r['err'] for r in runs],
                   'o-', color=c, label=n, ms=6, lw=1.2, mec='k', mew=0.4)
    for c, n in zip(s_col, sci_keys):
        runs = groups[n]
        ax2.loglog([r['wall']*1000 for r in runs], [r['err'] for r in runs],
                   's--', color=c, label=n, ms=5, lw=1.0, mec='k', mew=0.3,
                   alpha=0.85)
    front = pareto_front(rows)
    if len(front) > 1:
        ax2.loglog([r['wall']*1000 for r in front], [r['err'] for r in front],
                   '-', color='k', lw=2.5, alpha=0.4, zorder=0,
                   label='Pareto front')
    ax2.set_xlabel('Wall time (ms)'); ax2.set_ylabel('L∞ error')
    ax2.set_title(f'Pareto: {name}', fontsize=10)
    ax2.legend(fontsize=7, loc='upper right', ncol=2)
    ax2.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def headline_plot(summary, out_path):
    """Bar-chart summary across all problems."""
    n = len(summary)
    fig, ax = plt.subplots(1, 1, figsize=(11, max(6, n*0.4)))
    names = [r['problem'] for r in summary]
    speedups = np.array([r.get('speedup_at_match', np.nan) for r in summary],
                        dtype=float)
    log_sp = np.log10(np.clip(speedups, 1e-3, 1e4))
    colors = ['#208040' if s > 10 else '#a0a020' if s > 1 else '#c02020'
              for s in speedups]
    y = np.arange(n)
    ax.barh(y, log_sp, height=0.7, color=colors, edgecolor='k', lw=0.5,
            alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('log₁₀(scipy wall / MCPI wall) at matching precision')
    ax.set_title('MCPI speedup over best scipy method, per problem')
    ax.grid(axis='x', alpha=0.3)
    for i, (s, r) in enumerate(zip(speedups, summary)):
        m = r.get('best_scipy_method', '')
        label = f'{s:.1f}× ({m})' if not np.isnan(s) else 'no MCPI win'
        ax.text(log_sp[i] + (0.05 if log_sp[i] >= 0 else -0.05), i,
                label, va='center', ha='left' if log_sp[i] >= 0 else 'right',
                fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run a small subset for quick verification')
    parser.add_argument('--out', default='./bench_out',
                        help='Output directory for plots and JSON')
    parser.add_argument('--skip', nargs='*', default=['robertson'],
                        help='Problem names to skip')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    problems = PROBLEMS[:3] if args.quick else PROBLEMS
    problems = [p for p in problems if p[0] not in args.skip]

    # Warmup numba
    print("Warming up numba (first call compiles)...")
    name0, dim0, rhs0, y0, t0, _, _ = problems[0]
    mcpi.solve(rhs0, t0, y0, tol=1e-9, n_cheb=8, max_iter=5,
               t_initial=0.05, max_segments=100)
    print("done.\n")

    summary = []
    all_rows = {}

    for entry in problems:
        name, dim, rhs_jit, y0, t_span, category, regime = entry
        print("=" * 72)
        print(f"  {name}    ({category}, {regime})")
        print("=" * 72)

        # Reference
        n_grid = 401 if dim > 1 else 2001
        t_eval = np.linspace(*t_span, n_grid)
        try:
            u_ref, ref_label = get_reference(name, y0, t_span, t_eval)
        except Exception as e:
            print(f"  reference computation FAILED: {e}")
            continue
        print(f"  reference: {ref_label}, grid: {n_grid} points")

        # Bench
        rows = bench_one(name, dim, rhs_jit, y0, t_span, t_eval, u_ref)
        all_rows[name] = rows

        # Print best per method
        print(f"\n  Best precision per method:")
        by_class = {}
        for r in rows:
            k = r['method']
            if k not in by_class or r['err'] < by_class[k]['err']:
                by_class[k] = r
        for k in sorted(by_class.keys()):
            r = by_class[k]
            print(f"    {k:<20} tol={r['tol']:.0e}  steps={r['n_steps']:5d}  "
                  f"err={r['err']:.2e}  wall={r['wall']*1000:8.2f}ms")

        # Plot
        plot_problem(name, f"{name} ({regime})", rows, t_eval, u_ref, dim,
                     os.path.join(args.out, f'pareto_{name}.png'))

        # Summary entry
        mcpi_rows = [r for r in rows if r['method_class'] == 'MCPI']
        sci_rows = [r for r in rows if r['method_class'] != 'MCPI']
        if not mcpi_rows or not sci_rows:
            continue
        best_mcpi = min(mcpi_rows, key=lambda r: r['err'])
        best_sci  = min(sci_rows,  key=lambda r: r['err'])
        target = max(best_mcpi['err'], best_sci['err']) * 2
        mcpi_at = [r for r in mcpi_rows if r['err'] <= target]
        sci_at = [r for r in sci_rows if r['err'] <= target]
        mcpi_fast = min(mcpi_at, key=lambda r: r['wall']) if mcpi_at else best_mcpi
        sci_fast  = min(sci_at,  key=lambda r: r['wall']) if sci_at  else best_sci
        speedup = sci_fast['wall'] / mcpi_fast['wall']
        summary.append({
            'problem': name, 'category': category, 'regime': regime,
            'reference': ref_label,
            'mcpi_best_err': best_mcpi['err'],
            'mcpi_match_wall_ms': mcpi_fast['wall'] * 1000,
            'best_scipy_method': sci_fast['method'],
            'best_scipy_err': sci_fast['err'],
            'best_scipy_wall_ms': sci_fast['wall'] * 1000,
            'speedup_at_match': speedup,
        })
        print(f"\n  At matching precision (err <= {target:.1e}):")
        print(f"    MCPI:  {mcpi_fast['wall']*1000:7.2f}ms  err={mcpi_fast['err']:.2e}")
        print(f"    {sci_fast['method']:<6}: {sci_fast['wall']*1000:7.2f}ms  err={sci_fast['err']:.2e}")
        print(f"    -> MCPI is {speedup:.1f}× faster" if speedup > 1
              else f"    -> {sci_fast['method']} is {1/speedup:.1f}× faster")

    # Headline plot
    if summary:
        headline_plot(summary, os.path.join(args.out, 'headline.png'))
        print(f"\nHeadline plot: {os.path.join(args.out, 'headline.png')}")

    # JSON dump
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump({'summary': summary, 'all_rows': all_rows}, f, indent=2,
                   default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"JSON: {os.path.join(args.out, 'summary.json')}")

    # Final table
    print("\n" + "=" * 72)
    print("MCPI vs scipy: speedup at matching precision")
    print("=" * 72)
    print(f"  {'problem':<22}{'MCPI ms':>10}{'scipy ms':>10}{'speedup':>10}{'best scipy':>12}")
    for r in summary:
        print(f"  {r['problem']:<22}{r['mcpi_match_wall_ms']:>10.2f}"
              f"{r['best_scipy_wall_ms']:>10.2f}{r['speedup_at_match']:>9.1f}×"
              f"  {r['best_scipy_method']:>10}")


if __name__ == '__main__':
    main()
