#!/usr/bin/env python3
"""
compute_statistics.py
=====================
Computes, from the archived per-seed registries, every statistic the manuscript
quotes for Table 3 / Section 4.2. Nothing here is hardcoded: each number printed
is derived from the CSV rows.

Place in experiments/ next to severity_sweep_per_seed.csv and per_seed_results.csv,
or pass --dir.

Computed:
  1. Table 3 rows (seed means, SD, margin), unrounded and rounded to 3 dp.
  2. OLS slope of mAP@0.5 vs mean target attenuation, per model (pooled).
  3. ANCOVA interaction  y ~ 1 + x + group + x:group  (F-test on the x:group term).
  4. Seed-level slope test: fit one slope per seed per model, paired t(4).
  5. Differential degradation-induced drop (proposed minus baseline), paired t(4),
     and a percentile bootstrap CI over seeds.

Caveats (printed with the results):
  * The pooled ANCOVA treats the 5 severity levels x 5 seeds as 25 independent
    points per model. They are repeated measures (same seed across levels), so the
    pooled p-value is optimistic. The seed-level slope test (item 4) respects the
    pairing and is the more defensible number to report.
  * The bootstrap here resamples the 5 seeds. It is NOT the "cluster bootstrap over
    187 test volumes" described in the paper; that needs per-volume predictions,
    which the registry does not contain.
"""

import os
import csv
import argparse
from collections import OrderedDict

import numpy as np
from scipy import stats

ATTEN = OrderedDict([  # mean target attenuation per severity level (paper Table 3)
    ("Clean (unattenuated)", 0.000),
    ("Mild attenuation", 0.042),
    ("Moderate attenuation", 0.089),
    ("Strong attenuation", 0.135),
    ("Severe stress", 0.184),
])


def load_sweep(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "level": r["severity_level"],
                "bmin": float(r["b_min"]),
                "seed": int(r["seed"]),
                "base": float(r["baseline_map50"]),
                "prop": float(r["proposed_map50"]),
            })
    return rows


def load_master(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["model"], {})[int(r["seed"])] = r
    return out


def table3(rows):
    print("=" * 96)
    print("TABLE 3 (computed from severity_sweep_per_seed.csv; mean over seeds)")
    print("=" * 96)
    print(f"{'Level':<24}{'Bmin':>6}{'Atten':>8}{'Baseline':>11}{'Proposed':>11}"
          f"{'Margin':>10}{'  3dp (B/P/M)':>22}")
    for lvl, att in ATTEN.items():
        sel = [r for r in rows if r["level"] == lvl]
        b = np.array([r["base"] for r in sel]); p = np.array([r["prop"] for r in sel])
        bm, pm = b.mean(), p.mean()
        print(f"{lvl:<24}{sel[0]['bmin']:>6.2f}{att:>8.3f}{bm:>11.4f}{pm:>11.4f}"
              f"{pm - bm:>+10.4f}   {bm:.3f}/{pm:.3f}/{pm - bm:+.3f}")
    print()


def design(x, g):
    # columns: intercept, x, group, x*group
    return np.column_stack([np.ones_like(x), x, g, x * g])


def ols_rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float(resid @ resid)


def pooled_slopes_and_ancova(rows):
    x = np.array([ATTEN[r["level"]] for r in rows] * 2)
    y = np.array([r["base"] for r in rows] + [r["prop"] for r in rows])
    g = np.array([0.0] * len(rows) + [1.0] * len(rows))  # 0=baseline, 1=proposed

    n = len(y)
    X_full = design(x, g)
    beta, rss_full = ols_rss(X_full, y)
    X_red = X_full[:, :3]                      # drop interaction term
    _, rss_red = ols_rss(X_red, y)
    df_num, df_den = 1, n - X_full.shape[1]
    F = ((rss_red - rss_full) / df_num) / (rss_full / df_den)
    p = float(stats.f.sf(F, df_num, df_den))

    slope_b = beta[1]
    slope_p = beta[1] + beta[3]

    print("=" * 96)
    print("POOLED OLS / ANCOVA (y = mAP@0.5, x = mean target attenuation)")
    print("=" * 96)
    print(f"OLS slope baseline : {slope_b:+.4f}  ({slope_b:+.3f})")
    print(f"OLS slope proposed : {slope_p:+.4f}  ({slope_p:+.3f})")
    print(f"ANCOVA interaction : F({df_num},{df_den}) = {F:.3f}, p = {p:.4f}")
    print("NOTE: pooled test treats seed x level points as independent -> optimistic.")
    print()
    return slope_b, slope_p, F, p


def seed_level_slope_test(rows):
    seeds = sorted({r["seed"] for r in rows})
    sb, sp = [], []
    for s in seeds:
        sel = [r for r in rows if r["seed"] == s]
        x = np.array([ATTEN[r["level"]] for r in sel])
        sb.append(np.polyfit(x, [r["base"] for r in sel], 1)[0])
        sp.append(np.polyfit(x, [r["prop"] for r in sel], 1)[0])
    sb, sp = np.array(sb), np.array(sp)
    diff = sp - sb                              # positive = proposed degrades less
    t, p = stats.ttest_rel(sp, sb)
    n = len(diff)
    print("=" * 96)
    print("SEED-LEVEL SLOPE TEST (one slope per seed per model; paired)")
    print("=" * 96)
    for s, a, b in zip(seeds, sb, sp):
        print(f"  seed {s}: baseline slope {a:+.4f} | proposed slope {b:+.4f} | diff {b - a:+.4f}")
    print(f"mean slopes: baseline {sb.mean():+.4f}, proposed {sp.mean():+.4f}")
    print(f"mean diff (proposed - baseline) = {diff.mean():+.4f}, "
          f"paired t({n - 1}) = {t:.3f}, p = {p:.4f}")
    print()


def bootstrap_ci(vals, n_boot=10000, seed=42, ci=0.95):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a)))


def differential_drop(rows, master):
    seeds = sorted({r["seed"] for r in rows})

    def drop_from_sweep(key):
        out = []
        for s in seeds:
            clean = [r[key] for r in rows if r["seed"] == s and r["level"] == "Clean (unattenuated)"][0]
            sev = [r[key] for r in rows if r["seed"] == s and r["level"] == "Severe stress"][0]
            out.append(sev - clean)
        return np.array(out)

    db, dp = drop_from_sweep("base"), drop_from_sweep("prop")
    red = dp - db                              # positive = proposed loses less
    t, p = stats.ttest_rel(dp, db)
    lo, hi = bootstrap_ci(red)
    t_lo, t_hi = stats.t.interval(0.95, df=len(red) - 1, loc=np.mean(red), scale=stats.sem(red))

    print("=" * 96)
    print("DIFFERENTIAL DEGRADATION-INDUCED DROP (severe-stress minus clean, per seed)")
    print("=" * 96)
    print(f"baseline drop mean : {db.mean():+.4f}")
    print(f"proposed drop mean : {dp.mean():+.4f}")
    print(f"differential reduction (proposed - baseline) = {red.mean():+.4f}")
    print(f"paired t({len(red) - 1}) = {t:.3f}, p = {p:.4f}")
    print(f"seed-level Student's t 95% CI: [{t_lo:+.4f}, {t_hi:+.4f}] -> [{t_lo:+.3f}, {t_hi:+.3f}] (parametric t(4) interval)")
    print(f"seed-level bootstrap 95% CI:   [{lo:+.4f}, {hi:+.4f}] -> [{lo:+.3f}, {hi:+.3f}] (resamples 5 seeds; percentile)")
    print(f"NOTE: With N={len(red)} seeds, the parametric Student's t({len(red) - 1}) interval accounts for degrees of freedom")
    print("      and is the defensible parametric interval reported in the manuscript.")

    # Cross-check against delta_drop in the master registry
    try:
        mb = np.array([float(master["Baseline YOLOv8n"][s]["delta_drop"]) for s in seeds])
        mp = np.array([float(master["Proposed Model"][s]["delta_drop"]) for s in seeds])
        t2, p2 = stats.ttest_rel(mp, mb)
        print(f"[master registry delta_drop] baseline {mb.mean():+.4f}, proposed {mp.mean():+.4f}, "
              f"diff {np.mean(mp - mb):+.4f}, paired t({len(mb) - 1}) = {t2:.3f}, p = {p2:.4f}")
    except KeyError as e:
        print(f"[master registry cross-check skipped: missing {e}]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    rows = load_sweep(os.path.join(args.dir, "severity_sweep_per_seed.csv"))
    master = load_master(os.path.join(args.dir, "per_seed_results.csv"))

    table3(rows)
    pooled_slopes_and_ancova(rows)
    seed_level_slope_test(rows)
    differential_drop(rows, master)


if __name__ == "__main__":
    main()
