#!/usr/bin/env python3
"""
run_analysis.py
=====================================================================
End-to-end analysis runner for the SRL / shared-control soldering study.

Reads a recordings/ folder (as written by session_runner.py + trial_logger.py),
extracts every per-trial metric, aggregates to participant x condition, writes
tidy CSVs, and renders descriptive figures (box + individual points + 95% CI).
No significance testing — this is the extraction + descriptives + plots layer.

STUDY STRUCTURE (matches session_runner.py)
    trial   = one Williams pass over the 8 faces = 8 button-pair presses.
    block   = 9 trials = a full 3x3 Latin square over the 3 conditions (each of
              C0/C1/C2 appears 3x per block; condition changes every trial).
    session = 3 blocks -> 27 trials/participant, 9 per condition. Each participant
              x condition aggregate is the mean over that condition's 9 trials.

Examples
    # analyse real data
    python3 run_analysis.py --recordings recordings --out analysis_out

    # generate synthetic data first, then analyse it (self-test / demo)
    python3 make_synthetic_data.py --out synthetic_recordings --n 9
    python3 run_analysis.py --recordings synthetic_recordings --out analysis_out

    # drop the first 2 warm-up trials of each condition
    python3 run_analysis.py --recordings recordings --drop-warmup 2

Outputs (under --out)
    per_trial_metrics.csv           one row per trial, every metric
    per_participant_condition.csv   mean over the condition's trials, + questionnaire scores
    condition_descriptives.csv      n, mean, sd, median, IQR, bootstrap 95% CI
    figures/*.png

Note: error-rate (wrong-pair / single-button presses) is near-zero in practice, so it
is kept in per_trial_metrics.csv as a data-quality check but is NOT featured as an
outcome metric in the figures or the console summary.
"""

from __future__ import annotations
import os
import re
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- publication-quality figure defaults (applied to every figure) ---------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "xtick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "lines.markersize": 6,
})

import srl_analysis as A
from srl_common import CONDITIONS

COND_COLOUR = {"C0": "#888888", "C1": "#4C78A8", "C2": "#E45756"}

# --- HYPOTHESIS-ALIGNED TESTING -------------------------------------------------------------
# The hypotheses are about the robot-active condition (C2) vs the two controls, so the confirmatory
# tests are PLANNED CONTRASTS: C2 vs C0 (does a robot help vs hands) and C2 vs C1 (does ACTIVE
# reorientation beat a passive holder — the key contrast). Holm-corrected over these TWO only (the
# C0-vs-C1 comparison is not hypothesised, so it is not tested / does not consume correction). The
# omnibus RM-ANOVA is kept as a descriptive gate. Contrasts are direction-aware: a metric supports
# the hypothesis only if C2 moves the "better" way AND the corrected p < .05.
TREATMENT = "C2"
CONTROLS = ["C0", "C1"]          # C2-vs-C1 is the theoretically decisive one
BETTER_DIR = {"completion_time_s": "lower", "median_rt_s": "lower", "median_sync_ms": "lower",
              "pct_sync_within100": "higher", "singleton_rate": "lower", "rtlx": "lower",
              "accuracy": "higher", "error_rate": "lower", "freeze_count": "lower"}
# Featured outcome metrics. Error-rate and first-try accuracy are intentionally
# excluded (near-zero / non-discriminating in this task) — see module docstring.
PRIMARY = [("completion_time_s", "Sequence time (s)"),
           ("median_rt_s", "Median press RT (s)"),
           ("median_sync_ms", "Bimanual offset (ms)"),
           ("rtlx", "Raw NASA-TLX (0-100)")]
C2_METRICS = [("reposition_correctness", "Reposition correct (frac)"),
              ("functional_delay_s", "Functional delay (s)"),
              ("robot_active_frac", "Concurrent/assist frac"),
              ("freeze_count", "Freezes per sequence")]


def _box_strip(ax, ppc, metric, title):
    """Box (median/IQR) + individual participant points per condition."""
    data, xs = [], []
    for c in CONDITIONS:
        v = ppc.loc[ppc["condition"] == c, metric].dropna().to_numpy(float)
        data.append(v)
    present = [(c, d) for c, d in zip(CONDITIONS, data) if d.size]
    if not present:
        ax.set_visible(False); return
    positions = range(1, len(present) + 1)
    ax.boxplot([d for _, d in present], positions=list(positions),
               widths=0.55, showmeans=True, meanline=False,
               medianprops=dict(color="black"))
    for i, (c, d) in enumerate(present, start=1):
        jit = (np.random.default_rng(0).random(d.size) - 0.5) * 0.18
        ax.scatter(np.full(d.size, i) + jit, d, s=22, alpha=0.7,
                   color=COND_COLOUR.get(c, "#333"), edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(list(positions)); ax.set_xticklabels([c for c, _ in present])
    ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.25)
    dmin = min([float(d.min()) for _, d in present if d.size], default=0.0)
    ax.set_ylim(bottom=min(0.0, dmin))       # anchor at 0 for non-negative metrics


def fig_primary(ppc, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8))
    for ax, (metric, title) in zip(axes.ravel(), PRIMARY):
        if metric in ppc.columns:
            wide = ppc.pivot_table(index="participant", columns="condition", values=metric)
            _condition_box_with_stats(ax, wide, CONDITIONS, title, COND_COLOUR, title="")
        else:
            ax.set_visible(False)
    # hide any unused axes if PRIMARY has fewer entries than the grid
    for ax in axes.ravel()[len(PRIMARY):]:
        ax.set_visible(False)
    fig.suptitle("Primary metrics by condition — RM-ANOVA + Holm pairwise (box = median/IQR, "
                 "points = participants)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(outdir, "fig_primary_metrics.png"); fig.savefig(p); plt.close(fig)
    return p


def condition_stats_table(ppc, metrics):
    """Per-metric omnibus RM-ANOVA (perm) + hypothesis-aligned PLANNED CONTRASTS: C2 vs each control
    (C0, C1), Holm-corrected over the two, direction-aware. Reports the signed mean diff (C2 minus
    control), the corrected p, and whether the hypothesis is supported (C2 better AND p<.05)."""
    rows = []
    for metric, label in metrics:
        if metric not in ppc.columns:
            continue
        wide = ppc.pivot_table(index="participant", columns="condition", values=metric)
        conds = [c for c in CONDITIONS if c in wide.columns]
        comp = wide[conds].dropna()
        F, p, df1, df2, n = _rm_anova_perm(comp.to_numpy()) if len(conds) >= 2 else (np.nan,) * 5
        row = {"metric": metric, "label": label, "n": n,
               "rm_anova_F": F, "rm_anova_df1": df1, "rm_anova_df2": df2, "rm_anova_p": p}
        for c in conds:
            row[f"mean_{c}"] = float(wide[c].mean()) if wide[c].notna().any() else np.nan
        better = BETTER_DIR.get(metric, "lower")
        controls = [c for c in CONTROLS if c in conds and TREATMENT in conds]
        praw = []
        for c in controls:
            d = wide[[c, TREATMENT]].dropna()
            praw.append(_signflip_p((d[TREATMENT] - d[c]).to_numpy())[0] if len(d) >= 2 else np.nan)
        for c, pa in zip(controls, _holm(praw)):
            d = wide[[c, TREATMENT]].dropna()
            diff = float((d[TREATMENT] - d[c]).mean()) if len(d) else np.nan   # C2 - control
            supported = bool(np.isfinite(pa) and pa < .05 and
                             ((diff < 0) if better == "lower" else (diff > 0)))
            row[f"{TREATMENT}_vs_{c}_diff"] = diff          # C2 minus control (better dir per BETTER_DIR)
            row[f"{TREATMENT}_vs_{c}_p_holm"] = pa
            row[f"{TREATMENT}_vs_{c}_supported"] = supported
        rows.append(row)
    return pd.DataFrame(rows)


def fig_participant_lines(ppc, outdir, metrics=("completion_time_s", "rtlx")):
    metrics = [m for m in metrics if m in ppc.columns]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    axes = np.atleast_1d(axes)
    xmap = {c: i + 1 for i, c in enumerate(CONDITIONS)}
    for ax, metric in zip(axes, metrics):
        for pid, sub in ppc.groupby("participant", observed=True):
            sub = sub.sort_values("condition")
            ax.plot([xmap[c] for c in sub["condition"]], sub[metric], "-o",
                    color="#999", alpha=0.6, markersize=4)
        # condition means
        means = [ppc.loc[ppc["condition"] == c, metric].mean() for c in CONDITIONS]
        ax.plot(list(xmap.values()), means, "-o", color="#E45756", linewidth=2.5,
                markersize=8, label="mean", zorder=5)
        ax.set_xticks(list(xmap.values())); ax.set_xticklabels(list(CONDITIONS))
        ax.set_title(dict(PRIMARY).get(metric, metric), fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if np.nanmin(ppc[metric].to_numpy(float)) >= 0:
            ax.set_ylim(bottom=0)               # anchor paired trajectories at 0
    fig.suptitle("Per-participant trajectories across conditions (paired)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(outdir, "fig_participant_lines.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_c2(ppc, outdir):
    metrics = [(m, t) for m, t in C2_METRICS if m in ppc.columns
               and ppc.loc[ppc["condition"] == "C2", m].notna().any()]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, (metric, title) in zip(axes, metrics):
        v = ppc.loc[ppc["condition"] == "C2", metric].dropna().to_numpy(float)
        ax.boxplot([v], widths=0.5, showmeans=True, medianprops=dict(color="black"))
        jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.15
        ax.scatter(np.ones(v.size) + jit, v, s=26, color=COND_COLOUR["C2"],
                   edgecolor="white", zorder=3)
        ax.set_xticks([1]); ax.set_xticklabels(["C2"]); ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if v.size and v.min() >= 0:
            ax.set_ylim(bottom=0)
    fig.suptitle("C2 robot / fluency metrics", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(outdir, "fig_c2_robot.png"); fig.savefig(p); plt.close(fig)
    return p


# participant folders to exclude entirely (test / practice runs, e.g. recordings/Participanttry)
EXCLUDE_PARTICIPANT_PREFIXES = ("try", "test")


def _drop_test_participants(per_trial):
    """Drop trials from test folders whose participant id starts with an excluded prefix
    (so recordings/Participanttry* is ignored while recordings/Participant<n>* is kept)."""
    if "participant" not in per_trial.columns:
        return per_trial, 0
    pn = per_trial["participant"].astype(str).str.strip().str.lower()
    keep = ~pn.apply(lambda s: any(s.startswith(p) for p in EXCLUDE_PARTICIPANT_PREFIXES))
    return per_trial[keep].copy(), int((~keep).sum())


def _drop_warmup(per_trial, k):
    """Drop the first k trials of each participant x condition (warm-up).

    Replaces the old --exclude-block1: with one block per condition (9 trials),
    there is no separate 'block 1' to drop, so warm-up is defined at the trial
    level instead. Orders by the global trial index within each condition."""
    if k <= 0:
        return per_trial, 0
    need = {"participant", "condition"}
    order_col = "trial" if "trial" in per_trial.columns else None
    if not need.issubset(per_trial.columns) or order_col is None:
        print(f"[analysis] --drop-warmup ignored: need columns "
              f"{sorted(need)} + 'trial' in per_trial.")
        return per_trial, 0
    pt = per_trial.sort_values(list(need) + [order_col]).copy()
    rank = pt.groupby(list(need), observed=True).cumcount()   # 0-based within condition
    kept = pt[rank >= k].copy()
    return kept, len(per_trial) - len(kept)


# --- across-block LEARNING / practice effect (block 1 -> 2 -> 3) ------------------------------
# Does task performance improve as the session goes on? For each metric the trend is a slope +
# correlation vs block number, on per-participant x block means (each participant contributes one
# point per block). "improves_when" says which direction is better.
LEARN_METRICS = [
    ("completion_time_s", "Completion time (s)",   "lower"),
    ("median_rt_s",       "Median press RT (s)",   "lower"),
    ("median_sync_ms",    "Bimanual offset (ms)",  "lower"),
    ("accuracy",          "First-try accuracy",    "higher"),
    ("error_rate",        "Error rate",            "lower"),
]


def _corr_p(x, y, n_perm=20000, seed=0):
    """Pearson r with a permutation two-sided p (dependency-free). Returns (r, p, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed); c = 1
    for _ in range(n_perm):
        if abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= abs(r) - 1e-12:
            c += 1
    return r, c / (n_perm + 1), len(x)


def learning_by_block(per_trial):
    """(participant x block means, per-block means, trend_df) or (None, None, None)."""
    if "block" not in per_trial.columns or per_trial["block"].dropna().nunique() < 2:
        return None, None, None
    pt = per_trial.dropna(subset=["block"]).copy()
    pt["block"] = pt["block"].astype(int)
    metrics = [m for m, _, _ in LEARN_METRICS if m in pt.columns and pt[m].notna().any()]
    if not metrics:
        return None, None, None
    pb = pt.groupby(["participant", "block"], as_index=False)[metrics].mean()
    blk = pb.groupby("block", as_index=False)[metrics].mean().sort_values("block")
    better = {m: d for m, _, d in LEARN_METRICS}
    rows = []
    for m in metrics:
        sub = pb[["block", m]].dropna()
        if sub["block"].nunique() < 2 or len(sub) < 3:
            continue
        x = sub["block"].to_numpy(float); y = sub[m].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        r, p, n = _corr_p(x, y)
        first, last = float(blk[m].iloc[0]), float(blk[m].iloc[-1])
        improved = (last < first) if better[m] == "lower" else (last > first)
        rows.append({"metric": m, "improves_when": better[m], "slope_per_block": slope,
                     "pearson_r": r, "pearson_p": p, "n": n,
                     "first_block_mean": first, "last_block_mean": last,
                     "improved_1_to_last": bool(improved)})
    return pb, blk, pd.DataFrame(rows)


def fig_learning(per_trial, figdir):
    pb, blk, trend = learning_by_block(per_trial)
    if pb is None:
        return None
    metrics = [(m, t) for m, t, _ in LEARN_METRICS if m in pb.columns and pb[m].notna().any()]
    if not metrics:
        return None
    ncol = min(3, len(metrics)); nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.0 * nrow), squeeze=False)
    axes = axes.ravel()
    blocks = sorted(pb["block"].unique())
    for ax, (m, title) in zip(axes, metrics):
        for _, s in pb.groupby("participant", observed=True):
            s = s.sort_values("block")
            ax.plot(s["block"], s[m], "-o", color="#AAAAAA", alpha=0.5, markersize=4)
        bm = blk.sort_values("block")
        ax.plot(bm["block"], bm[m], "-o", color="#E45756", linewidth=2.6, markersize=9, zorder=5)
        sub = ""
        row = trend[trend["metric"] == m] if trend is not None else None
        if row is not None and len(row):
            rr = row.iloc[0]
            tag = "improves" if rr["improved_1_to_last"] else "no gain"
            sub = (f"slope={rr['slope_per_block']:+.2f}/blk  r={rr['pearson_r']:+.2f} "
                   f"(p={rr['pearson_p']:.3f})  [{tag}]")
        ax.set_xticks(blocks); ax.set_xlabel("block")
        ax.set_title(f"{title}\n{sub}", fontsize=10); ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Learning across blocks (1 → 3): grey = participants, red = mean", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(figdir, "fig_learning.png"); fig.savefig(p); plt.close(fig)
    return p


# --- PROGRESS BY CONDITION: does C2 improve FASTER across blocks? -----------------------------
# Method: reduce each participant x condition to ONE progress score (block-1 -> last-block change,
# as a % so conditions with different baselines are comparable), then compare the 3 conditions.
# This is the condition x block INTERACTION, made robust for a feasibility n:
#   * Friedman omnibus (nonparametric repeated-measures) across the 3 conditions;
#   * planned C2-vs-C0 and C2-vs-C1 contrasts on the paired progress scores (sign-flip permutation).
# % change is signed: for a "lower = better" metric (time), a MORE-NEGATIVE % = faster improvement.
PROGRESS_METRICS = [
    ("completion_time_s", "Completion time (s)",  "lower"),
    ("median_rt_s",       "Median press RT (s)",  "lower"),
    ("median_sync_ms",    "Bimanual offset (ms)", "lower"),
    ("accuracy",          "First-try accuracy",   "higher"),
]


def progress_by_condition(per_trial):
    """Tidy per participant x condition progress: first/last block value, delta, %change, slope."""
    need = {"participant", "condition", "block"}
    if not need.issubset(per_trial.columns) or per_trial["block"].dropna().nunique() < 2:
        return None
    pt = per_trial.dropna(subset=["block"]).copy(); pt["block"] = pt["block"].astype(int)
    metrics = [m for m, _, _ in PROGRESS_METRICS if m in pt.columns and pt[m].notna().any()]
    if not metrics:
        return None
    pcb = pt.groupby(["participant", "condition", "block"], as_index=False)[metrics].mean()
    rows = []
    for (pid, cond), g in pcb.groupby(["participant", "condition"]):
        g = g.sort_values("block")
        if g["block"].nunique() < 2:
            continue
        b0, b1 = g["block"].min(), g["block"].max()
        for m in metrics:
            fv = g.loc[g["block"] == b0, m]; lv = g.loc[g["block"] == b1, m]
            if fv.empty or lv.empty or not np.isfinite(fv.iloc[0]) or not np.isfinite(lv.iloc[0]):
                continue
            f, l = float(fv.iloc[0]), float(lv.iloc[0])
            gm = g[["block", m]].dropna()
            slope = (float(np.polyfit(gm["block"].to_numpy(float), gm[m].to_numpy(float), 1)[0])
                     if gm["block"].nunique() >= 2 else np.nan)
            rows.append({"participant": pid, "condition": cond, "metric": m,
                         "first_block": f, "last_block": l, "delta": l - f,
                         "pct_change": ((l - f) / f * 100.0) if f != 0 else np.nan,
                         "slope_per_block": slope})
    return pd.DataFrame(rows)


def _friedman(mat):
    """Friedman chi-square for an (n participants x k conditions) matrix; exact p for k=3
    (chi-square df=2 has survival exp(-x/2))."""
    n, k = mat.shape
    ranks = np.vstack([pd.Series(r).rank().to_numpy() for r in mat])
    Rj = ranks.sum(axis=0)
    chi = 12.0 / (n * k * (k + 1)) * np.sum(Rj ** 2) - 3 * n * (k + 1)
    p = float(np.exp(-chi / 2.0)) if k == 3 else np.nan
    return float(chi), p, n


def _signflip_p(diffs, n_perm=20000, seed=0):
    """Two-sided p for mean(paired diff) != 0 via sign-flip permutation (dependency-free)."""
    d = np.asarray(diffs, float); d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan, len(d)
    obs = abs(float(np.mean(d))); rng = np.random.default_rng(seed); c = 1
    for _ in range(n_perm):
        if abs(float(np.mean(d * rng.choice([-1.0, 1.0], size=len(d))))) >= obs - 1e-12:
            c += 1
    return c / (n_perm + 1), len(d)


def compare_progress(progress, metric, score="pct_change"):
    """Friedman across conditions + paired C2-vs-C0 / C2-vs-C1 on the progress score."""
    sub = progress[progress["metric"] == metric][["participant", "condition", score]].dropna()
    wide = sub.pivot_table(index="participant", columns="condition", values=score)
    conds = [c for c in CONDITIONS if c in wide.columns]
    out = {"metric": metric, "score": score}
    comp = wide[conds].dropna()                        # complete cases for the omnibus
    out["n_complete"] = int(len(comp))
    if len(comp) >= 2 and len(conds) == 3:
        chi, p, n = _friedman(comp.to_numpy())
        out["friedman_chi2"], out["friedman_p"] = chi, p
    for other in ("C0", "C1"):
        if "C2" in wide.columns and other in wide.columns:
            d = (wide["C2"] - wide[other]).dropna().to_numpy()
            p, n = _signflip_p(d)
            out[f"C2_vs_{other}_mean_diff"] = float(np.mean(d)) if len(d) else np.nan
            out[f"C2_vs_{other}_p"] = p
            out[f"C2_vs_{other}_n"] = int(len(d))
    return out


def fig_progress(per_trial, figdir):
    prog = progress_by_condition(per_trial)
    if prog is None or prog.empty:
        return None
    pt = per_trial.dropna(subset=["block"]).copy(); pt["block"] = pt["block"].astype(int)
    metrics = [(m, t, d) for m, t, d in PROGRESS_METRICS if m in pt.columns and pt[m].notna().any()]
    if not metrics:
        return None
    ncol = len(metrics)
    fig, axes = plt.subplots(2, ncol, figsize=(4.8 * ncol, 8.2), squeeze=False)
    blocks = sorted(pt["block"].unique())
    present_conds = [c for c in CONDITIONS if c in pt["condition"].unique()]
    for j, (m, title, better) in enumerate(metrics):
        # TOP: condition x block trajectory (mean across participants) — compare the slopes
        ax = axes[0][j]
        pcb = pt.groupby(["participant", "condition", "block"], as_index=False)[m].mean()
        for c in present_conds:
            g = pcb[pcb["condition"] == c].groupby("block", as_index=False)[m].mean().sort_values("block")
            ax.plot(g["block"], g[m], "-o", color=COND_COLOUR.get(c, "#333"),
                    linewidth=2.4, markersize=7, label=c)
        ax.set_xticks(blocks); ax.set_xlabel("block"); ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, title=None)
        ax.set_title(f"{title}\ntrajectory per condition", fontsize=10)
        # BOTTOM: per-participant block1->last % change by condition + Friedman p
        ax2 = axes[1][j]
        sub = prog[prog["metric"] == m]
        data, labels = [], []
        for c in present_conds:
            v = sub.loc[sub["condition"] == c, "pct_change"].dropna().to_numpy(float)
            data.append(v); labels.append(c)
        if any(len(d) for d in data):
            ax2.boxplot(data, positions=range(1, len(data) + 1), widths=0.5,
                        showmeans=True, medianprops=dict(color="black"))
            for i, (c, v) in enumerate(zip(labels, data), start=1):
                jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.15
                ax2.scatter(np.full(v.size, i) + jit, v, s=26, color=COND_COLOUR.get(c, "#333"),
                            edgecolor="white", zorder=3)
            ax2.axhline(0, color="#888", linewidth=1, linestyle="--")
            ax2.set_xticks(range(1, len(labels) + 1)); ax2.set_xticklabels(labels)
            cmp = compare_progress(prog, m, "pct_change")
            fp = cmp.get("friedman_p", float("nan"))
            gain = "more negative = faster improvement" if better == "lower" else "more positive = faster"
            ax2.set_title(f"block1→last % change  (Friedman p={fp:.3f})\n{gain}", fontsize=9)
            ax2.set_ylabel("% change")
        else:
            ax2.set_visible(False)
    fig.suptitle("Progress by condition — is C2's improvement rate better?", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(figdir, "fig_progress_by_condition.png"); fig.savefig(p); plt.close(fig)
    return p


# --- H1 focus: bimanual synchronicity — which condition improves most across blocks? ----------
def _rm_anova_perm(wide, n_perm=20000, seed=0):
    """One-way REPEATED-MEASURES ANOVA (condition effect) with a within-subject permutation
    p-value (exact-ish, no distributional assumption — appropriate for a feasibility n).
    `wide` = participants x conditions (complete cases). Returns (F, p, df1, df2, n)."""
    M = np.asarray(wide, float)
    M = M[~np.isnan(M).any(axis=1)]
    n, k = M.shape if M.ndim == 2 else (0, 0)
    df1, df2 = (k - 1, (k - 1) * (n - 1)) if n > 1 and k > 1 else (0, 0)
    if n < 2 or k < 2:
        return np.nan, np.nan, df1, df2, n

    def F_of(X):
        grand = X.mean()
        ss_cond = n * np.sum((X.mean(0) - grand) ** 2)
        ss_subj = k * np.sum((X.mean(1) - grand) ** 2)
        ss_err = np.sum((X - grand) ** 2) - ss_cond - ss_subj
        if ss_err <= 1e-12 or df2 <= 0:
            return np.nan
        return (ss_cond / df1) / (ss_err / df2)

    F = F_of(M)
    if not np.isfinite(F):
        return F, np.nan, df1, df2, n
    rng = np.random.default_rng(seed); c = 1
    for _ in range(n_perm):
        P = np.array([rng.permutation(row) for row in M])   # permute conditions WITHIN each subject
        Fp = F_of(P)
        if np.isfinite(Fp) and Fp >= F - 1e-12:
            c += 1
    return float(F), c / (n_perm + 1), df1, df2, n


def block_anova(per_trial, metric):
    """Per-block repeated-measures ANOVA across conditions on `metric`. Returns a tidy DataFrame."""
    need = {"participant", "condition", "block"}
    if not need.issubset(per_trial.columns) or metric not in per_trial.columns:
        return None
    pt = per_trial.dropna(subset=["block", metric]).copy(); pt["block"] = pt["block"].astype(int)
    pcb = pt.groupby(["participant", "condition", "block"], as_index=False)[metric].mean()
    conds = [c for c in CONDITIONS if c in pcb["condition"].unique()]
    rows = []
    for b in sorted(pcb["block"].unique()):
        wide = pcb[pcb["block"] == b].pivot_table(index="participant", columns="condition", values=metric)
        wide = wide[[c for c in conds if c in wide.columns]].dropna()
        F, p, df1, df2, n = _rm_anova_perm(wide.to_numpy())
        rows.append({"metric": metric, "block": int(b), "F": F, "p_perm": p,
                     "df1": df1, "df2": df2, "n": n})
    return pd.DataFrame(rows)


def _p_stars(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def _holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = [1.0 if (v != v) else float(v) for v in pvals]      # NaN -> 1
    m = len(p); order = sorted(range(m), key=lambda i: p[i]); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def _sig_bracket(ax, x1, x2, y, h, label):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=9,
            fontweight="bold" if label not in ("n.s.", "n/a") else "normal")


def _condition_box_with_stats(ax, wide, conditions, ylabel, colours, title="", planned=True):
    """Box + strip per condition, omnibus within-subject RM-ANOVA (permutation) in the title, and
    significance brackets. planned=True -> PLANNED CONTRASTS (C2 vs each control, Holm over two,
    both drawn); planned=False -> all pairwise, only the significant ones drawn."""
    conds = [c for c in conditions if c in wide.columns]
    data = [wide[c].dropna().to_numpy(float) for c in conds]
    bp = ax.boxplot(data, positions=range(1, len(conds) + 1), widths=0.55,
                    showmeans=True, medianprops=dict(color="black"), patch_artist=True)
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor(colours.get(c, "#ccc")); patch.set_alpha(0.30)
    for i, (c, v) in enumerate(zip(conds, data), 1):
        if len(v):
            jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.16
            ax.scatter(np.full(v.size, i) + jit, v, s=28, color=colours.get(c, "#333"),
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds); ax.set_ylabel(ylabel)
    comp = wide[conds].dropna()
    F, p, df1, df2, n = _rm_anova_perm(comp.to_numpy()) if len(conds) >= 2 else (np.nan, np.nan, 0, 0, 0)
    ax.set_title((title + "  " if title else "") + f"RM-ANOVA {_p_stars(p)} (p={p:.3f}, n={n})",
                 loc="left", fontsize=10)
    # PLANNED CONTRASTS: C2 vs each control (Holm over these two); draw both, incl. n.s.
    if planned and TREATMENT in conds:
        pairs = [(conds.index(c), conds.index(TREATMENT)) for c in CONTROLS if c in conds]
        draw_all = True
    else:
        pairs = [(i, j) for i in range(len(conds)) for j in range(i + 1, len(conds))]
        draw_all = False
    praw = []
    for i, j in pairs:
        d = wide[[conds[i], conds[j]]].dropna()
        praw.append(_signflip_p((d[conds[i]] - d[conds[j]]).to_numpy())[0] if len(d) >= 2 else np.nan)
    padj = _holm(praw)
    ys = [v.max() for v in data if len(v)]; lo = [v.min() for v in data if len(v)]
    if ys and pairs:
        ymax = max(ys); ymin = min(lo); step = (ymax - ymin) * 0.12 or 0.1; lvl = 0
        for (i, j), pa in zip(pairs, padj):
            if not np.isfinite(pa):
                continue
            if draw_all or _p_stars(pa) not in ("n.s.", "n/a"):
                _sig_bracket(ax, i + 1, j + 1, ymax + step * (1 + lvl), step * 0.3, _p_stars(pa)); lvl += 1
        if lvl:
            ax.set_ylim(top=ymax + step * (1.6 + lvl))
    # publication: anchor the y-axis at 0 for non-negative magnitude metrics (don't start at ~20 and
    # exaggerate small differences); keep the natural range if the data legitimately go negative.
    dmin = min([float(v.min()) for v in data if len(v)], default=0.0)
    ax.set_ylim(bottom=min(0.0, dmin))


def fig_sync_by_block(per_trial, figdir, metric="median_sync_ms", ylabel="Bimanual offset (ms)"):
    """Publication figure for H1: (a) synchronicity trajectory per condition across blocks with a
    per-block repeated-measures ANOVA, (b) block-1→last improvement per condition (Friedman +
    which condition improves most)."""
    need = {"participant", "condition", "block"}
    if not need.issubset(per_trial.columns) or metric not in per_trial.columns:
        return None
    pt = per_trial.dropna(subset=["block", metric]).copy(); pt["block"] = pt["block"].astype(int)
    if pt["block"].nunique() < 2:
        return None
    blocks = sorted(pt["block"].unique())
    conds = [c for c in CONDITIONS if c in pt["condition"].unique()]
    pcb = pt.groupby(["participant", "condition", "block"], as_index=False)[metric].mean()
    anova = block_anova(per_trial, metric)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # (a) trajectory per condition, mean ± SEM, with per-block RM-ANOVA
    ax = axes[0]
    ymax = pcb[metric].max()
    for c in conds:
        g = pcb[pcb["condition"] == c].groupby("block")[metric]
        m = g.mean().reindex(blocks); se = g.sem().reindex(blocks)
        ax.errorbar(blocks, m.to_numpy(), yerr=se.to_numpy(), marker="o", capsize=3,
                    capthick=1.2, elinewidth=1.2, color=COND_COLOUR.get(c, "#333"),
                    label=c, zorder=4)
    if anova is not None:
        for _, r in anova.iterrows():
            ax.annotate(f"{_p_stars(r['p_perm'])}\np={r['p_perm']:.3f}", (r["block"], ymax * 1.04),
                        ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(blocks); ax.set_xlim(blocks[0] - 0.3, blocks[-1] + 0.3)
    ax.set_ylim(bottom=0, top=ymax * 1.22)          # anchor at 0 (offset in ms; 0 = perfect sync)
    ax.set_xlabel("Block"); ax.set_ylabel(ylabel)
    ax.set_title("(a) Synchronicity across blocks by condition", loc="left")
    ax.text(0.0, -0.19, "mean ± SEM; per-block repeated-measures ANOVA (within-subject permutation)",
            transform=ax.transAxes, fontsize=8, color="#666")
    ax.legend(title="Condition", loc="best")

    # (b) block-1 → last improvement per condition + Friedman + most-improved
    ax2 = axes[1]
    prog = progress_by_condition(per_trial)
    sub = prog[prog["metric"] == metric] if prog is not None else None
    if sub is not None and not sub.empty:
        data = [sub.loc[sub["condition"] == c, "pct_change"].dropna().to_numpy(float) for c in conds]
        bp = ax2.boxplot(data, positions=range(1, len(conds) + 1), widths=0.55,
                         showmeans=True, medianprops=dict(color="black"), patch_artist=True)
        for patch, c in zip(bp["boxes"], conds):
            patch.set_facecolor(COND_COLOUR.get(c, "#ccc")); patch.set_alpha(0.30)
        for i, (c, v) in enumerate(zip(conds, data), start=1):
            jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.16
            ax2.scatter(np.full(v.size, i) + jit, v, s=30, color=COND_COLOUR.get(c, "#333"),
                        edgecolor="white", linewidth=0.5, zorder=3)
        ax2.axhline(0, color="#888", lw=1, ls="--")
        ax2.set_xticks(range(1, len(conds) + 1)); ax2.set_xticklabels(conds)
        ax2.set_ylabel("Block 1 → last  (% change)")
        cmp = compare_progress(prog, metric, "pct_change")
        fp = cmp.get("friedman_p", float("nan"))
        means = {c: (np.nanmean(d) if len(d) else np.nan) for c, d in zip(conds, data)}
        best = min(means, key=lambda c: means[c]) if any(np.isfinite(list(means.values()))) else "?"
        ax2.set_title(f"(b) Improvement by condition — Friedman {_p_stars(fp)} (p={fp:.3f})", loc="left")
        ax2.text(0.0, -0.19, f"negative = faster/better; most improved: {best}",
                 transform=ax2.transAxes, fontsize=8, color="#666")
        # planned-contrast brackets (C2 vs each control, Holm x2; paired sign-flip on % change)
        w = sub.pivot_table(index="participant", columns="condition", values="pct_change")
        pairs = [(conds.index(c), conds.index(TREATMENT)) for c in CONTROLS
                 if c in conds and TREATMENT in conds]
        praw = []
        for i, j in pairs:
            d = w[[conds[i], conds[j]]].dropna() if conds[i] in w and conds[j] in w else w.iloc[:0]
            praw.append(_signflip_p((d[conds[i]] - d[conds[j]]).to_numpy())[0] if len(d) >= 2 else np.nan)
        padj = _holm(praw)
        ys = [v.max() for v in data if len(v)]; lo = [v.min() for v in data if len(v)]
        if ys and pairs:
            ymx = max(ys); ymn = min(lo); st = (ymx - ymn) * 0.12 or 1.0; lvl = 0
            for (i, j), pa in zip(pairs, padj):
                if np.isfinite(pa):
                    _sig_bracket(ax2, i + 1, j + 1, ymx + st * (1 + lvl), st * 0.3, _p_stars(pa)); lvl += 1
            if lvl:
                ax2.set_ylim(top=ymx + st * (1.6 + lvl))
    else:
        ax2.set_visible(False)

    fig.suptitle("Bimanual synchronicity — learning by condition (H1)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    p = os.path.join(figdir, "fig_sync_by_block.png"); fig.savefig(p); plt.close(fig)
    return p


def c1_c2_improvement(per_trial, metric, better="lower"):
    """Per participant, mean C1 vs mean C2 (trial-level) + a Welch two-sample C1-vs-C2 test on that
    person's trials. improved_sig = C2 moved the BETTER direction AND p<.05. Returns a tidy table."""
    rows = []
    for pid, g in per_trial.groupby("participant"):
        c1 = g.loc[g["condition"] == "C1", metric].dropna().to_numpy(float)
        c2 = g.loc[g["condition"] == "C2", metric].dropna().to_numpy(float)
        if len(c1) < 1 or len(c2) < 1:
            continue
        m1, m2 = float(np.mean(c1)), float(np.mean(c2))
        t, dfw, p = _welch_ttest(c2, c1)
        better_c2 = (m2 < m1) if better == "lower" else (m2 > m1)
        rows.append({"participant": pid, "mean_C1": m1, "mean_C2": m2, "diff_C2_minus_C1": m2 - m1,
                     "welch_t": t, "welch_df": dfw, "welch_p": p, "n_C1": len(c1), "n_C2": len(c2),
                     "improved_sig": bool(better_c2 and np.isfinite(p) and p < .05),
                     "improved_nominal": bool(better_c2)})
    return pd.DataFrame(rows)


def fig_c1_c2_improvers(per_trial, figdir, metric, ylabel, title, fname, better="lower"):
    """Paired C1→C2 dumbbell per participant, HIGHLIGHTING those who significantly improved (Welch
    p<.05 in the better direction): gold + bold + asterisk. Answers 'who did the robot actually help?'"""
    imp = c1_c2_improvement(per_trial, metric, better)
    if imp is None or imp.empty:
        return None
    imp = imp.sort_values("mean_C2")
    nsig = int(imp["improved_sig"].sum()); ntot = len(imp)
    sig_ids = [f"P{p}" for p in imp.loc[imp["improved_sig"], "participant"].tolist()]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for _, r in imp.iterrows():
        sig, nom = r["improved_sig"], r["improved_nominal"]
        col = "#E4A11B" if sig else ("#4C78A8" if nom else "#BBBBBB")
        lw = 2.8 if sig else 1.4
        ax.plot([1, 2], [r["mean_C1"], r["mean_C2"]], "-o", color=col, lw=lw,
                markersize=8 if sig else 5, zorder=5 if sig else 2, alpha=0.95 if sig else 0.65)
        if sig or nom:
            ax.annotate(f"P{r['participant']}" + ("*" if sig else ""), (2, r["mean_C2"]),
                        fontsize=8.5, fontweight="bold" if sig else "normal", color=col,
                        xytext=(7, 0), textcoords="offset points", va="center")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["C1 (passive stand)", "C2 (robot)"])
    ax.set_xlim(0.7, 2.55); ax.set_ylabel(ylabel)
    if float(np.nanmin([imp["mean_C1"].min(), imp["mean_C2"].min()])) >= 0:
        ax.set_ylim(bottom=0)
    ax.set_title(f"{title}\n{nsig}/{ntot} significantly improved C1→C2 (Welch p<.05): "
                 f"{', '.join(sig_ids) or '—'}", loc="left", fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#E4A11B", lw=2.8, marker="o", label="significant improvement (p<.05) *"),
        Line2D([0], [0], color="#4C78A8", lw=1.4, marker="o", label="improved, not significant"),
        Line2D([0], [0], color="#BBBBBB", lw=1.4, marker="o", label="no improvement")],
        loc="best", fontsize=8)
    fig.tight_layout()
    p = os.path.join(figdir, fname); fig.savefig(p); plt.close(fig)
    return p


# --- SINGLETON (one-handed / incomplete) presses from the raw button stream ------------------
# A pad pair needs BOTH hands: B (one hand) + some A_i (other hand). A "singleton" is a press
# episode — from a contact going DOWN to when it LEAVES (releases) — during which the OTHER hand
# never joins, no matter how long it is held. If the partner engages at ANY point while the contact
# is held, it is a completed pair, not a singleton (no time window: the second hand may come 10 ms
# or 3 s later — as long as it comes during the hold, it counts as paired). This is a distinct
# bimanual-error metric, separate from the wrong-face error_rate and from the synchronicity offset.
_A_COLS = [f"A{i+1}" for i in range(8)]
_BTN_META_RE = re.compile(r"(C\d)_b(\d+)_seq(\d+)", re.I)


def _high_segments(mask):
    """Press episodes of a boolean signal -> list of (i0, i1_exclusive), press-down to release."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return []
    d = np.diff(m.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(m)]
    return list(zip(starts, ends))


def _singletons_one_file(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "B" not in df.columns:
        return None
    Bmask = (pd.to_numeric(df["B"], errors="coerce") > 0.5).to_numpy()
    a_present = [c for c in _A_COLS if c in df.columns]
    Amask = np.zeros(len(df), bool)
    for c in a_present:
        Amask |= (pd.to_numeric(df[c], errors="coerce") > 0.5).to_numpy()

    def split(segs, partner_mask):
        singl = paired = 0
        for i0, i1 in segs:
            if partner_mask[i0:i1].any():         # the other hand joined DURING the hold -> pair
                paired += 1
            else:                                 # pressed and released, partner never came -> singleton
                singl += 1
        return singl, paired

    B_single, B_paired = split(_high_segments(Bmask), Amask)   # B alone (no A during its hold)
    A_single, _ = split(_high_segments(Amask), Bmask)          # an A-face alone (no B during its hold)
    singles = B_single + A_single
    n_presses = len(_high_segments(Bmask)) + len(_high_segments(Amask))
    return {"n_completed_pairs": B_paired, "singleton_presses": singles,
            "singleton_rate": (singles / n_presses) if n_presses else np.nan,
            "n_presses": n_presses}


def singleton_analysis(recordings, outdir):
    """Scan the raw *_buttons.csv files, count one-handed/incomplete (singleton) presses per trial,
    aggregate by condition, and write CSVs + a figure. Returns the per-trial DataFrame or None."""
    import glob
    paths = glob.glob(os.path.join(str(recordings), "**", "*_buttons.csv"), recursive=True)
    rows = []
    for p in paths:
        folder = os.path.basename(os.path.dirname(p))
        part = folder.replace("Participant", "").strip()
        if part.lower().startswith(EXCLUDE_PARTICIPANT_PREFIXES):   # skip Participanttry etc.
            continue
        m = _BTN_META_RE.search(os.path.basename(p))
        cond = m.group(1).upper() if m else None
        block = int(m.group(2)) if m else None
        res = _singletons_one_file(p)
        if res is None or cond is None:
            continue
        res.update({"participant": part, "condition": cond, "block": block})
        rows.append(res)
    if not rows:
        print("[singleton] no *_buttons.csv found — skipped.")
        return None
    os.makedirs(str(outdir), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "singleton_by_trial.csv"), index=False)
    bycond = (df.groupby(["participant", "condition"], as_index=False)
                .agg(singleton_rate=("singleton_rate", "mean"),
                     singleton_presses=("singleton_presses", "mean"),
                     n_completed_pairs=("n_completed_pairs", "mean")))
    bycond.to_csv(os.path.join(outdir, "singleton_by_condition.csv"), index=False)
    figdir = os.path.join(str(outdir), "figures"); os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    conds = [c for c in CONDITIONS if c in bycond["condition"].unique()]
    wide = bycond.pivot_table(index="participant", columns="condition", values="singleton_rate")
    _condition_box_with_stats(ax, wide, conds,
                              "singleton-press rate  (one-handed / total)", COND_COLOUR,
                              title="One-handed (singleton) presses")
    fig.suptitle("One-handed (singleton) presses by condition", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_singletons.png")); plt.close(fig)
    print("\nSingleton (one-handed) presses — mean rate by condition "
          "(pressed & released with the other hand never joining):")
    for c in conds:
        v = bycond.loc[bycond["condition"] == c, "singleton_rate"]
        print(f"  {c}: rate={v.mean():.3f}  (per-trial count "
              f"{df.loc[df.condition == c, 'singleton_presses'].mean():.1f})")
    return df


# --- PER-PARTICIPANT synchronicity ANOVA: for how many people did C2 actually improve sync? ----
# For each participant, a one-way ANOVA across C0/C1/C2 on their TRIAL-LEVEL bimanual offset (does
# condition matter for this person), plus the key C2-vs-C1 contrast (two-sample permutation on that
# person's C1 vs C2 trials). "Improved with C2" = C2 mean offset LOWER than C1 AND p<.05.
def _oneway_F(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, 0, 0
    allv = np.concatenate(groups); grand = allv.mean(); k = len(groups); N = len(allv)
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df1, df2 = k - 1, N - k
    if ssw <= 1e-12 or df2 <= 0:
        return np.nan, df1, df2
    return (ssb / df1) / (ssw / df2), df1, df2


def _anova_perm_p(groups, n_perm=10000, seed=0):
    F, _, _ = _oneway_F(groups)
    if not np.isfinite(F):
        return np.nan
    sizes = [len(g) for g in groups]; data = np.concatenate([np.asarray(g, float) for g in groups])
    rng = np.random.default_rng(seed); c = 1
    for _ in range(n_perm):
        perm = rng.permutation(data); i = 0; gs = []
        for s in sizes:
            gs.append(perm[i:i + s]); i += s
        Fp, _, _ = _oneway_F(gs)
        if np.isfinite(Fp) and Fp >= F - 1e-12:
            c += 1
    return c / (n_perm + 1)


def _twosample_perm_p(a, b, n_perm=10000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(a.mean() - b.mean()); pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed); c = 1
    for _ in range(n_perm):
        p = rng.permutation(pool)
        if abs(p[:na].mean() - p[na:].mean()) >= obs - 1e-12:
            c += 1
    return c / (n_perm + 1)


def _welch_ttest(a, b):
    """Independent two-sample Welch t-test (unequal variance). Returns (t, df, two-sided p)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se <= 0:
        return np.nan, np.nan, np.nan
    t = (a.mean() - b.mean()) / se
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:                                       # normal-approx fallback if no scipy
        import math
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    return float(t), float(df), p


# =============================================================================================
# PER BUTTON-PAIR analysis: reaction time, synchronicity and singletons for each of the 8 faces,
# and how coordination scales with the DISTANCE between successive buttons in the presentation order.
# Board geometry (face -> 3x3 grid cell) is derived exactly as in face_mapping_check.py.
# =============================================================================================
# face -> (row, col) on the 3x3 board (row0=top, col0=left), from the fixed marker anchors.
FACE_CELL = {1: (2, 0), 2: (1, 0), 3: (0, 0), 4: (0, 1),
             5: (0, 2), 6: (1, 2), 7: (2, 2), 8: (2, 1)}
CELL_PITCH = 1.0    # distance unit = grid cells; set to the physical cell spacing (mm) if known


def _face_dist(a, b):
    """Euclidean distance between two faces on the board grid, in CELL_PITCH units."""
    if a not in FACE_CELL or b not in FACE_CELL:
        return np.nan
    (r1, c1), (r2, c2) = FACE_CELL[a], FACE_CELL[b]
    return float(np.hypot(r1 - r2, c1 - c2)) * CELL_PITCH


def _button_pairs_one_file(path):
    """One trial's *_buttons.csv -> per button-pair rows: face, press order, bimanual offset (sync),
    singleton flag, reaction time (gap from the previous pair's release to this press onset), and the
    board distance from the previous face. Uses the same B / A_i pad model as the singleton parser."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "B" not in df.columns:
        return None
    if "t_device_ms" in df.columns:
        t = pd.to_numeric(df["t_device_ms"], errors="coerce").to_numpy(float) / 1000.0
    elif "t_clock" in df.columns:
        t = pd.to_numeric(df["t_clock"], errors="coerce").to_numpy(float)
    else:
        t = np.arange(len(df), dtype=float) / 200.0    # assume 200 Hz if no clock
    Bmask = (pd.to_numeric(df["B"], errors="coerce") > 0.5).to_numpy()
    B_segs = _high_segments(Bmask)

    def t_at(i, last):
        i = min(max(i, 0), len(t) - 1)
        return float(t[i])

    presses = []
    for c in _A_COLS:
        if c not in df.columns:
            continue
        face = int(c[1:])                                  # "A3" -> 3
        Am = (pd.to_numeric(df[c], errors="coerce") > 0.5).to_numpy()
        for i0, i1 in _high_segments(Am):
            onset = t_at(i0, False); a_rel = t_at(i1 - 1, True)
            partner = next(((b0, b1) for b0, b1 in B_segs if b0 < i1 and b1 > i0), None)
            if partner is not None:
                b0, b1 = partner
                sync_ms = abs(onset - t_at(b0, False)) * 1000.0
                pair_end = max(a_rel, t_at(b1 - 1, True))
                singleton = 0
            else:
                sync_ms = np.nan; pair_end = a_rel; singleton = 1
            presses.append({"face": face, "onset": onset, "end": pair_end,
                            "sync_ms": sync_ms, "singleton": singleton})
    if not presses:
        return None
    p = pd.DataFrame(presses).sort_values("onset").reset_index(drop=True)
    p["seq"] = np.arange(1, len(p) + 1)                    # position in the presentation order
    p["rt_s"] = p["onset"] - p["end"].shift(1)            # end of previous press -> this press onset
    prev = p["face"].shift(1)
    p["dist"] = [_face_dist(a, b) if pd.notna(a) else np.nan for a, b in zip(prev, p["face"])]
    return p


def button_pair_analysis(recordings, outdir):
    """Build the per-button-pair table from all *_buttons.csv, write it, and make (A) per-face figures
    of RT / synchronicity / singletons and (B) distance-vs-coordination correlations per condition."""
    import glob
    paths = glob.glob(os.path.join(str(recordings), "**", "*_buttons.csv"), recursive=True)
    frames = []
    for p in paths:
        folder = os.path.basename(os.path.dirname(p))
        part = folder.replace("Participant", "").strip()
        if part.lower().startswith(EXCLUDE_PARTICIPANT_PREFIXES):
            continue
        m = _BTN_META_RE.search(os.path.basename(p))
        cond = m.group(1).upper() if m else None
        block = int(m.group(2)) if m else None
        d = _button_pairs_one_file(p)
        if d is None or cond is None:
            continue
        d["participant"] = part; d["condition"] = cond; d["block"] = block
        frames.append(d)
    if not frames:
        print("[button-pairs] no *_buttons.csv found — skipped.")
        return None
    allp = pd.concat(frames, ignore_index=True)
    bpdir = os.path.join(str(outdir), "button_pairs"); os.makedirs(bpdir, exist_ok=True)
    allp.to_csv(os.path.join(bpdir, "per_button_press.csv"), index=False)
    fig_button_pair_by_face(allp, bpdir)
    fig_distance_vs_coordination(allp, bpdir)
    print(f"[button-pairs] {len(allp)} presses over {allp['participant'].nunique()} participants "
          f"-> {bpdir}/")
    return allp


def fig_button_pair_by_face(allp, bpdir):
    """(A) For each of the 8 faces (button pairs): reaction time, bimanual offset, singleton rate,
    one line per condition (mean ± SEM)."""
    conds = [c for c in CONDITIONS if c in allp["condition"].unique()]
    metrics = [("rt_s", "Reaction time (s)  [prev release → press]"),
               ("sync_ms", "Bimanual offset (ms)"),
               ("singleton", "Singleton rate")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (col, ylab) in zip(axes, metrics):
        for c in conds:
            sub = allp[allp["condition"] == c]
            g = sub.groupby("face")[col]
            m = g.mean().reindex(range(1, 9)); se = g.sem().reindex(range(1, 9))
            ax.errorbar(range(1, 9), m.to_numpy(), yerr=se.to_numpy(), marker="o", capsize=3,
                        capthick=1.1, elinewidth=1.1, color=COND_COLOUR.get(c, "#333"), label=c)
        ax.set_xlabel("face (button pair)"); ax.set_ylabel(ylab); ax.set_xticks(range(1, 9))
        vals = pd.to_numeric(allp[col], errors="coerce")
        if vals.min() >= 0:
            ax.set_ylim(bottom=0)
    axes[0].legend(title="Condition", loc="best")
    fig.suptitle("Per button-pair (face): reaction time, synchronicity and singletons (mean ± SEM)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(bpdir, "fig_button_pair_by_face.png"); fig.savefig(p); plt.close(fig)
    return p


def _corr_slope(x, y, kind="pearson", n_perm=10000, seed=0):
    """Correlation with permutation two-sided p (dependency-free). Returns (r, p, slope, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, n
    if kind == "spearman":
        xr = pd.Series(x).rank().to_numpy(); yr = pd.Series(y).rank().to_numpy()
        r = float(np.corrcoef(xr, yr)[0, 1])
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed); cnt = 1
    base = (pd.Series(x).rank().to_numpy() if kind == "spearman" else x)
    yy = (pd.Series(y).rank().to_numpy() if kind == "spearman" else y)
    for _ in range(n_perm):
        rp = float(np.corrcoef(base, rng.permutation(yy))[0, 1])
        if abs(rp) >= abs(r) - 1e-12:
            cnt += 1
    return r, cnt / (n_perm + 1), slope, n


def fig_distance_vs_coordination(allp, bpdir):
    """(B) Does coordination worsen with the DISTANCE between successive buttons, and does the robot
    flatten that? Bimanual offset and singleton rate vs board distance, one line per condition, with
    per-condition correlation + slope. Points are participant×distance means (reduces pseudoreplication)."""
    conds = [c for c in CONDITIONS if c in allp["condition"].unique()]
    sub = allp.dropna(subset=["dist"]).copy()
    if sub.empty:
        return None
    metrics = [("sync_ms", "Bimanual offset (ms)  [lower=better]"),
               ("singleton", "Singleton rate  [lower=better]")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    stat_rows = []
    for ax, (col, ylab) in zip(axes, metrics):
        for c in conds:
            g = sub[sub["condition"] == c]
            # participant × distance means, then a per-condition line + correlation
            pm = g.groupby(["participant", "dist"], as_index=False)[col].mean()
            dm = pm.groupby("dist")[col]
            xs = dm.mean().index.to_numpy(float); ys = dm.mean().to_numpy(); se = dm.sem().to_numpy()
            ax.errorbar(xs, ys, yerr=se, marker="o", capsize=3, color=COND_COLOUR.get(c, "#333"),
                        label=c, zorder=4)
            r, p, slope, n = _corr_slope(pm["dist"], pm[col], "spearman")
            rp, pp, _, _ = _corr_slope(pm["dist"], pm[col], "pearson")
            if np.isfinite(slope):
                xx = np.linspace(pm["dist"].min(), pm["dist"].max(), 50)
                b1, b0 = np.polyfit(pm["dist"], pm[col], 1)
                ax.plot(xx, b0 + b1 * xx, "--", color=COND_COLOUR.get(c, "#333"), lw=1.2, alpha=0.7)
            stat_rows.append({"metric": col, "condition": c, "pearson_r": rp, "pearson_p": pp,
                              "spearman_rho": r, "spearman_p": p, "slope_per_unit": slope, "n": n})
        ax.set_xlabel(f"distance between successive buttons ({'cells' if CELL_PITCH == 1 else 'mm'})")
        ax.set_ylabel(ylab)
        if pd.to_numeric(sub[col], errors="coerce").min() >= 0:
            ax.set_ylim(bottom=0)
        ax.legend(title="Condition", loc="best")
    fig.suptitle("Coordination vs distance between successive buttons — by condition\n"
                 "(does the robot (C2) flatten the distance penalty?)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(bpdir, "fig_distance_vs_coordination.png"); fig.savefig(p); plt.close(fig)
    pd.DataFrame(stat_rows).to_csv(os.path.join(bpdir, "distance_vs_coordination_stats.csv"), index=False)
    return p


def per_participant_sync(per_trial, outdir, metric="median_sync_ms"):
    """One-way ANOVA per participant on trial-level bimanual offset + the C2-vs-C1 contrast.
    Counts how many participants had C2 SIGNIFICANTLY lower (improved) vs the passive stand."""
    need = {"participant", "condition", metric}
    if not need.issubset(per_trial.columns):
        print(f"\n[per-participant sync] missing columns {need - set(per_trial.columns)} — skipped.")
        return None
    rows = []
    for pid, g in per_trial.groupby("participant"):
        groups = {c: g.loc[g["condition"] == c, metric].dropna().to_numpy(float)
                  for c in CONDITIONS if (g["condition"] == c).any()}
        if len(groups) < 2:
            continue
        F, df1, df2 = _oneway_F(list(groups.values()))
        row = {"participant": pid, "anova_F": F, "anova_df1": df1, "anova_df2": df2,
               "anova_p": _anova_perm_p(list(groups.values()))}
        for c in CONDITIONS:
            if c in groups:
                row[f"mean_{c}"] = float(groups[c].mean()); row[f"n_{c}"] = int(len(groups[c]))
        if "C1" in groups and "C2" in groups:
            c1, c2 = groups["C1"], groups["C2"]
            diff = float(c2.mean() - c1.mean())                       # <0 = C2 tighter (better)
            t, dfw, p_t = _welch_ttest(c2, c1)                        # independent 2-sample t-test
            row["C2_minus_C1_ms"] = diff
            row["welch_t"] = t; row["welch_df"] = dfw
            row["C2_vs_C1_p_ttest"] = p_t
            row["C2_vs_C1_p_perm"] = _twosample_perm_p(c2, c1)        # permutation p (robustness)
            row["C2_improved_sig"] = bool(diff < 0 and np.isfinite(p_t) and p_t < .05)
            row["C2_improved_nominal"] = bool(diff < 0)
        rows.append(row)
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "per_participant_sync_anova.csv"), index=False)
    ntot = len(tab)

    def ids(mask):
        return ", ".join(f"P{p}" for p in tab.loc[mask, "participant"].tolist()) or "—"

    anova_mask = tab["anova_p"] < .05
    sig_mask = tab.get("C2_improved_sig", pd.Series(False, index=tab.index)).fillna(False).astype(bool)
    nom_mask = tab.get("C2_improved_nominal", pd.Series(False, index=tab.index)).fillna(False).astype(bool)
    print("\n" + "=" * 64)
    print("PER-PARTICIPANT synchronicity — did C2 actually improve it? (vs C1)")
    print("=" * 64)
    for _, r in tab.iterrows():
        d = r.get("C2_minus_C1_ms", float("nan")); t = r.get("welch_t", float("nan"))
        dfw = r.get("welch_df", float("nan")); pt = r.get("C2_vs_C1_p_ttest", float("nan"))
        flag = "IMPROVED*" if r.get("C2_improved_sig") else ("lower(ns)" if r.get("C2_improved_nominal") else "not lower")
        print(f"  P{r['participant']}: ANOVA F({int(r['anova_df1'])},{int(r['anova_df2'])})="
              f"{r['anova_F']:.2f} p={r['anova_p']:.3f} | C0={r.get('mean_C0', float('nan')):.0f} "
              f"C1={r.get('mean_C1', float('nan')):.0f}(n={int(r.get('n_C1', 0))}) "
              f"C2={r.get('mean_C2', float('nan')):.0f}(n={int(r.get('n_C2', 0))}) ms | "
              f"C2-C1={d:+.0f}ms  t({dfw:.0f})={t:+.2f} p={pt:.3f}  ->  {flag}")
    print(f"\n  C2 SIGNIFICANTLY improved sync vs C1 (t-test p<.05): {int(sig_mask.sum())}/{ntot}"
          f"   ->  {ids(sig_mask)}")
    print(f"  C2 nominally lower than C1 (direction only):         {int(nom_mask.sum())}/{ntot}"
          f"   ->  {ids(nom_mask)}")
    print(f"  per-participant omnibus ANOVA significant:           {int(anova_mask.sum())}/{ntot}"
          f"   ->  {ids(anova_mask)}")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="synthetic_recordings")
    ap.add_argument("--questionnaires", default=None,
                    help="defaults to <recordings>/questionnaires.csv if present")
    ap.add_argument("--out", default="analysis_out")
    ap.add_argument("--drop-warmup", type=int, default=0, metavar="K",
                    help="drop the first K trials of each condition as warm-up "
                         "(replaces the old --exclude-block1; default 0)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.out, "figures"); os.makedirs(figdir, exist_ok=True)

    per_trial = A.build_per_trial(args.recordings)
    if per_trial.empty:
        raise SystemExit(f"No trial files found under {args.recordings!r} "
                         f"(expected P*_T*_C*_b*_seq*_*_buttons.csv etc.).")

    per_trial, n_test = _drop_test_participants(per_trial)
    if n_test:
        print(f"[analysis] excluded {n_test} trial(s) from test folders "
              f"(Participant{{{'/'.join(EXCLUDE_PARTICIPANT_PREFIXES)}}}*).")
    if per_trial.empty:
        raise SystemExit("No trials left after excluding test participants.")

    per_trial, n_dropped = _drop_warmup(per_trial, args.drop_warmup)

    qpath = args.questionnaires or os.path.join(args.recordings, "questionnaires.csv")
    quest = A.load_questionnaires(qpath)

    ppc = A.aggregate_participant_condition(per_trial, quest)
    desc = A.condition_descriptives(ppc)

    per_trial.to_csv(os.path.join(args.out, "per_trial_metrics.csv"), index=False)
    ppc.to_csv(os.path.join(args.out, "per_participant_condition.csv"), index=False)
    desc.to_csv(os.path.join(args.out, "condition_descriptives.csv"), index=False)

    figs = [fig_primary(ppc, figdir), fig_participant_lines(ppc, figdir), fig_c2(ppc, figdir)]

    # statistical test table for the primary metrics (omnibus RM-ANOVA + Holm pairwise)
    prim_stats = condition_stats_table(ppc, PRIMARY)
    prim_stats.to_csv(os.path.join(args.out, "primary_stats.csv"), index=False)

    # across-block learning / practice effect (block 1 -> 2 -> 3)
    lb_pb, lb_blk, lb_trend = learning_by_block(per_trial)
    if lb_pb is not None:
        lb_pb.to_csv(os.path.join(args.out, "learning_by_participant_block.csv"), index=False)
        if lb_trend is not None and not lb_trend.empty:
            lb_trend.to_csv(os.path.join(args.out, "learning_trends.csv"), index=False)
        figs.append(fig_learning(per_trial, figdir))

    # progress by condition — does C2 improve FASTER? (condition x block interaction)
    prog = progress_by_condition(per_trial)
    prog_cmp = None
    if prog is not None and not prog.empty:
        prog.to_csv(os.path.join(args.out, "progress_by_participant_condition.csv"), index=False)
        prog_cmp = pd.DataFrame([compare_progress(prog, m, "pct_change")
                                 for m, _, _ in PROGRESS_METRICS if m in prog["metric"].unique()])
        prog_cmp.to_csv(os.path.join(args.out, "progress_comparison.csv"), index=False)
        figs.append(fig_progress(per_trial, figdir))

    # H1: one-handed / incomplete (singleton) presses from the raw button stream
    sing_df = singleton_analysis(args.recordings, args.out)

    # per button-pair (face) RT / sync / singletons + distance-vs-coordination by condition
    button_pair_analysis(args.recordings, args.out)

    # H1: per-participant ANOVA — for how many people did C2 actually improve synchronicity?
    per_participant_sync(per_trial, args.out, "median_sync_ms")

    # H1 synchronicity focus: per-block RM-ANOVA + which condition improves most
    sync_anova = block_anova(per_trial, "median_sync_ms")
    if sync_anova is not None and not sync_anova.empty:
        sync_anova.to_csv(os.path.join(args.out, "sync_block_anova.csv"), index=False)
        figs.append(fig_sync_by_block(per_trial, figdir))

    # HIGHLIGHT which participants the robot actually helped C1→C2 (significant per-participant Welch)
    figs.append(fig_c1_c2_improvers(
        per_trial, figdir, "median_sync_ms", "Bimanual offset (ms)  [lower = better]",
        "Synchronicity: C1 (passive) → C2 (robot) per participant",
        "fig_sync_c1_c2_improvers.png", better="lower"))
    if sing_df is not None and {"participant", "condition", "singleton_rate"}.issubset(sing_df.columns):
        figs.append(fig_c1_c2_improvers(
            sing_df, figdir, "singleton_rate", "Singleton rate  [lower = better]",
            "One-handed (singleton) presses: C1 → C2 per participant",
            "fig_singleton_c1_c2_improvers.png", better="lower"))

    n_part = per_trial["participant"].nunique()
    warm = f"{n_dropped} warm-up trials dropped" if args.drop_warmup else "all trials"
    print(f"[analysis] {len(per_trial)} trials, {n_part} participants ({warm}).")
    print(f"[analysis] tables + figures -> {args.out}/")
    for f in filter(None, figs):
        print(f"           {f}")
    # quick console summary of the headline metrics (error-rate deliberately omitted)
    show = ["completion_time_s", "median_rt_s", "median_sync_ms", "rtlx",
            "reposition_correctness", "functional_delay_s"]
    d = desc[desc["metric"].isin(show)].copy()
    if not d.empty:
        print("\nCondition means (95% CI):")
        for metric in show:
            sub = d[d["metric"] == metric]
            if sub.empty:
                continue
            parts = [f"{r.condition} {r['mean']:.2f} [{r['ci95_low']:.2f},{r['ci95_high']:.2f}]"
                     for _, r in sub.iterrows()]
            print(f"  {metric:24s} " + "   ".join(parts))

    if lb_trend is not None and not lb_trend.empty:
        print("\nLearning across blocks (block-1 mean -> last-block mean; slope per block):")
        for _, r in lb_trend.iterrows():
            tag = "IMPROVED" if r["improved_1_to_last"] else "no gain"
            print(f"  {r['metric']:20s} {r['first_block_mean']:8.2f} -> {r['last_block_mean']:8.2f}"
                  f"  slope={r['slope_per_block']:+.2f}/blk  r={r['pearson_r']:+.2f}"
                  f" (p={r['pearson_p']:.3f})  [{tag}]")
    elif lb_pb is None:
        print("\n[learning] skipped — no usable 'block' column with >=2 blocks in per_trial.")

    if prog_cmp is not None and not prog_cmp.empty:
        print("\nProgress by condition (block1→last % change; is C2 faster?):")
        for _, r in prog_cmp.iterrows():
            fp = r.get("friedman_p", float("nan"))
            d0 = r.get("C2_vs_C0_mean_diff", float("nan")); p0 = r.get("C2_vs_C0_p", float("nan"))
            d1 = r.get("C2_vs_C1_mean_diff", float("nan")); p1 = r.get("C2_vs_C1_p", float("nan"))
            print(f"  {r['metric']:20s} Friedman p={fp:.3f}  |  "
                  f"C2-C0 Δ%={d0:+.1f} (p={p0:.3f})   C2-C1 Δ%={d1:+.1f} (p={p1:.3f})")

    if not prim_stats.empty:
        print("\nPrimary metrics — omnibus RM-ANOVA + planned contrasts (C2 vs each control, "
              "Holm x2, direction-aware):")
        for _, r in prim_stats.iterrows():
            bits = []
            for c in CONTROLS:
                pk, dk, sk = f"{TREATMENT}_vs_{c}_p_holm", f"{TREATMENT}_vs_{c}_diff", f"{TREATMENT}_vs_{c}_supported"
                if pk in r.index and pd.notna(r[pk]):
                    tag = "OK" if r[sk] else "no"
                    bits.append(f"C2-{c} Δ={r[dk]:+.1f} p={r[pk]:.3f}[{tag}]")
            print(f"  {r['metric']:18s} RM-ANOVA F({int(r['rm_anova_df1'])},{int(r['rm_anova_df2'])})="
                  f"{r['rm_anova_F']:.2f} p={r['rm_anova_p']:.3f}  |  " + "  ".join(bits))

    if sync_anova is not None and not sync_anova.empty:
        print("\nSynchronicity — per-block ANOVA across conditions (median_sync_ms):")
        for _, r in sync_anova.iterrows():
            print(f"  block {int(r['block'])}: F({int(r['df1'])},{int(r['df2'])})={r['F']:.2f}"
                  f"  p={r['p_perm']:.3f}  (n={int(r['n'])})")


if __name__ == "__main__":
    main()
