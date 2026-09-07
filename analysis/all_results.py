#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thesis_results.py -- single-entry pipeline for Results (Chapter 4).

Replaces run_analysis.py / results_analysis.py / orient_telemetry_analysis2.py /
analyze_tlx.py for everything that is *reported*. Produces every figure and table in
the chapter, plus a LaTeX caption file per figure whose numbers are generated from the
data (so a caption can never drift from what was plotted).

Design rules, each of which fixes a specific defect in the previous chapter draft:

  R1  NO SILENT EXCLUSIONS. All 81 C2 trials are retained. The two trials with
      completion time > EXTREME_COMPLETION_S are marked `extreme` and plotted as open
      markers; every validation correlation is reported twice (all trials / extremes
      excluded) as a declared sensitivity analysis. n is never quietly 79.
  R2  CLUSTERING. No trial-level correlation is reported as if trials were independent.
      Primary estimate = within-participant r; pooled r is reported alongside it with a
      participant-cluster bootstrap CI, and labelled descriptive.
  R3  MULTIPLICITY. Every hypothesis-family post-hoc test is Bonferroni-corrected
      within its family, and the family size is stated wherever the corrected p is
      printed. The validation correlations form one separate declared exploratory
      family and keep their Holm correction. Both Pearson and Spearman are always
      reported; the chapter must not lean on whichever one clears .05.
  R4  INFLUENCE. Every participant-level (n<=10) correlation carries a leave-one-out
      range and names the most influential participant.
  R5  UNITS FROM CONSTANTS. Captions interpolate PEN_CONVERGE_MM etc., so "15 mm" cannot
      become "15 cm".
  R6  NULLS ARE REPORTED AS NULLS. There is no equivalence testing in the primary
      analysis: a contrast that does not reach alpha is reported as showing no
      significant difference and nothing further is claimed from it. Each SESOI keeps
      its written justification and is drawn on the figures as a band of practical
      relevance, but it drives no test.
  R7  INTERACTIONS ARE TESTED. The TLX early->endpoint claim is a condition x phase
      interaction, not three separate within-condition tests.
  R8  HETEROGENEITY IS TESTED. "Cost is a property of the system, not operator skill" is
      backed by a random-slope LRT on completion time as well as on synchrony.
  R9  FIGURE HYGIENE. No statistics inside axes titles, no duplicated titles, no
      "p = 0.000", colourblind-safe palette with redundant markers, explicit (a)/(b)
      panel labels, and a real caption for every figure.

Hypotheses (fixed; every section maps to exactly one):
  H1 performance : C2 faster + tighter bimanual synchrony than C0/C1, no accuracy loss.
                   Key contrast C2 vs C1.
  H2 workload    : C2 lowers NASA-TLX vs C0/C1.
  H3 acceptance  : operators prefer C2 (rank + intention to use); SUS vs 68 benchmark.

Inputs (same schema as the old scripts):
  --trials      per_trial_metrics.csv   participant, condition, block, trial,
                                        completion_time_s, median_sync_ms, accuracy,
                                        settle_time_s, median_rt_s
  --singletons  singleton_by_trial.csv  participant, condition, trial, singleton_rate
  --telemetry   per_c2_trial.csv        participant, trial, completion_time_s,
                                        board_rms_mm, int_rms_mm, wander_norm,
                                        oof_frac, pen_oof_events, orient_frames,
                                        pen_gap_med_mm, pen_sep_deg, frac_converged
  --workbook    SUS_NASA-TLX.xlsx       sheets: "NASA-TLX C0/C1/C2", "Preference", "SUS"

Usage:
  python thesis_results.py --outdir results_out
  python thesis_results.py --selftest        # synthesises inputs, runs end to end
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except Exception:
    HAVE_SM = False

try:                    # optional: preferred backend for the RM ANOVA, not required
    import pingouin as pg
    HAVE_PG = True
except Exception:
    HAVE_PG = False

# which RM-ANOVA implementation actually ran; stamped into ANALYSIS_MANIFEST.csv
ANOVA_BACKEND = "pingouin.rm_anova" if HAVE_PG else "internal (numpy/scipy)"

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:                                    # mixed-model retries are noisy, not errors
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

# =============================================================================
# CONFIG
# =============================================================================

CONDITIONS = ["C0", "C1", "C2"]
COND_TITLE = {"C0": "C0\nhands", "C1": "C1\npassive", "C2": "C2\nactive"}

# Okabe-Ito, colourblind safe. Markers are redundant with colour throughout (R9).
PALETTE = {"C0": "#666666", "C1": "#0072B2", "C2": "#D55E00"}
MARKER = {"C0": "o", "C1": "s", "C2": "^"}

TREATMENT = "C2"
KEY_CONTRAST = ("C1", "C2")          # the decisive contrast for all three hypotheses

# ---- SESOI, each with the justification that must appear in the text (R6) ----
# metric -> (direction of "better", SESOI in metric units, label, justification)
METRIC_SPEC = {
    "completion_time_s": ("lower", 2.0, "Sequence time (s)",
                          "2 s = one quarter of the per-face handling time; below this "
                          "a cycle-time difference is absorbed by normal operator "
                          "variation and has no line-balancing consequence."),
    "median_sync_ms":    ("lower", 50.0, "Inter-hand offset (ms)",
                          "50 ms is the conventional bound for perceived simultaneity "
                          "of two self-generated manual actions; offsets below it are "
                          "not experienced as sequential."),
    "singleton_rate":    ("lower", 0.05, "One-handed press rate",
                          "5 percentage points = roughly one press in twenty; below "
                          "this the two-handed protocol is followed in practice."),
    "accuracy":          ("higher", 0.02, "Accuracy (fraction)",
                          "2 percentage points on a 16-press sequence is under one "
                          "press per two sequences."),
    "rtlx":              ("lower", 10.0, "Raw NASA-TLX (0-100)",
                          "10 points on the 0-100 raw scale is the smallest shift "
                          "routinely treated as practically meaningful in NASA-TLX "
                          "workload comparisons. NOTE: this is a wide band on a "
                          "0-100 scale, so a difference lying inside it is not "
                          "thereby negligible; the band is drawn for interpretation "
                          "and is not tested against."),
    "settle_time_s":     ("lower", 0.30, "Settle time (s)",
                          "0.3 s is below the operator's own reaction-time variability."),
    # Formerly the free-standing SESOI_INTENTION / SESOI_SUS constants. They live
    # here now so that every declared band sits in one table; like the rest, they
    # annotate the figures and are not tested against.
    "intention":         ("higher", 1.0, "Would use again (1-7)",
                          "1 point on the 1-7 intention scale is the smallest step "
                          "the response format can express."),
    "sus":               ("higher", 5.0, "SUS (0-100)",
                          "5 SUS points is roughly half a Bangor grade band."),
}

SUS_BENCHMARK = 68.0                 # Bangor et al. population mean. NOT 67.

# --- Standardised inference policy (applies to EVERY family) ------------------
# The convention of the human-robot augmentation literature, applied identically
# to H1, H2 and H3 so that no endpoint is tested on a different footing from any
# other. This follows the analysis strategy of Huang et al. (2020, IEEE T-MRB,
# doi:10.1109/TMRB.2020.3033137) and Noccaro et al. (2021, Sci Rep 11:9511,
# doi:10.1038/s41598-021-88862-9), both of which report within-subject
# supernumerary-limb studies at comparable n:
#
#   1. Test family        FIXED PER ENDPOINT, BEFORE ANY DATA ARE SEEN, by the
#                         measurement type of the endpoint (see METRIC_FAMILY).
#                         Continuous, unbounded endpoints are parametric; ordinal
#                         endpoints and endpoints sitting on a floor or ceiling
#                         are nonparametric. Nothing observed in the sample can
#                         move an endpoint between families.
#   2. Omnibus            parametric  -> repeated-measures ANOVA, reporting the
#                                        uncorrected F, its two df, p, partial
#                                        eta-squared, and the Greenhouse-Geisser
#                                        epsilon alongside so the reader can judge
#                                        sphericity for themselves. No correction
#                                        is applied by the script.
#                         nonparametric -> Friedman test, reporting chi-square,
#                                        df, p and Kendall's W as the effect size.
#   3. Post-hoc           parametric  -> paired t-test, with t, df, p, Cohen's
#                                        d_z, Hedges' g and the 95% CI of the mean
#                                        difference.
#                         nonparametric -> Wilcoxon signed-rank, with W, p and the
#                                        matched-pairs rank-biserial correlation.
#                         Bonferroni-corrected within the hypothesis family; the
#                         family and its size are stated in every table note.
#   4. Null results       reported as NON-SIGNIFICANT, and nothing more. There is
#                         no equivalence testing in the primary analysis, so the
#                         script never concludes that two conditions are the same.
#                         A non-significant contrast at this n is uninformative
#                         about the absence of an effect, and the SESOI values
#                         retained in METRIC_SPEC are used only to annotate the
#                         figures with a band of practical relevance -- they do
#                         not drive a test and support no equivalence claim.
#   5. Assumptions        skew, kurtosis and Shapiro-Wilk are reported
#                         DESCRIPTIVELY in the appendix and never switch a test.
#                         Test choice is fixed by rule 1, so these diagnostics
#                         document the data rather than steer the analysis;
#                         conditional pre-testing inflates Type I error, and
#                         Shapiro-Wilk has almost no power at n = 10 anyway.
#
# The exploratory correlation families (system validation, H1 mechanism) are NOT
# governed by this policy and are unchanged: permutation is the standard test for
# a correlation at small n, and those families keep their Holm correction.
PEN_CONVERGE_MM = 15.0               # line-to-line gap below this = pens on a point
PEN_PARALLEL_DEG = 20.0              # separation below this = intersection ill-defined
# Sensitivity sweep for the on-point criterion. 8.4 mm is the OBSERVED MEDIAN gap,
# so ~half the trials fall below it by construction: it is a reference row, never an
# independent criterion the data "passed". See pen_criterion_sweep().
PEN_CONVERGE_SWEEP_MM = [5.0, 8.4, 10.0, 15.0, 20.0, 25.0, 30.0]
PEN_SWEEP_OBSERVED_MED = 8.4         # flagged in every sweep output
PEN_SWEEP_MIN_SD = 0.01              # frac_converged SD below this = degenerate criterion
# The sweep repeats the published estimators at 7 criteria x 4 outcomes. It is a
# robustness display, not a headline test, so it uses a smaller resampling budget;
# the estimators themselves are identical to the published ones.
PEN_SWEEP_N_PERM = 4000
PEN_SWEEP_N_BOOT = 1000
EXTREME_COMPLETION_S = 80.0          # flagged, NEVER dropped (R1)
N_PERM = 20000
N_BOOT = 5000
RNG_SEED = 0

TLX_SUBS = ["Mental Demand", "Physical Demand", "Temporal Demand",
            "Performance", "Effort", "Frustration"]

# --- Endpoint -> test family (policy rule 1) ---------------------------------
# Fixed by MEASUREMENT TYPE, not by anything observed in the sample.
#   parametric    RM ANOVA  + paired t post-hoc      continuous, unbounded
#   nonparametric Friedman  + Wilcoxon post-hoc      ordinal, or floor/ceiling
# accuracy sits at a ceiling (~0.99 on a 16-press sequence) and singleton_rate at
# a floor near 0, so both are nonparametric despite being numeric. The two H3
# scales are ordinal by construction.
METRIC_FAMILY = {
    "completion_time_s":  "parametric",
    "settle_time_s":      "parametric",
    "median_sync_ms":     "parametric",
    "rtlx":               "parametric",
    "sus":                "parametric",
    "accuracy":           "nonparametric",
    "singleton_rate":     "nonparametric",
    "intention":          "nonparametric",
    "intention_to_use":   "nonparametric",
    "rank":               "nonparametric",
    "preference_rank":    "nonparametric",
}
# the six raw TLX subscales are scored on the same 0-100 continuous scale as rTLX
METRIC_FAMILY.update({s: "parametric" for s in TLX_SUBS})

FAMILY_TESTS = {
    "parametric":    dict(omnibus="RM ANOVA", posthoc="paired t",
                          effect="d_z", stat="t"),
    "nonparametric": dict(omnibus="Friedman", posthoc="Wilcoxon signed-rank",
                          effect="r_rb", stat="W"),
}


def metric_family(metric):
    """Test family for an endpoint. Unknown endpoints fail loudly rather than
    silently defaulting -- an endpoint with no declared family has not been
    through the policy, and guessing one at runtime is exactly what rule 1
    forbids."""
    fam = METRIC_FAMILY.get(metric)
    if fam is None:
        raise KeyError(
            f"endpoint '{metric}' has no declared test family. Add it to "
            f"METRIC_FAMILY as 'parametric' or 'nonparametric' on the basis of "
            f"its measurement type, not its observed distribution.")
    return fam

PARTICIPANT_COL_RE = re.compile(r"^(P\d+)\.([12])$")
PID_STRICT = re.compile(r"^(?:p|participant)?[_\-\s]?(\d+)$", re.IGNORECASE)
PID_RE = re.compile(r"^P\d+$")

SECTION_DIRS = {
    "V":  "V_system_validation",
    "H1": "H1_performance",
    "H2": "H2_workload",
    "H3": "H3_acceptance",
    "S":  "S_summary",
}

# ---- standardised figure geometry --------------------------------------------------
# Every figure is built from the same panel cell, so a one-panel and a three-panel
# figure share axis height, font size and marker scale on the printed page. Nothing
# below sets its own figsize.
PANEL_W, PANEL_H = 3.5, 3.5          # inches per panel cell
TEX_WIDTH = {1: 0.50, 2: 0.92, 3: 1.00}   # \includegraphics width by panel columns


def fig_size(ncols=1, nrows=1):
    return (PANEL_W * ncols, PANEL_H * nrows)


# ---- what actually goes in the chapter ---------------------------------------------
# CORE  = printed in Chapter 4. Renumbered 4.1..4.n in the order below.
# SUPP  = written to <section>/supplementary/ and numbered S1..Sn. Referenced from the
#         chapter by one cross-reference each, not reproduced in the body.
# The tier is a judgement about the 1000-word budget, not about data quality: the
# supplementary figures are all sound, they just do not each earn 100 words.
FIGURE_SPEC = [
    # key            section  filename                          tier     panels
    ("attenuation",   "V",  "fig_attenuation.png",              "core",  1),
    ("dropout",       "V",  "fig_tracking_dropout.png",         "core",  2),
    ("wander",        "V",  "fig_wander_vs_outcomes.png",       "supp",  2),
    ("pen_technique", "V",  "fig_pen_technique.png",            "supp",  3),
    ("pen_criterion", "V",  "fig_pen_criterion_sweep.png",      "supp",  2),
    ("settle",        "V",  "fig_intersection_vs_settle.png",   "supp",  1),
    ("completion",    "H1", "fig_completion_time.png",          "core",  1),
    ("penalty",       "H1", "fig_time_penalty_mechanism.png",   "core",  2),
    ("offset",        "H1", "fig_interhand_offset.png",         "core",  1),
    ("singleton",     "H1", "fig_singleton_rate.png",           "core",  1),
    ("tlx_endpoint",  "H2", "fig_tlx_endpoint.png",             "core",  1),
    ("tlx_change",    "H2", "fig_tlx_early_endpoint.png",       "supp",  3),
    ("tlx_subscales", "H2", "fig_tlx_subscales.png",            "supp",  3),
    ("preference",    "H3", "fig_preference.png",               "core",  2),
    ("sus",           "H3", "fig_sus.png",                      "core",  1),
]

TABLE_SPEC = {                        # everything not listed is supplementary
    "T_participant_flow":        "core",
    "H1_descriptives":           "core",
    "H1_time_penalty_mechanism": "core",
    "S_hypothesis_summary":      "core",
}

# assign printed numbers: core 4.1.. in listed order, supplementary S1..
_c, _s = 0, 0
FIGURE_INFO = {}
for _key, _sec, _f, _tier, _np_ in FIGURE_SPEC:
    if _tier == "core":
        _c += 1
        _num = f"4.{_c}"
    else:
        _s += 1
        _num = f"S{_s}"
    FIGURE_INFO[_f] = dict(key=_key, section=_sec, tier=_tier, panels=_np_, number=_num)

FIGURE_MAP = {v["number"]: (v["section"], f) for f, v in FIGURE_INFO.items()}

# Column aliases: the upstream scripts use these names. Applied on load so no CSV has
# to be renamed by hand, and so a rename upstream fails loudly here rather than
# silently dropping a figure.
ALIASES = {
    "int_rms_norm":       "wander_norm",        # RMS wander / button spacing
    "pen_angle_mean_deg": "pen_sep_deg",        # mean pen separation angle
    "atten_ratio":        "attenuation_ratio",  # board RMS / intersection RMS
}

RNG = np.random.default_rng(RNG_SEED)


def apply_aliases(df, source):
    """Rename upstream columns to the names this script uses, and report what matched."""
    hit = {k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns}
    if hit:
        df = df.rename(columns=hit)
        print(f"  [alias] {source}: " + ", ".join(f"{k} -> {v}" for k, v in hit.items()))
    return df


# =============================================================================
# FORMATTING
# =============================================================================

def apa_p(p) -> str:
    """APA p-value. Never returns 'p = 0.000' (R9)."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < .001:
        return "$p < .001$"
    if p > .999:
        return "$p > .999$"
    return "$p = %s$" % ("%.3f" % p).lstrip("0")


def plain_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < .001:
        return "p < .001"
    if p > .999:
        return "p > .999"
    return "p = %s" % ("%.3f" % p).lstrip("0")


def fmt_r(r) -> str:
    if r is None or not np.isfinite(r):
        return "n/a"
    return ("%+.2f" % r).replace("-", "\u2212")


def norm_pid(x):
    m = PID_STRICT.match(str(x).strip())
    return int(m.group(1)) if m else np.nan


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.autolayout": False,
    })


def finish(ax, xgrid=False):
    ax.grid(axis="y", alpha=0.25)
    if xgrid:
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.grid(axis="x", visible=False)


def face_list(is_extreme, colour):
    """Open markers for flagged-extreme points, filled otherwise (R1)."""
    return ["none" if bool(e) else colour for e in np.asarray(is_extreme)]


def panel_label(ax, letter):
    ax.text(-0.16, 1.04, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


# =============================================================================
# STATISTICS
# =============================================================================

def bonferroni(pvals):
    """Bonferroni correction. Returns adjusted p in the input order.

    The correction for every hypothesis-family post-hoc test (policy rule 3).
    The family size is the number of finite p values passed in, which is the
    number of pairwise comparisons declared for that endpoint -- it must be
    stated in the table note wherever the corrected p is printed."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    m = int(ok.sum())
    if m == 0:
        return out.tolist()
    out[ok] = np.minimum(1.0, m * p[ok])
    return out.tolist()


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p in the input order.

    RETAINED FOR THE EXPLORATORY CORRELATION FAMILIES ONLY (system validation and
    H1 mechanism), which sit outside the ANOVA/Friedman/Bonferroni policy and are
    unchanged. Every hypothesis-family test uses bonferroni() above."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 0.0
    for k, i in enumerate(order):
        adj = min(1.0, (m - k) * p[i])
        prev = max(prev, adj)
        out[i] = prev
    return out.tolist()


def dz(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else np.nan


def hedges_g_paired(diff):
    """dz with the small-sample correction, for the per-participant trial-level tests."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return np.nan
    g = dz(d)
    return float(g * (1 - 3 / (4 * n - 5)))


def paired_ci(diff, conf=0.95):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return (np.nan, np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + conf / 2, n - 1)
    return (float(d.mean() - t * se), float(d.mean() + t * se))


def rank_biserial_paired(diff):
    """Matched-pairs rank-biserial correlation, the effect size that partners the
    Wilcoxon signed-rank test: (R+ - R-) / (R+ + R-) over the signed ranks of the
    non-zero differences. Ranges -1..+1 and shares the sign of the mean rank."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    r_pos, r_neg = r[d > 0].sum(), r[d < 0].sum()
    tot = r_pos + r_neg
    return float((r_pos - r_neg) / tot) if tot > 0 else np.nan


def bca_ci(x, stat=np.mean, conf=0.95, n_boot=N_BOOT, seed=RNG_SEED):
    """Bias-corrected and accelerated bootstrap CI. Recommended over the t-interval
    for small-sample SUS (Clark et al. 2021), where n < 10 makes the normal-theory
    interval unreliable. Falls back to the percentile interval if the BCa
    acceleration is undefined."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(v, n, replace=True)) for _ in range(n_boot)])
    obs = stat(v)
    prop = float((boots < obs).mean())
    if prop <= 0 or prop >= 1:
        lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
        return (float(lo), float(hi))
    z0 = stats.norm.ppf(prop)
    jack = np.array([stat(np.delete(v, i)) for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    zs = stats.norm.ppf([(1 - conf) / 2, (1 + conf) / 2])
    pcts = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    lo, hi = np.percentile(boots, np.clip(pcts, 0, 1) * 100)
    return (float(lo), float(hi))


def bca_lower_bound(x, conf=0.95, n_boot=N_BOOT, seed=RNG_SEED):
    """One-sided BCa lower bound -- the correct partner for a one-sided test."""
    lo, _ = bca_ci(x, conf=1 - 2 * (1 - conf), n_boot=n_boot, seed=seed)
    return lo


def gg_epsilon(X):
    """Greenhouse-Geisser epsilon for a one-way repeated-measures design.

    Computed from the k x k covariance matrix of the condition columns. REPORTED
    ONLY: the script prints epsilon next to the uncorrected F and p so the reader
    can judge how far the data depart from sphericity, and applies no correction
    of its own (policy rule 2). epsilon = 1 is perfect sphericity; the lower bound
    is 1/(k-1)."""
    X = np.asarray(X, float)
    n, k = X.shape
    if n < 3 or k < 2:
        return np.nan
    S = np.cov(X, rowvar=False, ddof=1)
    S = np.atleast_2d(S)
    s_bar = S.mean()                       # grand mean of all k^2 elements
    s_ii = np.trace(S) / k                 # mean of the diagonal
    s_i = S.mean(axis=1)                   # per-row means
    num = (k ** 2) * (s_ii - s_bar) ** 2
    den = (k - 1) * ((S ** 2).sum() - 2 * k * (s_i ** 2).sum() + (k ** 2) * s_bar ** 2)
    if den <= 0:
        return np.nan
    return float(np.clip(num / den, 1.0 / (k - 1), 1.0))


def rm_anova(wide):
    """One-way repeated-measures ANOVA over CONDITIONS. Policy rule 2, parametric.

    Returns the UNCORRECTED F, df1, df2 and p, partial eta-squared, and the
    Greenhouse-Geisser epsilon alongside them. No sphericity correction is applied
    to the reported p: epsilon travels in the same row so the write-up can discuss
    sphericity without the script having silently adjusted anything.

    Uses pingouin.rm_anova when it is installed and falls back to a direct
    numpy/scipy implementation otherwise, so pingouin is never a hard dependency.
    Which path ran is recorded in ANALYSIS_MANIFEST.csv."""
    X = wide[CONDITIONS].dropna().to_numpy(float)
    n, k = X.shape
    empty = dict(F=np.nan, df1=np.nan, df2=np.nan, p=np.nan, np2=np.nan,
                 gg_eps=np.nan, n=n, test="RM ANOVA", backend=ANOVA_BACKEND)
    if n < 3 or k < 2:
        return empty

    if HAVE_PG:
        try:
            long = pd.DataFrame(X, columns=CONDITIONS)
            long.index.name = "subject"
            long = long.reset_index().melt(id_vars="subject", var_name="condition",
                                           value_name="value")
            aov = pg.rm_anova(data=long, dv="value", within="condition",
                              subject="subject", detailed=True, correction=True)
            row = aov.iloc[0]
            eps = float(row["eps"]) if "eps" in aov.columns else gg_epsilon(X)
            return dict(F=float(row["F"]), df1=float(row["ddof1"]),
                        df2=float(row["ddof2"]), p=float(row["p-unc"]),
                        np2=float(row["np2"]), gg_eps=eps, n=int(n),
                        test="RM ANOVA", backend="pingouin.rm_anova")
        except Exception:
            pass                            # fall through to the internal path

    gm = X.mean()
    sm = X.mean(axis=1, keepdims=True)
    cm = X.mean(axis=0, keepdims=True)
    ss_c = n * ((cm - gm) ** 2).sum()
    ss_e = ((X - sm - cm + gm) ** 2).sum()
    df1, df2 = k - 1, (n - 1) * (k - 1)
    if ss_e <= 0 or df2 <= 0:
        return empty
    F = (ss_c / df1) / (ss_e / df2)
    return dict(F=float(F), df1=float(df1), df2=float(df2),
                p=float(stats.f.sf(F, df1, df2)),
                np2=float(ss_c / (ss_c + ss_e)), gg_eps=gg_epsilon(X), n=int(n),
                test="RM ANOVA", backend="internal (numpy/scipy)")


def friedman_test(wide):
    """Friedman test over CONDITIONS. Policy rule 2, nonparametric.

    Returns chi-square, df, p and Kendall's W = chi2 / (n (k - 1)) as the effect
    size. W runs 0..1 and is the rank-based analogue of partial eta-squared."""
    X = wide[CONDITIONS].dropna()
    n, k = X.shape
    if n < 3 or k < 2:
        return dict(chi2=np.nan, df=np.nan, p=np.nan, kendalls_w=np.nan, n=n,
                    test="Friedman")
    try:
        chi2, p = stats.friedmanchisquare(*[X[c].to_numpy(float) for c in CONDITIONS])
    except Exception:
        return dict(chi2=np.nan, df=k - 1, p=np.nan, kendalls_w=np.nan, n=int(n),
                    test="Friedman")
    return dict(chi2=float(chi2), df=float(k - 1), p=float(p),
                kendalls_w=float(chi2 / (n * (k - 1))), n=int(n), test="Friedman")


def omnibus(wide, metric):
    """Dispatch the omnibus test on the endpoint's DECLARED family (rule 1)."""
    return rm_anova(wide) if metric_family(metric) == "parametric" \
        else friedman_test(wide)


def planned_contrast(wide, a, b, metric):
    """Post-hoc paired contrast b - a, dispatched on the endpoint's declared family.

    parametric     paired t-test:      t, df, p, Cohen's d_z, Hedges' g, 95% CI of
                                       the mean difference
    nonparametric  Wilcoxon signed-rank: W, p, matched-pairs rank-biserial r

    The returned dict keeps the column names the downstream tables and captions
    already use. Two of them carry the family-appropriate quantity rather than a
    fixed one, and `test_stat_name` / `effect_name` record which, so no table can
    mislabel what it prints:
        t   -> the t statistic (parametric) or the W statistic (nonparametric)
        dz  -> Cohen's d_z    (parametric) or rank-biserial r (nonparametric)
    Uncorrected p is returned as p_raw; the Bonferroni-corrected value is added by
    the caller, which is what knows the family size."""
    better, sesoi, label, _ = METRIC_SPEC.get(metric, ("lower", np.nan, metric, ""))
    fam = metric_family(metric)
    w = wide[[a, b]].dropna()
    d = (w[b] - w[a]).to_numpy(float)
    n = d.size
    if n < 3:
        return None
    lo, hi = paired_ci(d)
    supported = ((d.mean() < 0) if better == "lower" else (d.mean() > 0))
    row = dict(metric=metric, family_test=fam, contrast=f"{b} vs {a}", n=n,
               mean_a=float(w[a].mean()), mean_b=float(w[b].mean()),
               diff=float(d.mean()), ci_lo=lo, ci_hi=hi, sesoi=sesoi,
               test=FAMILY_TESTS[fam]["posthoc"],
               direction_favours_treatment=bool(supported))

    if fam == "parametric":
        t, p_t = stats.ttest_rel(w[b], w[a])
        row.update(t=float(t), df=float(n - 1), p_raw=float(p_t),
                   dz=dz(d), hedges_g=hedges_g_paired(d),
                   test_stat_name="t", effect_name="d_z")
    else:
        try:
            res = stats.wilcoxon(w[b], w[a])
            W, p_w = float(res.statistic), float(res.pvalue)
        except ValueError:                  # all differences zero
            W, p_w = 0.0, 1.0
        r_rb = rank_biserial_paired(d)
        row.update(t=W, df=np.nan, p_raw=p_w, dz=r_rb, rank_biserial=r_rb,
                   test_stat_name="W", effect_name="r_rb")
    return row


def perm_corr(x, y, method="pearson", n_perm=N_PERM, seed=RNG_SEED):
    """Correlation with a permutation p. Exact enough at n=9 where the t-approximation
    for r is not trustworthy."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan, n
    f = stats.pearsonr if method == "pearson" else stats.spearmanr
    r = float(f(x, y)[0])
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(float(f(x, rng.permutation(y))[0])) >= abs(r) - 1e-15:
            cnt += 1
    return r, float((cnt + 1) / (n_perm + 1)), n


def loo_corr(x, y, labels, method="pearson"):
    """Leave-one-out influence (R4). Returns (r_min, r_max, most_influential_label)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    labels = np.asarray(labels)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, labels = x[m], y[m], labels[m]
    n = x.size
    if n < 5:
        return (np.nan, np.nan, "")
    f = stats.pearsonr if method == "pearson" else stats.spearmanr
    r_all = float(f(x, y)[0])
    rs, shift = [], []
    for i in range(n):
        k = np.ones(n, bool)
        k[i] = False
        ri = float(f(x[k], y[k])[0])
        rs.append(ri)
        shift.append(abs(ri - r_all))
    j = int(np.argmax(shift))
    return (float(min(rs)), float(max(rs)), str(labels[j]))


def within_participant_r(df, x, y, participant="participant"):
    """Pearson r on participant-mean-centred values: the association WITHIN people (R2)."""
    d = df[[participant, x, y]].dropna()
    if d[participant].nunique() < 2 or len(d) < 6:
        return np.nan, np.nan, len(d)
    a = d[x] - d.groupby(participant)[x].transform("mean")
    b = d[y] - d.groupby(participant)[y].transform("mean")
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, len(d)
    # df corrected for the participant means already removed
    n, k = len(d), d[participant].nunique()
    r = float(np.corrcoef(a, b)[0, 1])
    dfree = max(1, n - k - 1)
    t = r * np.sqrt(dfree / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), dfree)), n


def cluster_boot_r(df, x, y, participant="participant", n_boot=N_BOOT, seed=RNG_SEED):
    """Participant-cluster bootstrap CI for the POOLED r (R2). Resamples participants,
    not trials, so the CI respects the real unit of independence."""
    d = df[[participant, x, y]].dropna()
    if d[participant].nunique() < 4:
        return (np.nan, np.nan)
    pids = d[participant].unique()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(pids, size=pids.size, replace=True)
        s = pd.concat([d[d[participant] == p] for p in pick], ignore_index=True)
        if s[x].std() == 0 or s[y].std() == 0:
            continue
        out.append(np.corrcoef(s[x], s[y])[0, 1])
    if len(out) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def random_slope_lrt(trials, metric, level=TREATMENT):
    """Does the size of the C2 effect vary between operators? (R8)
    LRT of random-intercept vs random-intercept-plus-random-slope-on-C2."""
    if not HAVE_SM:
        return None
    d = trials[["participant", "condition", metric]].dropna().copy()
    if d["condition"].nunique() < 2 or d["participant"].nunique() < 4:
        return None
    d["is_lvl"] = (d["condition"] == level).astype(float)
    d = d.rename(columns={metric: "y"})
    try:
        m0 = smf.mixedlm("y ~ C(condition)", d, groups=d["participant"]).fit(reml=False)
        m1 = smf.mixedlm("y ~ C(condition)", d, groups=d["participant"],
                         re_formula="~is_lvl").fit(reml=False)
        chi2 = float(2 * (m1.llf - m0.llf))
        p = float(stats.chi2.sf(max(chi2, 0.0), df=2))
        return dict(metric=metric, level=level, chi2=chi2, df=2, p=p,
                    n_obs=int(len(d)), n_participants=int(d["participant"].nunique()))
    except Exception as e:
        print(f"  [warn] random-slope LRT failed for {metric}: {e}")
        return None


def mixedlm_condition(trials, metric):
    """Trial-level condition effect with a participant random intercept (R2)."""
    if not HAVE_SM:
        return None
    d = trials[["participant", "condition", metric]].dropna().rename(columns={metric: "y"})
    if d["condition"].nunique() < 2 or d["participant"].nunique() < 3:
        return None
    try:
        m = smf.mixedlm("y ~ C(condition)", d, groups=d["participant"]).fit()
        return pd.DataFrame(dict(metric=metric, term=m.params.index,
                                 estimate=m.params.values, se=m.bse.values,
                                 z=m.tvalues.values, p=m.pvalues.values))
    except Exception:
        return None


# =============================================================================
# OUTPUT PLUMBING
# =============================================================================

class Out:
    """Owns the output tree, the figure/caption pairs and the table registry.

    Core outputs go in <section>/. Supplementary outputs go in
    <section>/supplementary/. Core figures are numbered 4.1.., supplementary S1..,
    and the number is stamped into the .tex label so the chapter cannot cite one and
    print the other.
    """

    def __init__(self, root):
        self.root = root
        self.tables = {}
        self.figures = []
        for d in SECTION_DIRS.values():
            os.makedirs(os.path.join(root, d), exist_ok=True)
            os.makedirs(os.path.join(root, d, "supplementary"), exist_ok=True)

    def dir(self, section, tier="core"):
        base = os.path.join(self.root, SECTION_DIRS[section])
        return base if tier == "core" else os.path.join(base, "supplementary")

    def table(self, name, df, section):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return
        self.tables[name] = (df, section, TABLE_SPEC.get(name, "supp"))

    def fig(self, fig, section, fname, caption, label=None):
        """Save PNG + a LaTeX float. Geometry, width and tier all come from
        FIGURE_INFO, so no caller sets its own size (R9)."""
        info = FIGURE_INFO.get(fname)
        if info is None:
            raise KeyError(f"{fname} is not declared in FIGURE_SPEC -- add it, with a "
                           f"tier and a panel count, so it gets a number and a folder.")
        tier, num, panels = info["tier"], info["number"], info["panels"]
        try:
            fig.tight_layout()
        except Exception:
            pass
        path = os.path.join(self.dir(section, tier), fname)
        fig.savefig(path)
        plt.close(fig)
        slug = os.path.splitext(fname)[0]
        width = TEX_WIDTH.get(panels, 1.00)
        sub = "figures/" if tier == "core" else "figures/supplementary/"
        caption = " ".join(str(caption).split())
        with open(os.path.join(self.dir(section, tier), slug + ".tex"), "w",
                  encoding="utf-8") as f:
            f.write("\\begin{figure}[htbp]\n  \\centering\n")
            f.write(f"  \\includegraphics[width={width}\\linewidth]{{{sub}{fname}}}\n")
            f.write("  \\caption{%s}\n" % caption)
            f.write("  \\label{fig:%s}\n" % (label or slug))
            f.write("\\end{figure}\n")
        self.figures.append(dict(number=num, tier=tier, section=section, file=fname,
                                 panels=panels, caption=caption))
        tag = "CORE" if tier == "core" else "supp"
        print(f"  [{tag:>4s} {num:>4s}] {os.path.relpath(path, self.root)}")

    def write(self):
        print("\n=== Writing tables ===")
        for name, (df, section, tier) in sorted(self.tables.items()):
            p = os.path.join(self.dir(section, tier), f"{name}.csv")
            df.to_csv(p, index=False)
            tag = "CORE" if tier == "core" else "supp"
            print(f"  [{tag:>4s}] {os.path.relpath(p, self.root)}  ({len(df)} rows)")
        man = pd.DataFrame(self.figures)
        if not man.empty:
            man = man.sort_values(["tier", "number"], ascending=[True, True])
        man.to_csv(os.path.join(self.root, "FIGURE_MANIFEST.csv"), index=False)
        pd.DataFrame([dict(name=n, section=sec, tier=t)
                      for n, (_, sec, t) in sorted(self.tables.items())]).to_csv(
            os.path.join(self.root, "TABLE_MANIFEST.csv"), index=False)
        self._write_analysis_manifest()
        self._print_checklist(man)

    def _write_analysis_manifest(self):
        """Record which inference machinery actually ran, so a result in the chapter
        can be traced to the code path that produced it. ANOVA_BACKEND in particular
        records whether pingouin or the internal numpy/scipy implementation was used
        for the RM ANOVA; the two agree to floating-point tolerance, but the reader
        should not have to take that on trust."""
        rows = [
            dict(item="rng_seed", value=RNG_SEED),
            dict(item="rm_anova_backend", value=ANOVA_BACKEND),
            dict(item="pingouin_available", value=HAVE_PG),
            dict(item="statsmodels_available", value=HAVE_SM),
            dict(item="omnibus_parametric", value="repeated-measures ANOVA "
                                                  "(uncorrected F; GG epsilon reported)"),
            dict(item="omnibus_nonparametric", value="Friedman (Kendall's W)"),
            dict(item="posthoc_parametric", value="paired t (d_z, Hedges' g, 95% CI)"),
            dict(item="posthoc_nonparametric", value="Wilcoxon signed-rank "
                                                     "(rank-biserial r)"),
            dict(item="multiplicity", value="Bonferroni within hypothesis family"),
            dict(item="multiplicity_exploratory",
                 value="Holm within the validation / mechanism correlation families"),
            dict(item="equivalence_testing", value="none (SESOI is annotative only)"),
            dict(item="test_selection",
                 value="fixed per endpoint by measurement type (METRIC_FAMILY); "
                       "normality diagnostics are descriptive and never switch a test"),
            dict(item="bootstrap_resamples", value=N_BOOT),
        ]
        rows += [dict(item=f"family:{m}", value=f) for m, f in sorted(METRIC_FAMILY.items())]
        pd.DataFrame(rows).to_csv(
            os.path.join(self.root, "ANALYSIS_MANIFEST.csv"), index=False)

    def _print_checklist(self, man):
        """The thing to work from when assembling the chapter."""
        print("\n" + "=" * 68)
        print("PUT THESE IN THE CHAPTER")
        print("=" * 68)
        core = man[man.tier == "core"] if not man.empty else man
        for _, r in core.iterrows():
            print(f"  Figure {r['number']:<5s} {r['section']:<3s} {r['file']}")
        for name, (_, sec, t) in sorted(self.tables.items()):
            if t == "core":
                print(f"  Table  {'':<5s} {sec:<3s} {name}.csv")
        print("\nSUPPLEMENTARY (one cross-reference each, do not reproduce in the body)")
        supp = man[man.tier == "supp"] if not man.empty else man
        for _, r in supp.iterrows():
            print(f"  Figure {r['number']:<5s} {r['section']:<3s} "
                  f"supplementary/{r['file']}")


# =============================================================================
# LOADING  +  PARTICIPANT FLOW
# =============================================================================

def load_trials(path):
    df = pd.read_csv(path)
    df = apply_aliases(df, os.path.basename(path))
    df["participant"] = df["participant"].map(norm_pid)
    df = df[df["participant"].notna()]
    df = df[df["condition"].isin(CONDITIONS)]
    df = df.dropna(subset=["completion_time_s"])
    if "block" in df.columns:
        df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["extreme"] = df["completion_time_s"] > EXTREME_COMPLETION_S   # flag, never drop
    return df.reset_index(drop=True)


def load_singletons(path):
    df = pd.read_csv(path)
    df["participant"] = df["participant"].map(norm_pid)
    df = df[df["participant"].notna() & df["condition"].isin(CONDITIONS)]
    if "singleton_rate" not in df.columns:
        if {"n_singleton", "n_press"} <= set(df.columns):
            df["singleton_rate"] = df["n_singleton"] / df["n_press"].replace(0, np.nan)
    return df.reset_index(drop=True)


def load_telemetry(path):
    df = pd.read_csv(path)
    df = apply_aliases(df, os.path.basename(path))
    df["participant"] = df["participant"].map(norm_pid)
    df = df[df["participant"].notna()]           # rejects participanttry*/participanttest*
    if "attenuation_ratio" not in df.columns and {"board_rms_mm", "int_rms_mm"} <= set(df.columns):
        df["attenuation_ratio"] = df["board_rms_mm"] / df["int_rms_mm"].replace(0, np.nan)
    df["extreme"] = df.get("completion_time_s", pd.Series(np.nan, index=df.index)) \
                      > EXTREME_COMPLETION_S
    return df.reset_index(drop=True)


def participant_flow(trials, singles, telem, tlx, sus, pref, out):
    """Explicit accounting of who contributes to what (fixes the drifting n)."""
    def ids(df, col="participant"):
        return set() if df is None or df.empty else set(map(int, df[col].dropna().unique()))

    sets = {
        "trials (all conditions)": ids(trials),
        "singleton stream": ids(singles) if singles is not None else set(),
        "C2 telemetry": ids(telem) if telem is not None else set(),
        "NASA-TLX": ids(tlx) if tlx is not None else set(),
        "Preference / intention": set(pref or []),
        "SUS": set(sus or []),
    }
    allp = sorted(set().union(*sets.values())) if sets else []
    rows = []
    for name, s in sets.items():
        missing = sorted(set(allp) - s)
        rows.append(dict(analysis=name, n=len(s),
                         participants=",".join(f"P{p}" for p in sorted(s)),
                         missing=",".join(f"P{p}" for p in missing) or "-"))
    flow = pd.DataFrame(rows)
    out.table("T_participant_flow", flow, "S")

    print("\n=== Participant flow ===")
    for _, r in flow.iterrows():
        print(f"  {r['analysis']:<26s} n={r['n']:<3d} missing: {r['missing']}")
    print("  -> state this table in Methods. Every reported n must be traceable to a row here.")

    n_ex = int(trials["extreme"].sum()) if trials is not None else 0
    if n_ex:
        ex = trials[trials.extreme]
        print(f"\n  [extreme] {n_ex} trial(s) with completion time > {EXTREME_COMPLETION_S:.0f} s "
              f"({', '.join('P%d/%s' % (p, c) for p, c in zip(ex.participant, ex.condition))}). "
              f"RETAINED in all analyses; shown as open markers; every validation "
              f"correlation is also reported with them removed as a sensitivity check.")
    return flow


# =============================================================================
# SHARED PLOT PRIMITIVES
# =============================================================================

def wide_of(df, metric, agg="mean"):
    if df is None or metric not in df.columns:
        return None
    w = (df.groupby(["participant", "condition"])[metric].agg(agg)
           .unstack("condition").reindex(columns=CONDITIONS))
    return w.dropna(how="any")


def sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=8)


def box_points(ax, wide, ylabel, contrasts=None, equivalence=None, extreme_ids=None):
    """Box + per-participant points. NO statistics in the title (R9) -- brackets carry
    the corrected contrast result, the caption carries the numbers."""
    data = [wide[c].dropna().to_numpy(float) for c in CONDITIONS]
    bp = ax.boxplot(data, positions=range(len(CONDITIONS)), widths=0.55,
                    showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(PALETTE[c])
        patch.set_alpha(0.18)
        patch.set_edgecolor(PALETTE[c])
    for el in ("medians", "whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color("black")
            ln.set_linewidth(1.0)

    rng = np.random.default_rng(RNG_SEED)
    for i, c in enumerate(CONDITIONS):
        v = wide[c].dropna()
        jit = rng.uniform(-0.11, 0.11, size=len(v))
        for (pid, val), j in zip(v.items(), jit):
            is_ex = extreme_ids is not None and (pid, c) in extreme_ids
            ax.scatter(i + j, val, s=34, marker=MARKER[c],
                       facecolor="none" if is_ex else PALETTE[c],
                       edgecolor=PALETTE[c], linewidth=1.1,
                       alpha=1.0 if is_ex else 0.75, zorder=3)
        ax.scatter(i, v.mean(), marker="D", s=48, facecolor="white",
                   edgecolor="black", linewidth=1.1, zorder=4)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_TITLE[c] for c in CONDITIONS])
    ax.set_ylabel(ylabel)

    lo, hi = ax.get_ylim()
    span = hi - lo
    step = 0.09 * span
    top = max(np.nanmax(wide[CONDITIONS].to_numpy(float)), hi)
    lvl = 0
    for (a, b), text in (contrasts or []):
        sig_bracket(ax, CONDITIONS.index(a), CONDITIONS.index(b),
                    top + step * (0.4 + lvl), 0.028 * span, text)
        lvl += 1
    if equivalence:
        (a, b), text = equivalence
        sig_bracket(ax, CONDITIONS.index(a), CONDITIONS.index(b),
                    top + step * (0.4 + lvl), 0.028 * span, text)
        lvl += 1
    ax.set_ylim(lo, top + step * (1.0 + lvl))
    finish(ax)


def scatter_with_fit(ax, x, y, labels, xlabel, ylabel, extreme=None):
    """Labelled scatter + least-squares line. Stats go in the caption, not the title."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    ex = np.zeros(x.shape, bool) if extreme is None else np.asarray(extreme, bool)
    cmap = plt.get_cmap("tab10")
    for i in np.where(m)[0]:
        ax.scatter(x[i], y[i], s=46, marker="o",
                   facecolor="none" if ex[i] else cmap(i % 10),
                   edgecolor=cmap(i % 10), linewidth=1.2, zorder=3)
        if labels is not None and len(labels) == len(x):
            ax.annotate(str(labels[i]), (x[i], y[i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.5, color="#444")
    if m.sum() >= 3:
        b = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(x[m].min(), x[m].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="black", lw=1.3, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    finish(ax, xgrid=True)


def corr_row(name, unit, x, y, labels, family, df=None, xcol=None, ycol=None,
             extreme_mask=None, n_perm=N_PERM, n_boot=N_BOOT):
    """One fully-reported correlation: Pearson + Spearman with permutation p, LOO range,
    within-participant r and cluster-bootstrap CI where the unit is the trial, and a
    sensitivity re-run with extreme trials removed (R1-R4)."""
    r_p, p_p, n = perm_corr(x, y, "pearson", n_perm=n_perm)
    r_s, p_s, _ = perm_corr(x, y, "spearman", n_perm=n_perm)
    lo_r, hi_r, infl = loo_corr(x, y, labels, "pearson")
    row = dict(family=family, outcome=name, unit=unit, n=n,
               pearson_r=r_p, pearson_p=p_p, spearman_rho=r_s, spearman_p=p_s,
               loo_r_min=lo_r, loo_r_max=hi_r, most_influential=infl)
    if unit == "trial" and df is not None and xcol and ycol:
        rw, pw, nw = within_participant_r(df, xcol, ycol)
        blo, bhi = cluster_boot_r(df, xcol, ycol, n_boot=n_boot)
        row.update(within_participant_r=rw, within_participant_p=pw,
                   within_n=nw, pooled_ci_lo=blo, pooled_ci_hi=bhi)
        if extreme_mask is not None and np.any(extreme_mask):
            d2 = df[~np.asarray(extreme_mask, bool)]
            r2, p2, n2 = perm_corr(d2[xcol], d2[ycol], "pearson", n_perm=n_perm)
            row.update(sens_pearson_r=r2, sens_pearson_p=p2, sens_n=n2)
    return row



# =============================================================================
# LATEX TABLE EMITTER
# =============================================================================
# Every appendix table is written as a booktabs float from the dataframe that backs
# it, so a number in the thesis cannot disagree with the number in the CSV. Each
# family gets a PAIR: table A = descriptives (what the data look like), table B =
# inference (what was tested, with its effect size, CI and Bonferroni-corrected p).

REGISTRY = {"desc": {}, "infer": {}}      # family -> dataframe


def tex_escape(x):
    return (str(x).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


def tex_num(v, d=2, dash="--"):
    """Numbers with a maths minus, so the column aligns in print."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    if isinstance(v, str):
        return tex_escape(v)
    return ("%.*f" % (d, v)).replace("-", "$-$")


def tex_p(v):
    if v is None or not np.isfinite(v):
        return "--"
    if v < .001:
        return "$<$.001"
    if v > .999:
        return "$>$.999"
    return ("%.3f" % v).lstrip("0")


def write_tex_table(df, path, caption, label, spec, note=None, landscape=False):
    """spec = [(column, header, formatter)] where formatter is 'f2','f1','f3','p','s'."""
    fmt = {"f0": lambda v: tex_num(v, 0), "f1": lambda v: tex_num(v, 1),
           "f2": lambda v: tex_num(v, 2), "f3": lambda v: tex_num(v, 3),
           "p": tex_p, "s": lambda v: tex_escape(v) if v is not None else "--"}
    cols = [c for c, _, _ in spec if c in df.columns]
    if not cols:
        return False
    spec = [t for t in spec if t[0] in df.columns]
    align = "".join("l" if f == "s" else "r" for _, _, f in spec)
    env = "sidewaystable" if landscape else "table"
    L = [f"\\begin{{{env}}}[htbp]", "  \\centering", "  \\small",
         "  \\caption{%s}" % caption, "  \\label{tab:%s}" % label,
         "  \\begin{tabular}{%s}" % align, "    \\toprule",
         "    " + " & ".join(h for _, h, _ in spec) + " \\\\", "    \\midrule"]
    for _, r in df.iterrows():
        L.append("    " + " & ".join(fmt[f](r.get(c)) for c, _, f in spec) + " \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}"]
    if note:
        L.append("  \\begin{minipage}{\\linewidth}\\footnotesize\\vspace{2pt}%s"
                 "\\end{minipage}" % note)
    L += [f"\\end{{{env}}}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return True


DESC_SPEC = [("condition", "Condition", "s"), ("n", "$n$", "f0"),
             ("mean", "$M$", "f2"), ("sd", "$SD$", "f2"),
             ("median", "$Mdn$", "f2"), ("q1", "$Q_1$", "f2"), ("q3", "$Q_3$", "f2")]

# The dataframe column `p_holm` is kept so that no downstream code or caption has
# to be rewired, but it now carries a BONFERRONI-corrected p, so the printed header
# says so -- a column headed "Holm" over Bonferroni numbers would misreport the
# analysis. Likewise `t` and `dz` carry the family-appropriate statistic and effect
# size, which is why their headers name both possibilities and every table note
# states which family the rows belong to.
INFER_SPEC = [("contrast", "Contrast", "s"), ("n", "$n$", "f0"),
              ("diff", "$\\Delta$", "f2"), ("ci_lo", "95\\% CI LL", "f2"),
              ("ci_hi", "UL", "f2"), ("dz", "$d_z$ / $r_{rb}$", "f2"),
              ("hedges_g", "$g$", "f2"),
              ("t", "$t$ / $W$", "f2"), ("p_raw", "$p_{\\text{unc}}$", "p"),
              ("p_holm", "$p_{\\text{bonf}}$", "p"),
              ("sesoi", "SESOI", "f2"),
              ("verdict", "Verdict", "s")]


def write_appendix_tables(out):
    """One descriptives table and one inference table per family (R6, R9)."""
    n = 0
    for fam, (df, section, metric, label, unit) in sorted(REGISTRY["desc"].items()):
        ok = write_tex_table(
            df, os.path.join(out.dir(section, "supp"), f"tabA_{label}_descriptives.tex"),
            caption=("Descriptive statistics for %s by condition. %s" % (fam, unit)),
            label=f"{label}-desc", spec=DESC_SPEC)
        n += int(ok)
    for fam, (df, section, label, omni, note) in sorted(REGISTRY["infer"].items()):
        has_omni = np.isfinite(omni.get("F", np.nan)) or \
            np.isfinite(omni.get("chi2", np.nan))
        cap = ("Inferential results for %s. %s $p_{\\text{unc}}$ is the uncorrected "
               "post-hoc $p$ and $p_{\\text{bonf}}$ the Bonferroni-corrected value. "
               "The SESOI column is the pre-declared band of practical relevance and "
               "is reported for interpretation only; it is not tested against." %
               (fam, ("Omnibus %s." % omnibus_text(omni, tex=True))
                if has_omni else ""))
        ok = write_tex_table(
            df, os.path.join(out.dir(section, "supp"), f"tabB_{label}_inference.tex"),
            caption=cap, label=f"{label}-inference", spec=INFER_SPEC, note=note,
            landscape=True)
        n += int(ok)
    print(f"\n=== Appendix LaTeX tables ===\n  wrote {n} .tex tables "
          f"(tabA_* descriptives, tabB_* inference) into the supplementary folders")
    print("  preamble needed: \\usepackage{booktabs} and \\usepackage{rotating} "
          "(tabB_* are sidewaystable floats)")


# =============================================================================
# SECTION V -- SYSTEM VALIDATION  (Figs 4.1-4.5, Table V1)
# =============================================================================

# =============================================================================
# PEN ON-POINT CRITERION SENSITIVITY SWEEP
# =============================================================================
# PEN_CONVERGE_MM is one chosen threshold. Two claims lean on it: that the pens
# were used as intended, and that no validation correlation survives correction.
# Both should hold across a range of plausible criteria, or the dependence must
# be stated. This section recomputes frac_converged from the per-frame gaps at
# each criterion and re-runs the SAME corr_row() used by the published family,
# so every swept row carries the same permutation p, LOO range,
# within-participant r and extreme-trial sensitivity.
#
# Holm is applied WITHIN each criterion, exactly as the published family is
# corrected -- not across the sweep. The sweep is a robustness display of one
# hypothesis, not a new family of hypotheses.

def load_raw_frames(pattern):
    """Per-frame ORIENT logs. Needs gap_mm plus participant/trial keys."""
    files = sorted(glob.glob(pattern, recursive=True))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "gap_mm" not in df.columns:
            continue
        df = apply_aliases(df, os.path.basename(f))
        if "participant" not in df.columns or "trial" not in df.columns:
            continue
        df["participant"] = df["participant"].map(norm_pid)
        frames.append(df[df["participant"].notna()])
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    if "cond" in raw.columns:
        raw = raw[raw["cond"].astype(str).str.upper() == "C2"]
    raw["gap_mm"] = pd.to_numeric(raw["gap_mm"], errors="coerce")
    raw["trial"] = pd.to_numeric(raw["trial"], errors="coerce")
    return raw[np.isfinite(raw["gap_mm"]) & raw["trial"].notna()].reset_index(drop=True)


def _frac_converged_at(raw, criterion_mm):
    """Per-trial fraction of frames with the two pen lines closer than the
    criterion. Index is (participant, trial), matching the telemetry keys."""
    return (raw.assign(_ok=(raw["gap_mm"] < criterion_mm).astype(float))
            .groupby(["participant", "trial"])["_ok"].mean()
            .rename("frac_converged"))


def pen_criterion_sweep(telem, raw, out):
    """Coverage and every validation correlation, re-run at each criterion."""
    if raw is None or raw.empty:
        print("  [skip] criterion sweep needs per-frame logs (--raw); "
              "the published single-criterion results are unaffected.")
        return pd.DataFrame(), pd.DataFrame()
    if "trial" not in telem.columns:
        print("  [skip] criterion sweep needs a trial key in the telemetry.")
        return pd.DataFrame(), pd.DataFrame()

    key = telem.set_index(["participant", "trial"])
    outcomes = [(c, lab) for c, lab in
                (("completion_time_s", "Completion time"),
                 ("settle_time_s", "Settle time"),
                 ("median_sync_ms", "Inter-hand offset"),
                 ("oof_frac", "Pen out-of-view fraction"))
                if c in key.columns and key[c].notna().sum() >= 4]
    if not outcomes:
        print("  [skip] criterion sweep found no outcome column to correlate.")
        return pd.DataFrame(), pd.DataFrame()

    cov_rows, corr_rows = [], []
    for crit in PEN_CONVERGE_SWEEP_MM:
        fc = _frac_converged_at(raw, crit)
        per_part = fc.groupby(level=0).mean()
        lo, hi = bca_ci(per_part.to_numpy())
        is_med = abs(crit - PEN_SWEEP_OBSERVED_MED) < 1e-6
        degenerate = bool(fc.std(ddof=1) < PEN_SWEEP_MIN_SD or fc.nunique() < 3)
        cov_rows.append(dict(
            criterion_mm=crit, frac_converged_mean=float(per_part.mean()),
            frac_converged_sd=float(per_part.std(ddof=1)),
            frac_converged_median=float(per_part.median()),
            ci_lo=lo, ci_hi=hi, n_participants=int(per_part.size),
            n_trials=int(fc.size), is_observed_median=is_med,
            degenerate=degenerate))

        fam_here = []
        for col, lab in outcomes:
            d = (pd.concat([fc, key[col].rename("outcome"),
                            key["extreme"].rename("extreme")], axis=1)
                 .dropna(subset=["frac_converged", "outcome"]).reset_index())
            if degenerate or len(d) < 4 or d["frac_converged"].nunique() < 3:
                fam_here.append(dict(family="criterion_sweep", outcome=lab,
                                     unit="trial", n=len(d), pearson_r=np.nan,
                                     pearson_p=np.nan, spearman_rho=np.nan,
                                     spearman_p=np.nan, loo_r_min=np.nan,
                                     loo_r_max=np.nan, most_influential=""))
                continue
            fam_here.append(corr_row(
                lab, "trial", d["frac_converged"], d["outcome"], d["participant"],
                "criterion_sweep", df=d, xcol="frac_converged", ycol="outcome",
                extreme_mask=d["extreme"].fillna(False).to_numpy(bool),
                n_perm=PEN_SWEEP_N_PERM, n_boot=PEN_SWEEP_N_BOOT))

        t = pd.DataFrame(fam_here)
        # Holm WITHIN this criterion, mirroring the published family.
        t["pearson_p_holm"] = holm(t["pearson_p"].tolist())
        t["spearman_p_holm"] = holm(t["spearman_p"].tolist())
        t.insert(0, "criterion_mm", crit)
        t["frac_converged_mean"] = float(per_part.mean())
        t["is_observed_median"] = is_med
        t["degenerate"] = degenerate
        corr_rows.append(t)

    cov = pd.DataFrame(cov_rows)
    corr = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()
    out.table("V_pen_criterion_coverage", cov, "V")
    out.table("V_pen_criterion_sweep", corr, "V")

    print(f"\n  Pen on-point criterion sweep ({len(PEN_CONVERGE_SWEEP_MM)} criteria, "
          f"{len(outcomes)} outcomes each):")
    print(f"    {PEN_SWEEP_OBSERVED_MED:.1f} mm is the observed median gap and is "
          "flagged as a reference row, not an independent criterion.")
    for _, r in cov.iterrows():
        surv = int((corr[(corr.criterion_mm == r.criterion_mm)]
                    ["pearson_p_holm"] < .05).sum())
        tags = ("  <- published" if abs(r.criterion_mm - PEN_CONVERGE_MM) < 1e-6 else "")
        tags += "  <- observed median" if r.is_observed_median else ""
        tags += "  [degenerate: no variance left]" if r.degenerate else ""
        print(f"    {r.criterion_mm:5.1f} mm  on-point {r.frac_converged_mean*100:5.1f}% "
              f"[{r.ci_lo*100:5.1f}, {r.ci_hi*100:5.1f}]  "
              f"{surv}/{len(outcomes)} survive Holm{tags}")
    surv_all = corr.groupby("criterion_mm")["pearson_p_holm"].apply(
        lambda s: int((s < .05).sum()))
    if surv_all.nunique() == 1:
        print(f"    -> {int(surv_all.iloc[0])}/{len(outcomes)} survive at EVERY "
              "criterion; the conclusion does not depend on the cut.")
    else:
        print(f"    -> survivor count VARIES across criteria "
              f"({surv_all.to_dict()}); the conclusion is criterion-dependent "
              "and must be reported as such.")

    fig, axes = plt.subplots(1, 2, figsize=fig_size(2))
    ax = axes[0]
    ax.plot(cov["criterion_mm"], cov["frac_converged_mean"] * 100, marker="o",
            color=PALETTE["C2"], lw=1.4, zorder=3)
    ax.fill_between(cov["criterion_mm"], cov["ci_lo"] * 100, cov["ci_hi"] * 100,
                    color=PALETTE["C2"], alpha=0.18, zorder=2)
    ax.axvline(PEN_CONVERGE_MM, color="#009E73", ls="--", lw=1.2)
    ax.set_xlabel("on-point criterion (mm)")
    ax.set_ylabel("frames on-point (%)")
    ax.set_ylim(0, 100)
    finish(ax, xgrid=True); panel_label(ax, "a")

    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, (lab, sub) in enumerate(corr.groupby("outcome")):
        sub = sub.sort_values("criterion_mm")
        ax.plot(sub["criterion_mm"], sub["pearson_r"], marker="o", lw=1.3,
                color=cmap(i % 10), label=lab, zorder=3)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(PEN_CONVERGE_MM, color="#009E73", ls="--", lw=1.2)
    ax.set_xlabel("on-point criterion (mm)")
    ax.set_ylabel("Pearson $r$ with outcome")
    ax.legend(fontsize=6, frameon=False)
    finish(ax, xgrid=True); panel_label(ax, "b")
    fig.tight_layout()

    out.fig(fig, "V", "fig_pen_criterion_sweep.png",
            f"Sensitivity of the pen on-point analysis to the criterion "
            f"($n = {telem['participant'].nunique()}$ participants, "
            f"{len(raw)} frames). (a) Fraction of frames with the two pen lines "
            f"closer than the criterion, participant mean with a bootstrap CI; "
            f"the dashed line is the {PEN_CONVERGE_MM:.0f}\\,mm criterion used "
            f"throughout. (b) Pearson correlation between that fraction and each "
            f"outcome, at every criterion. Points at "
            f"{PEN_SWEEP_OBSERVED_MED:.1f}\\,mm sit at the observed median gap, "
            f"so roughly half the trials fall below by construction. $p$ values "
            f"are Holm-corrected within each criterion, not across the sweep, "
            f"because the sweep displays the robustness of one association "
            f"rather than testing a new family.")
    return cov, corr

def section_validation(trials, telem, singles, out, raw=None):
    print("\n=== V  System validation (C2 telemetry) ===")
    if telem is None or telem.empty:
        print("  [skip] no telemetry")
        return
    expected = ["attenuation_ratio", "wander_norm", "oof_frac", "pen_gap_med_mm",
                "pen_sep_deg", "frac_converged", "settle_time_s"]
    missing = [c for c in expected if c not in telem.columns]
    if missing:
        print(f"  [warn] telemetry is missing {missing} -- the figures that need them "
              f"will be skipped. Check the alias list if you expected them.")
    fam = []
    n_trials = len(telem)
    n_part = telem["participant"].nunique()
    n_ex = int(telem["extreme"].sum())

    # ---- Fig 4.1  attenuation ratio ------------------------------------------------
    if "attenuation_ratio" in telem.columns:
        v = telem["attenuation_ratio"].dropna()
        med = float(v.median())
        below = int((v < 1).sum())
        fig, ax = plt.subplots(figsize=fig_size(1))
        ax.boxplot([v.to_numpy()], positions=[0], widths=0.5, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=PALETTE["C2"], alpha=0.18,
                                 edgecolor=PALETTE["C2"]),
                   medianprops=dict(color="black"))
        rng = np.random.default_rng(RNG_SEED)
        ex = telem["extreme"].to_numpy(bool)[:len(v)]
        for i, (val, e) in enumerate(zip(v.to_numpy(), ex)):
            ax.scatter(rng.uniform(-0.16, 0.16), val, s=22, marker="o",
                       facecolor="none" if e else PALETTE["C2"],
                       edgecolor=PALETTE["C2"], linewidth=0.9, alpha=0.7, zorder=3)
        ax.axhline(1.0, color="black", ls="--", lw=1.0)
        ax.set_xticks([0])
        ax.set_xticklabels(["C2"])
        ax.set_ylabel("attenuation ratio\n(board RMS / intersection RMS)")
        finish(ax)
        fig.tight_layout()
        out.fig(fig, "V", "fig_attenuation.png",
                f"Ratio of board-pose RMS to intersection-point RMS for each C2 trial "
                f"($n = {len(v)}$; all trials retained, open markers are the "
                f"{n_ex} trial(s) exceeding {EXTREME_COMPLETION_S:.0f}\\,s). "
                f"Values below the dashed line indicate that the board moved less than "
                f"the operator's intent point; median {med:.2f}, {below} of {len(v)} "
                f"trials below 1. Note that any low-pass admittance law produces a ratio "
                f"below 1 by construction, so this figure establishes the magnitude of "
                f"attenuation, not its existence.")
        out.table("V_attenuation", pd.DataFrame([dict(
            n=len(v), median=med, q1=float(v.quantile(.25)), q3=float(v.quantile(.75)),
            frac_below_1=below / len(v))]), "V")
        print(f"  attenuation: median {med:.2f}, {below}/{len(v)} below 1")

    # ---- participant-level frame for the n<=10 correlations ------------------------
    pp = telem.groupby("participant").mean(numeric_only=True)
    sync_pp = (trials[trials.condition == "C2"].groupby("participant")["median_sync_ms"]
               .median() if "median_sync_ms" in trials.columns else None)
    sing_pp = (singles[singles.condition == "C2"].groupby("participant")["singleton_rate"]
               .mean() if singles is not None and "singleton_rate" in singles.columns
               else None)
    if sync_pp is not None:
        pp = pp.join(sync_pp.rename("median_sync_ms"), how="left")
    if sing_pp is not None:
        pp = pp.join(sing_pp.rename("singleton_rate"), how="left")
    plabels = [f"P{int(i)}" for i in pp.index]

    # ---- Fig 4.2  wander vs coordination -------------------------------------------
    if "wander_norm" in pp.columns:
        panels = [("median_sync_ms", "Bimanual offset (ms) [lower = better]"),
                  ("singleton_rate", "One-handed press rate [lower = better]")]
        panels = [(c, l) for c, l in panels if c in pp.columns]
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=fig_size(len(panels)))
            axes = np.atleast_1d(axes)
            for ax, (col, lab), letter in zip(axes, panels, "ab"):
                scatter_with_fit(ax, pp["wander_norm"], pp[col], plabels,
                                 "RMS wander / button spacing", lab)
                panel_label(ax, letter)
            fig.tight_layout()
            bits = []
            for col, lab in panels:
                row = corr_row(lab, "participant", pp["wander_norm"], pp[col],
                               plabels, "validation")
                fam.append(row)
                bits.append(
                    f"{lab.split(' [')[0]}: $r = {fmt_r(row['pearson_r'])}$, "
                    f"{apa_p(row['pearson_p'])}; $\\rho = {fmt_r(row['spearman_rho'])}$, "
                    f"{apa_p(row['spearman_p'])}; leave-one-out $r$ "
                    f"{row['loo_r_min']:+.2f} to {row['loo_r_max']:+.2f} "
                    f"(most influential {row['most_influential']})")
            out.fig(fig, "V", "fig_wander_vs_outcomes.png",
                    f"Per-participant intersection wander, normalised by button spacing, "
                    f"against (a) median bimanual offset and (b) one-handed press rate in "
                    f"C2 ($n = {len(pp)}$). Each point is one participant; the line is a "
                    f"least-squares fit and lower is better on both $y$ axes. "
                    + ". ".join(bits) +
                    ". These correlations are exploratory, are Holm-corrected within the "
                    "validation family (Table~\\ref{tab:validation-correlations}), and "
                    "the leave-one-out ranges show how much each rests on a single "
                    "participant.")

    # ---- Fig 4.3  tracking dropout --------------------------------------------------
    if "oof_frac" in telem.columns:
        g = telem.groupby("participant")["oof_frac"].mean()
        fig, axes = plt.subplots(1, 2, figsize=fig_size(2))
        ax = axes[0]
        ax.boxplot([g.to_numpy()], positions=[0], widths=0.5, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=PALETTE["C2"], alpha=0.18,
                                 edgecolor=PALETTE["C2"]),
                   medianprops=dict(color="black"))
        cmap = plt.get_cmap("tab10")
        for i, (pid, val) in enumerate(g.items()):
            ax.scatter(0, val, s=46, color=cmap(i % 10), zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.set_xticks([0])
        ax.set_xticklabels(["C2"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("pen out-of-view fraction\n(active-tracking frames)")
        finish(ax)
        panel_label(ax, "a")

        ax = axes[1]
        d = telem[["participant", "oof_frac", "completion_time_s", "extreme"]].dropna()
        for i, (pid, sub) in enumerate(d.groupby("participant")):
            ax.scatter(sub["oof_frac"], sub["completion_time_s"], s=26,
                       facecolor=face_list(sub["extreme"], cmap(i % 10)),
                       edgecolor=[cmap(i % 10)] * len(sub), linewidth=0.9,
                       alpha=0.85, zorder=3)
        if len(d) >= 3:
            b = np.polyfit(d["oof_frac"], d["completion_time_s"], 1)
            xs = np.linspace(d["oof_frac"].min(), d["oof_frac"].max(), 50)
            ax.plot(xs, np.polyval(b, xs), color="black", lw=1.3)
        ax.set_xlabel("pen out-of-view fraction")
        ax.set_ylabel("Completion time (s)")
        finish(ax, xgrid=True)
        panel_label(ax, "b")
        fig.tight_layout()

        row = corr_row("Completion time", "trial", d["oof_frac"], d["completion_time_s"],
                       d["participant"], "validation", df=d, xcol="oof_frac",
                       ycol="completion_time_s", extreme_mask=d["extreme"].to_numpy())
        fam.append(row)
        sens = (f" Excluding the {n_ex} trial(s) above "
                f"{EXTREME_COMPLETION_S:.0f}\\,s as a sensitivity check gives "
                f"$r = {fmt_r(row.get('sens_pearson_r'))}$ "
                f"({apa_p(row.get('sens_pearson_p'))}, $n = {row.get('sens_n')}$)."
                if n_ex else "")
        out.fig(fig, "V", "fig_tracking_dropout.png",
                f"Pen tracking dropout in C2. (a) Mean fraction of active-tracking frames "
                f"with at least one pen out of view, one point per participant "
                f"($n = {len(g)}$). (b) Out-of-view fraction against completion time for "
                f"every C2 trial ($n = {len(d)}$), coloured by participant; open markers "
                f"are the {n_ex} trial(s) above {EXTREME_COMPLETION_S:.0f}\\,s, which are "
                f"retained. Because trials are nested within participants, the pooled "
                f"fit in (b) is descriptive only: pooled $r = {fmt_r(row['pearson_r'])}$ "
                f"(participant-cluster bootstrap 95\\% CI "
                f"[{row.get('pooled_ci_lo', float('nan')):+.2f}, "
                f"{row.get('pooled_ci_hi', float('nan')):+.2f}]), whereas the "
                f"within-participant association is "
                f"$r = {fmt_r(row.get('within_participant_r'))}$ "
                f"({apa_p(row.get('within_participant_p'))}).{sens}")

    # ---- Fig 4.4  pen technique ------------------------------------------------------
    have = [c for c in ("pen_gap_med_mm", "pen_sep_deg", "frac_converged")
            if c in telem.columns]
    if len(have) == 3:
        fig, axes = plt.subplots(1, 3, figsize=fig_size(3))
        rng = np.random.default_rng(RNG_SEED)

        ax = axes[0]
        v = telem["pen_gap_med_mm"].dropna()
        ax.boxplot([v.to_numpy()], positions=[0], widths=0.5, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#CCCCCC", alpha=0.4),
                   medianprops=dict(color="black"))
        ax.scatter(rng.uniform(-0.16, 0.16, len(v)), v, s=20, color=PALETTE["C1"],
                   alpha=0.6, zorder=3)
        ax.axhline(PEN_CONVERGE_MM, color="#009E73", ls="--", lw=1.2)
        ax.set_xticks([0]); ax.set_xticklabels(["C2"])
        ax.set_ylabel("median pen line-to-line gap (mm)")
        finish(ax); panel_label(ax, "a")

        ax = axes[1]
        v = telem["pen_sep_deg"].dropna()
        ax.boxplot([v.to_numpy()], positions=[0], widths=0.5, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#CCCCCC", alpha=0.4),
                   medianprops=dict(color="black"))
        ax.scatter(rng.uniform(-0.16, 0.16, len(v)), v, s=20, color="#CC79A7",
                   alpha=0.6, zorder=3)
        ax.axhline(PEN_PARALLEL_DEG, color="#D55E00", ls="--", lw=1.2)
        ax.set_xticks([0]); ax.set_xticklabels(["C2"])
        ax.set_ylabel("mean pen separation angle (deg)")
        finish(ax); panel_label(ax, "b")

        ax = axes[2]
        d = telem[["participant", "frac_converged", "completion_time_s", "extreme"]].dropna()
        cmap = plt.get_cmap("tab10")
        for i, (pid, sub) in enumerate(d.groupby("participant")):
            ax.scatter(sub["frac_converged"], sub["completion_time_s"], s=24,
                       facecolor=face_list(sub["extreme"], cmap(i % 10)),
                       edgecolor=[cmap(i % 10)] * len(sub), linewidth=0.9,
                       alpha=0.85, zorder=3)
        if len(d) >= 3:
            b = np.polyfit(d["frac_converged"], d["completion_time_s"], 1)
            xs = np.linspace(d["frac_converged"].min(), d["frac_converged"].max(), 50)
            ax.plot(xs, np.polyval(b, xs), color="black", lw=1.3)
        ax.set_xlabel("fraction of frames with pens on-point")
        ax.set_ylabel("Completion time (s)")
        finish(ax, xgrid=True); panel_label(ax, "c")
        fig.tight_layout()

        row = corr_row("Completion time (pen on-point)", "trial",
                       d["frac_converged"], d["completion_time_s"], d["participant"],
                       "validation", df=d, xcol="frac_converged",
                       ycol="completion_time_s", extreme_mask=d["extreme"].to_numpy())
        fam.append(row)
        out.fig(fig, "V", "fig_pen_technique.png",
                f"Pen technique in C2 ($n = {len(telem)}$ trials). (a) Median line-to-line "
                f"gap per trial; the dashed line is the {PEN_CONVERGE_MM:.0f}\\,mm "
                f"on-point criterion, below which the two pens are treated as meeting at "
                f"a single intersection. (b) Mean pen separation angle; below the dashed "
                f"{PEN_PARALLEL_DEG:.0f}$^\\circ$ line the intersection is "
                f"geometrically ill-defined. (c) Fraction of frames on-point against "
                f"completion time, coloured by participant. The association in (c) is "
                f"weak and does not agree across estimators: pooled "
                f"$r = {fmt_r(row['pearson_r'])}$ ({apa_p(row['pearson_p'])}), "
                f"$\\rho = {fmt_r(row['spearman_rho'])}$ ({apa_p(row['spearman_p'])}), "
                f"within-participant $r = {fmt_r(row.get('within_participant_r'))}$ "
                f"({apa_p(row.get('within_participant_p'))}). It is reported as "
                f"exploratory and is Holm-corrected within the validation family.")

    # ---- Pen on-point criterion sensitivity sweep ------------------------------------
    pen_criterion_sweep(telem, raw, out)

    # ---- Fig 4.5  intersection distance vs settle time ------------------------------
    settle_src = None
    if "settle_time_s" in telem.columns:
        settle_src = telem.groupby("participant")["settle_time_s"].mean()
    elif "settle_time_s" in trials.columns:
        settle_src = (trials[trials.condition == "C2"]
                      .groupby("participant")["settle_time_s"].mean())
    if "pen_gap_med_mm" in pp.columns and settle_src is not None:
        st = settle_src
        m = pp[["pen_gap_med_mm"]].join(st.rename("settle_time_s"), how="inner").dropna()
        if len(m) >= 5:
            fig, ax = plt.subplots(figsize=fig_size(1))
            scatter_with_fit(ax, m["pen_gap_med_mm"], m["settle_time_s"],
                             [f"P{int(i)}" for i in m.index],
                             "mean intersection distance (mm)",
                             "Settle time (s)")
            fig.tight_layout()
            row = corr_row("Settle time", "participant", m["pen_gap_med_mm"],
                           m["settle_time_s"], [f"P{int(i)}" for i in m.index],
                           "validation")
            fam.append(row)
            out.fig(fig, "V", "fig_intersection_vs_settle.png",
                    f"Mean intersection distance against settle time, one point per "
                    f"participant ($n = {len(m)}$). A larger gap means the operator "
                    f"converged the pens less well on a single point (the on-point "
                    f"criterion is {PEN_CONVERGE_MM:.0f}\\,mm). "
                    f"$r = {fmt_r(row['pearson_r'])}$ ({apa_p(row['pearson_p'])}); "
                    f"$\\rho = {fmt_r(row['spearman_rho'])}$ "
                    f"({apa_p(row['spearman_p'])}); leave-one-out $r$ "
                    f"{row['loo_r_min']:+.2f} to {row['loo_r_max']:+.2f}, most "
                    f"influential {row['most_influential']}. Pearson and Spearman "
                    f"disagree, so this is reported as a rank-level tendency only.")

    # ---- Table V1  the whole validation family, Holm-corrected ----------------------
    if fam:
        tab = pd.DataFrame(fam)
        tab["pearson_p_holm"] = holm(tab["pearson_p"].tolist())
        tab["spearman_p_holm"] = holm(tab["spearman_p"].tolist())
        out.table("V_validation_correlations", tab, "V")
        write_validation_table_tex(tab, os.path.join(out.dir("V", "supp"),
                                                     "tab_validation_correlations.tex"),
                                   n_trials, n_part, n_ex)
        n_sig = int((tab["pearson_p_holm"] < .05).sum())
        print(f"  validation family: {len(tab)} correlations, "
              f"{n_sig} survive Holm correction")
    return fam


def write_validation_table_tex(tab, path, n_trials, n_part, n_ex):
    """Table 4.1, with the CORRECT caption: pen out-of-VIEW, not out-of-fold."""
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Validation correlations in C2 between operator/tracking measures and "
        "task outcomes. Both Pearson and Spearman are reported for every row because the "
        "two disagree on several; $p$ values are permutation-based and Holm-corrected "
        "within this exploratory family. Trial-level rows additionally give the "
        "within-participant $r$, which is the estimate the text relies on, because trials "
        f"are nested within participants. All {n_trials} C2 trials from {n_part} "
        f"participants are included; the {n_ex} trial(s) exceeding "
        f"{EXTREME_COMPLETION_S:.0f}\\,s are retained and re-run as a sensitivity check.}}",
        "  \\label{tab:validation-correlations}",
        "  \\begin{tabular}{llrrrrrr}",
        "    \\toprule",
        "    & & & \\multicolumn{2}{c}{Pearson} & \\multicolumn{2}{c}{Spearman} & Within \\\\",
        "    \\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "    Outcome & Unit & $n$ & $r$ & $p_{\\text{holm}}$ & $\\rho$ & "
        "$p_{\\text{holm}}$ & $r$ \\\\",
        "    \\midrule",
    ]
    for _, r in tab.iterrows():
        def f(v, d=3):
            return "--" if v is None or not np.isfinite(v) else ("%.*f" % (d, v)).replace("-", "$-$")
        lines.append("    %s & %s & %d & %s & %s & %s & %s & %s \\\\" % (
            str(r["outcome"]).replace("&", "\\&"), r["unit"], int(r["n"]),
            f(r["pearson_r"], 3), f(r["pearson_p_holm"], 3),
            f(r["spearman_rho"], 3), f(r["spearman_p_holm"], 3),
            f(r.get("within_participant_r"), 3)))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# =============================================================================
# SECTION H1 -- PERFORMANCE  (Figs 4.6-4.8, descriptives, contrasts)
# =============================================================================

H1_METRICS = [
    ("completion_time_s", "mean",   "H1 efficiency"),
    ("median_sync_ms",    "median", "H1 coordination"),
    ("singleton_rate",    "mean",   "H1 coordination (secondary)"),
    ("accuracy",          "mean",   "H1 accuracy"),
]


def descriptives(trials, singles, out):
    """The table the chapter currently lacks: M, SD, n per condition per metric."""
    rows = []
    for metric, agg, fam in H1_METRICS:
        src = singles if metric == "singleton_rate" else trials
        w = wide_of(src, metric, agg)
        if w is None or w.empty:
            continue
        for c in CONDITIONS:
            v = w[c].dropna()
            rows.append(dict(family=fam, metric=metric, condition=c, n=len(v),
                             mean=float(v.mean()), sd=float(v.std(ddof=1)),
                             median=float(v.median()),
                             q1=float(v.quantile(.25)), q3=float(v.quantile(.75))))
    tab = pd.DataFrame(rows)
    out.table("H1_descriptives", tab, "H1")
    for metric, agg, fam in H1_METRICS:
        sub = tab[tab.metric == metric]
        if sub.empty:
            continue
        lab = METRIC_SPEC.get(metric, ("", 0, metric, ""))[2]
        unit = ("Values are participant %ss aggregated over that participant's trials."
                % ("median" if agg == "median" else "mean"))
        REGISTRY["desc"][f"{fam} -- {lab}"] = (sub, "H1", metric,
                                               metric.replace("_", "-"), unit)
    return tab


def omnibus_text(omni, tex=False):
    """One-line rendering of either omnibus, for prints and captions."""
    p = plain_p(omni.get("p")) if not tex else apa_p(omni.get("p"))
    if omni.get("test") == "Friedman":
        chi = "$\\chi^2$" if tex else "chi2"
        w = f", Kendall's {'$W$' if tex else 'W'} = {omni.get('kendalls_w', np.nan):.2f}"
        return f"Friedman {chi}({omni.get('df', np.nan):.0f}) = " \
               f"{omni.get('chi2', np.nan):.2f}, {p}{w}"
    eta = "$\\eta_p^2$" if tex else "partial eta2"
    eps = "$\\varepsilon_{\\text{GG}}$" if tex else "GG eps"
    return (f"RM ANOVA {'$F$' if tex else 'F'}({omni.get('df1', np.nan):.0f}, "
            f"{omni.get('df2', np.nan):.0f}) = {omni.get('F', np.nan):.2f}, {p}, "
            f"{eta} = {omni.get('np2', np.nan):.2f}, {eps} = "
            f"{omni.get('gg_eps', np.nan):.2f}")


def family_note(fam_size, fam_label, test_name, effect_name):
    """The sentence that must accompany every Bonferroni-corrected p (rule 3)."""
    eff = {"d_z": "Cohen's $d_z$", "r_rb": "the matched-pairs rank-biserial $r$"}
    return ("Post-hoc tests are %s, Bonferroni-corrected within the %s family of "
            "%d comparison%s; the $d_z$ column carries %s. Non-significant "
            "contrasts are reported as non-significant: no equivalence test is "
            "performed, so no claim is made that the conditions are the same."
            % (test_name, fam_label, fam_size, "" if fam_size == 1 else "s",
               eff.get(effect_name, effect_name)))


def contrast_block(wide, metric, out, section, family):
    """Omnibus + the two post-hoc contrasts (C2 vs C0, C2 vs C1), Bonferroni across
    the two.

    Standardised policy: the test family is fixed by the endpoint (rule 1), so the
    omnibus is an RM ANOVA or a Friedman test and the post-hoc is a paired t or a
    Wilcoxon signed-rank accordingly. Nothing about the observed sample selects a
    test."""
    fam = metric_family(metric)
    omni = omnibus(wide, metric)
    rows = []
    for a in ("C0", "C1"):
        r = planned_contrast(wide, a, TREATMENT, metric)
        if r:
            rows.append(r)
    if not rows:
        return omni, pd.DataFrame()
    tab = pd.DataFrame(rows)
    tab["primary_test"] = FAMILY_TESTS[fam]["posthoc"]
    tab["correction"] = "Bonferroni"
    tab["family_size"] = len(tab)
    tab["p_holm"] = bonferroni(tab["p_raw"].tolist())   # column name kept; Bonferroni
    tab["verdict"] = np.where(
        tab["p_holm"] < .05,
        np.where(tab["direction_favours_treatment"],
                 "differs, favours C2", "differs, OPPOSITE to hypothesis"),
        "no significant difference")
    tab.insert(0, "family", family)
    tab.insert(1, "omnibus_test", omni.get("test"))
    tab.insert(2, "omnibus_stat", omni.get("F", omni.get("chi2")))
    tab.insert(3, "omnibus_p", omni["p"])
    out.table(f"{section}_{metric}_contrasts", tab, section)
    lab = METRIC_SPEC.get(metric, ("", 0, metric, ""))[2]
    note = (family_note(len(tab), family, FAMILY_TESTS[fam]["posthoc"],
                        FAMILY_TESTS[fam]["effect"])
            + " The SESOI shown is annotative only, marking the band of practical "
              "relevance declared for this endpoint; it is not tested against. "
              "SESOI justification: %s"
            % METRIC_SPEC.get(metric, ("", 0, "", "not declared"))[3])
    REGISTRY["infer"][f"{family} -- {lab}"] = (tab, section,
                                              metric.replace("_", "-"), omni, note)
    print(f"\n  {METRIC_SPEC.get(metric, ('', 0, metric, ''))[2]}  "
          f"[{fam}]  ({omnibus_text(omni)}, n={omni['n']})")
    for _, r in tab.iterrows():
        print(f"    {r['contrast']:<10s} diff={r['diff']:+.2f} "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  "
              f"{r['effect_name']}={r['dz']:+.2f}  "
              f"{r['test_stat_name']}={r['t']:+.2f}  "
              f"{plain_p(r['p_holm'])} (Bonferroni across {int(r['family_size'])}, "
              f"{r['primary_test']}) -> {r['verdict']}")
    return omni, tab


def bracket_text(row):
    """Bracket label for the box-and-points figures. Significant contrasts get
    stars; everything else is n.s. -- the script never labels a null 'equivalent'."""
    if row["p_holm"] < .05:
        stars = "***" if row["p_holm"] < .001 else ("**" if row["p_holm"] < .01 else "*")
        return stars + ("" if row["direction_favours_treatment"] else " (opposite)")
    return "n.s."


def section_h1(trials, singles, out):
    print("\n=== H1  Performance ===")
    desc = descriptives(trials, singles, out)
    extreme_ids = set(map(tuple, trials.loc[trials.extreme,
                                            ["participant", "condition"]].to_numpy()))
    summary = []

    # ---- Fig 4.6  completion time ----------------------------------------------------
    w = wide_of(trials, "completion_time_s", "mean")
    if w is not None and not w.empty:
        omni, tab = contrast_block(w, "completion_time_s", out, "H1", "H1 efficiency")
        fig, ax = plt.subplots(figsize=fig_size(1))
        box_points(ax, w, METRIC_SPEC["completion_time_s"][2],
                   contrasts=[((r["contrast"].split()[-1], TREATMENT), bracket_text(r))
                              for _, r in tab.iterrows()],
                   extreme_ids=extreme_ids)
        fig.tight_layout()
        means = ", ".join(f"{c} {w[c].mean():.1f}\\,s" for c in CONDITIONS)
        c1 = tab[tab.contrast == "C2 vs C1"].iloc[0]
        # Stated from the table rather than asserted: under Bonferroni a contrast
        # that cleared alpha under the old policy may no longer do so, and a caption
        # that hardcodes "both significant" would then be false in print.
        sig = tab[tab["p_holm"] < .05]
        against = sig[~sig["direction_favours_treatment"]]
        if sig.empty:
            sig_txt = ("Neither contrast reaches significance after correction, which "
                       "at this $n$ is not evidence that the conditions are equal.")
        elif len(against) == len(sig):
            sig_txt = ("%s significant contrast%s run%s counter to H1."
                       % ("Both" if len(sig) == 2 else "The",
                          "s" if len(sig) > 1 else "", "" if len(sig) > 1 else "s"))
        elif against.empty:
            sig_txt = ("%s significant contrast%s favour%s C2, as H1 predicts."
                       % ("Both" if len(sig) == 2 else "The",
                          "s" if len(sig) > 1 else "", "" if len(sig) > 1 else "s"))
        else:
            sig_txt = ("Of the significant contrasts, %d run counter to H1 and %d "
                       "favour C2." % (len(against), len(sig) - len(against)))
        out.fig(fig, "H1", "fig_completion_time.png",
                f"Completion time for one eight-face sequence, by condition "
                f"($n = {len(w)}$). Points are participant means (open markers are "
                f"participants contributing a trial above "
                f"{EXTREME_COMPLETION_S:.0f}\\,s, all retained); the box shows median and "
                f"IQR and the diamond the group mean. Means: {means}. Brackets show the "
                f"two post-hoc contrasts against C2, Bonferroni-corrected across the "
                f"pair; "
                f"$^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$. The decisive contrast "
                f"C2 vs C1 is $\\Delta = {c1['diff']:+.1f}$\\,s "
                f"95\\% CI [{c1['ci_lo']:+.1f}, {c1['ci_hi']:+.1f}], "
                f"$d_z = {c1['dz']:+.2f}$, {apa_p(c1['p_holm'])}. {sig_txt}")
        summary.append(h1_row("H1 efficiency", "completion_time_s", tab))

        lrt = random_slope_lrt(trials, "completion_time_s")
        if lrt:
            out.table("H1_random_slope_completion", pd.DataFrame([lrt]), "H1")
            print(f"    random-slope LRT (C2 effect varies by operator?): "
                  f"chi2({lrt['df']})={lrt['chi2']:.2f}, {plain_p(lrt['p'])} "
                  f"-> {'heterogeneous' if lrt['p'] < .05 else 'no reliable heterogeneity'}")

        # per-participant trial-level C2 vs C1, Holm across participants
        pp = per_participant_contrast(trials, "completion_time_s")
        if pp is not None:
            out.table("H1_per_participant_completion", pp, "H1")
            k = int((pp["p_holm"] < .05).sum())
            print(f"    per-participant C2>C1: {k}/{len(pp)} individually significant "
                  f"after Bonferroni; g range {pp['hedges_g'].min():.2f}-{pp['hedges_g'].max():.2f}")

    # ---- Fig 4.7  inter-hand offset ---------------------------------------------------
    w = wide_of(trials, "median_sync_ms", "median")
    if w is not None and not w.empty:
        omni, tab = contrast_block(w, "median_sync_ms", out, "H1", "H1 coordination")
        c1 = tab[tab.contrast == "C2 vs C1"].iloc[0]
        fig, ax = plt.subplots(figsize=fig_size(1))
        box_points(ax, w, METRIC_SPEC["median_sync_ms"][2],
                   contrasts=[((r["contrast"].split()[-1], TREATMENT), bracket_text(r))
                              for _, r in tab.iterrows() if r["contrast"] != "C2 vs C1"],
                   equivalence=(("C1", "C2"), bracket_text(c1)))
        fig.tight_layout()
        means = ", ".join(f"{c} {w[c].mean():.1f}\\,ms" for c in CONDITIONS)
        out.fig(fig, "H1", "fig_interhand_offset.png",
                f"Median offset between the two hands engaging a face, by condition "
                f"($n = {len(w)}$). Points are participant medians; the diamond marks the "
                f"group mean. Group means: {means}. Brackets show the post-hoc "
                f"contrasts against C2, Bonferroni-corrected across the pair. "
                f"C2 vs C1 "
                f"$\\Delta = {c1['diff']:+.1f}$\\,ms 95\\% CI "
                f"[{c1['ci_lo']:+.1f}, {c1['ci_hi']:+.1f}], $d_z = {c1['dz']:+.2f}$, "
                f"{apa_p(c1['p_holm'])}"
                + ("" if c1["p_holm"] < .05 else
                   f"; this contrast is not significant, which at $n = {len(w)}$ is "
                   f"not evidence that the two conditions are equivalent. The "
                   f"$\\pm{METRIC_SPEC['median_sync_ms'][1]:.0f}$\\,ms SESOI is "
                   f"reported as a band of practical relevance, not as a test")
                + ".")
        summary.append(h1_row("H1 coordination", "median_sync_ms", tab))

        lrt = random_slope_lrt(trials, "median_sync_ms")
        if lrt:
            out.table("H1_random_slope_sync", pd.DataFrame([lrt]), "H1")
            print(f"    random-slope LRT: chi2({lrt['df']})={lrt['chi2']:.2f}, "
                  f"{plain_p(lrt['p'])}")

    # ---- Fig 4.8  singleton rate ------------------------------------------------------
    if singles is not None and "singleton_rate" in singles.columns:
        w = wide_of(singles, "singleton_rate", "mean")
        if w is not None and not w.empty:
            omni, tab = contrast_block(w, "singleton_rate", out, "H1",
                                       "H1 coordination (secondary)")
            fig, ax = plt.subplots(figsize=fig_size(1))
            box_points(ax, w, "One-handed press rate",
                       contrasts=[((r["contrast"].split()[-1], TREATMENT),
                                   bracket_text(r)) for _, r in tab.iterrows()])
            fig.tight_layout()
            means = ", ".join(f"{c} {w[c].mean():.3f}" for c in CONDITIONS)
            out.fig(fig, "H1", "fig_singleton_rate.png",
                    f"Proportion of press episodes in which the partner hand never "
                    f"engaged, by condition ($n = {len(w)}$). Points are participant "
                    f"means; the diamond marks the group mean. Means: {means}. This "
                    f"endpoint sits on a floor near zero, so it is tested with a "
                    f"Friedman omnibus and Wilcoxon signed-rank post-hoc tests, "
                    f"Bonferroni-corrected across the pair; brackets show those "
                    f"contrasts against C2. Contrasts marked n.s. are non-significant "
                    f"and are not a demonstration of equivalence.")
            summary.append(h1_row("H1 coordination (secondary)", "singleton_rate", tab))

    # ---- accuracy: ceiling-bounded, so Friedman + Wilcoxon (policy rule 1) -----------
    w = wide_of(trials, "accuracy", "mean")
    if w is not None and not w.empty:
        omni, tab = contrast_block(w, "accuracy", out, "H1", "H1 accuracy")
        summary.append(h1_row("H1 accuracy", "accuracy", tab))
        key = tab[tab.contrast == "C2 vs C1"].iloc[0]
        if key["p_holm"] < .05:
            print(f"    accuracy C2 vs C1: differs, {plain_p(key['p_holm'])} "
                  f"(Wilcoxon, Bonferroni)")
        else:
            print(f"    accuracy C2 vs C1: no significant difference "
                  f"({plain_p(key['p_holm'])}, Wilcoxon, Bonferroni). This is not "
                  f"evidence of equal accuracy -- the study is not powered for that "
                  f"claim and no equivalence test is performed.")

    return pd.DataFrame([s for s in summary if s])


def h1_row(family, metric, tab):
    if tab is None or tab.empty:
        return None
    r = tab[tab.contrast == "C2 vs C1"]
    if r.empty:
        return None
    r = r.iloc[0]
    return dict(hypothesis=family, endpoint=metric, key_contrast="C2 vs C1",
                diff=r["diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], dz=r["dz"],
                effect_name=r.get("effect_name", "d_z"),
                test=r.get("test", ""), p_holm=r["p_holm"], sesoi=r["sesoi"],
                verdict=r["verdict"])


def per_participant_contrast(trials, metric, a="C1", b="C2"):
    """Trial-level a-vs-b test inside each participant, Bonferroni-corrected across
    participants. Supports (but does not by itself establish) the 'system, not operator'
    claim -- that needs the random-slope LRT."""
    rows = []
    for pid, g in trials.groupby("participant"):
        x = g.loc[g.condition == a, metric].dropna().to_numpy(float)
        y = g.loc[g.condition == b, metric].dropna().to_numpy(float)
        if x.size < 3 or y.size < 3:
            continue
        t, p = stats.ttest_ind(y, x, equal_var=False)
        sp = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
                     / max(1, x.size + y.size - 2))
        g_eff = (y.mean() - x.mean()) / sp if sp > 0 else np.nan
        g_eff *= (1 - 3 / (4 * (x.size + y.size) - 9))
        rows.append(dict(participant=int(pid), n_a=x.size, n_b=y.size,
                         mean_a=x.mean(), mean_b=y.mean(),
                         ratio=y.mean() / x.mean() if x.mean() else np.nan,
                         hedges_g=g_eff, p_raw=float(p)))
    if not rows:
        return None
    tab = pd.DataFrame(rows)
    tab["correction"] = "Bonferroni"
    tab["family_size"] = len(tab)
    tab["p_holm"] = bonferroni(tab["p_raw"].tolist())
    return tab



# =============================================================================
# H1 (mechanism) -- does tracking dropout EXPLAIN the C2 time penalty?
# =============================================================================
# The validation section reports how often the pens were lost. That is a reliability
# statistic and it is NOT a test of mechanism: correlating out-of-view fraction with
# completion time inside C2 cannot explain a C2-vs-C1 gap, because C1 has no pen
# tracking to fail. The quantity that needs explaining is the PENALTY, C2 minus C1.
# Three tests, each of which can fail in a different way:
#   (1) dose-response across operators: does a worse-tracked operator pay a bigger
#       penalty?
#   (2) within-participant and duration-normalised: does a worse-tracked TRIAL run
#       long, once the exposure artefact (longer trials contain more frames, so more
#       episodes) is removed?
#   (3) targeted outcome: dropouts should cost time during reorientation specifically,
#       so settle time is a sharper test than total completion time. An association
#       that is absent there is hard to attribute to tracking at all.

def section_h1_mechanism(trials, telem, out):
    print("\n=== H1 (mechanism)  Does tracking dropout explain the C2 time penalty? ===")
    if telem is None or telem.empty or "oof_frac" not in telem.columns:
        print("  [skip] no telemetry / no out-of-view column")
        return

    # duration-normalised dropout rate kills the exposure artefact
    t = telem.copy()
    if "pen_oof_events" in t.columns and "completion_time_s" in t.columns:
        t["oof_rate_hz"] = t["pen_oof_events"] / t["completion_time_s"].replace(0, np.nan)

    # ---- (1) per-operator dose-response on the PENALTY ------------------------------
    w = (trials.groupby(["participant", "condition"])["completion_time_s"].mean()
         .unstack("condition"))
    rows = []
    pen = None
    if {"C1", "C2"} <= set(w.columns):
        pen = (w["C2"] - w["C1"]).rename("penalty_s")
        burden = t.groupby("participant").mean(numeric_only=True)
        m = burden.join(pen, how="inner")
        labels = [f"P{int(i)}" for i in m.index]
        for col, lab in [("oof_frac", "Out-of-view fraction"),
                         ("oof_rate_hz", "Dropout episodes s$^{-1}$")]:
            if col not in m.columns:
                continue
            rows.append(dict(test="dose-response across operators",
                             predictor=lab, outcome="C2-C1 penalty (s)",
                             **{k: v for k, v in
                                corr_row(lab, "participant", m[col], m["penalty_s"],
                                         labels, "mechanism").items()
                                if k not in ("family", "outcome", "unit")}))

    # ---- (2)+(3) within-participant, duration-normalised, targeted outcome ----------
    for xcol, xlab in [("oof_frac", "Out-of-view fraction"),
                       ("oof_rate_hz", "Dropout episodes s$^{-1}$")]:
        if xcol not in t.columns:
            continue
        for ycol, ylab in [("completion_time_s", "Completion time (s)"),
                           ("settle_time_s", "Settle time (s)")]:
            if ycol not in t.columns:
                continue
            d = t[["participant", xcol, ycol]].dropna()
            if len(d) < 10:
                continue
            r_w, p_w, n_w = within_participant_r(d, xcol, ycol)
            r_p, p_p, n_p = perm_corr(d[xcol], d[ycol], "pearson")
            blo, bhi = cluster_boot_r(d, xcol, ycol)
            rows.append(dict(test="within participant (trial level)", predictor=xlab,
                             outcome=ylab, n=n_p, pearson_r=r_p, pearson_p=p_p,
                             within_participant_r=r_w, within_participant_p=p_w,
                             pooled_ci_lo=blo, pooled_ci_hi=bhi))
    if not rows:
        return
    tab = pd.DataFrame(rows)
    tab["p_holm"] = holm(tab.get("within_participant_p",
                                 tab["pearson_p"]).fillna(tab["pearson_p"]).tolist())
    out.table("H1_time_penalty_mechanism", tab, "H1")
    mt = tab.copy()
    mt["contrast"] = mt["predictor"].astype(str) + " $\\rightarrow$ " + mt["outcome"].astype(str)
    mt["dz"] = mt.get("within_participant_r", mt.get("pearson_r"))
    mt["dz"] = mt["dz"].fillna(mt["pearson_r"])
    mt["verdict"] = np.where(mt["p_holm"] < .05, "associated",
                             "no association after correction")
    REGISTRY["infer"]["H1 mechanism -- tracking dropout vs time cost"] = (
        mt, "H1", "h1-mechanism", dict(F=np.nan, p=np.nan),
        ("The $d_z$ column carries the correlation $r$: within-participant where the "
         "unit is the trial, Pearson across operators where the unit is the "
         "participant. Holm correction is across all six tests.")) 
    for _, r in tab.iterrows():
        est = r.get("within_participant_r")
        est = r["pearson_r"] if est is None or not np.isfinite(est) else est
        print(f"    {r['test']:<32s} {str(r['predictor'])[:26]:<26s} -> "
              f"{str(r['outcome'])[:22]:<22s} r={est:+.2f} "
              f"{plain_p(r['p_holm'])} (Holm)")
    n_sig = int((tab["p_holm"] < .05).sum())
    print(f"    -> {n_sig}/{len(tab)} tests survive correction"
          + ("" if n_sig else
             ": tracking dropout does not account for the C2 time penalty. Occlusion "
             "is a real reliability problem but the cost is elsewhere -- most likely "
             "the reorientation motion itself."))

    # ---- figure ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=fig_size(2))
    ax = axes[0]
    if pen is not None and "oof_frac" in t.columns:
        burden = t.groupby("participant")["oof_frac"].mean()
        m = pd.concat([burden.rename("oof"), pen], axis=1).dropna()
        scatter_with_fit(ax, m["oof"], m["penalty_s"],
                         [f"P{int(i)}" for i in m.index],
                         "pen out-of-view fraction",
                         "C2 $-$ C1 time penalty (s)")
        ax.axhline(0, color="black", lw=0.8, ls=":")
    panel_label(ax, "a")

    ax = axes[1]
    ycol = "settle_time_s" if "settle_time_s" in t.columns else "completion_time_s"
    xcol = "oof_rate_hz" if "oof_rate_hz" in t.columns else "oof_frac"
    d = t[["participant", xcol, ycol]].dropna()
    if not d.empty:
        cx = d[xcol] - d.groupby("participant")[xcol].transform("mean")
        cy = d[ycol] - d.groupby("participant")[ycol].transform("mean")
        cmap = plt.get_cmap("tab10")
        for i, (pid, idx) in enumerate(d.groupby("participant").groups.items()):
            ax.scatter(cx.loc[idx], cy.loc[idx], s=24, color=cmap(i % 10),
                       alpha=0.8, zorder=3)
        if len(d) >= 3:
            b = np.polyfit(cx, cy, 1)
            xs = np.linspace(cx.min(), cx.max(), 50)
            ax.plot(xs, np.polyval(b, xs), color="black", lw=1.3)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel("dropout rate, centred within participant")
        ax.set_ylabel(f"{'settle' if ycol.startswith('settle') else 'completion'} "
                      f"time, centred (s)")
        finish(ax, xgrid=True)
    panel_label(ax, "b")

    def _get(test, pred, outc, key):
        q = tab[(tab.test.str.startswith(test)) & (tab.predictor.str.contains(pred))]
        if outc:
            q = q[q.outcome.str.contains(outc)]
        return float(q.iloc[0][key]) if len(q) and key in q.columns else float("nan")

    out.fig(fig, "H1", "fig_time_penalty_mechanism.png",
            f"Testing whether pen tracking dropout accounts for the C2 time penalty. "
            f"(a) Each operator's mean out-of-view fraction against that operator's "
            f"own C2 $-$ C1 penalty; if lost tracking drove the cost, worse-tracked "
            f"operators would pay more "
            f"($r = {fmt_r(_get('dose', 'Out-of-view', '', 'pearson_r'))}$, "
            f"{apa_p(_get('dose', 'Out-of-view', '', 'pearson_p'))}). "
            f"(b) Trial-level dropout rate against settle time, both centred within "
            f"participant so that between-operator differences in speed and the "
            f"exposure artefact are removed "
            f"($r = {fmt_r(_get('within', 'episodes', 'Settle', 'within_participant_r'))}$, "
            f"{apa_p(_get('within', 'episodes', 'Settle', 'within_participant_p'))}). "
            f"Raw episode counts correlate with completion time only because longer "
            f"trials contain more frames; duration-normalised and within-participant, "
            f"that relationship does not hold. Occlusion and limited field of view are "
            f"a real reliability problem, but the time cost is not attributable to "
            f"them.")


# =============================================================================
# SECTION H2 -- WORKLOAD  (Figs 4.9-4.11)
# =============================================================================

def extract_tlx(xlsx):
    xl = pd.ExcelFile(xlsx)
    rows = []
    for sheet in xl.sheet_names:
        m = re.match(r"NASA-TLX\s+(\S+)", sheet)
        if not m:
            continue
        cond = m.group(1)
        raw = xl.parse(sheet, header=None)
        hdr = next((i for i in range(len(raw))
                    if any(isinstance(v, str) and PARTICIPANT_COL_RE.match(v)
                           for v in raw.iloc[i])), None)
        if hdr is None:
            continue
        for j, lab in enumerate(raw.iloc[hdr].tolist()):
            mm = PARTICIPANT_COL_RE.match(str(lab)) if isinstance(lab, str) else None
            if not mm:
                continue
            pid = norm_pid(mm.group(1))
            phase = {"1": "early", "2": "endpoint"}[mm.group(2)]
            for i, sub in enumerate(TLX_SUBS):
                v = pd.to_numeric(raw.iloc[hdr + 1 + i, j], errors="coerce")
                if np.isfinite(v):
                    rows.append(dict(participant=pid, condition=cond, phase=phase,
                                     subscale=sub, value=float(v)))
    return pd.DataFrame(rows)


def section_h2(tlx, out):
    print("\n=== H2  Workload ===")
    if tlx is None or tlx.empty:
        print("  [skip] no TLX")
        return pd.DataFrame()
    comp = (tlx.groupby(["participant", "condition", "phase"])["value"].mean()
            .rename("rtlx").reset_index())
    summary = []

    # ---- Fig 4.9  endpoint ------------------------------------------------------------
    end = comp[comp.phase == "endpoint"]
    w = end.pivot_table(index="participant", columns="condition",
                        values="rtlx").reindex(columns=CONDITIONS).dropna()
    if not w.empty:
        drows = [dict(condition=c, n=int(w[c].notna().sum()), mean=float(w[c].mean()),
                      sd=float(w[c].std(ddof=1)), median=float(w[c].median()),
                      q1=float(w[c].quantile(.25)), q3=float(w[c].quantile(.75)))
                 for c in CONDITIONS]
        REGISTRY["desc"]["H2 workload -- Raw NASA-TLX"] = (
            pd.DataFrame(drows), "H2", "rtlx", "rtlx",
            "Raw unweighted NASA-TLX at the end-of-session rating, 0--100.")
        omni, tab = contrast_block(w, "rtlx", out, "H2", "H2 workload")
        c1 = tab[tab.contrast == "C2 vs C1"].iloc[0]
        c0 = tab[tab.contrast == "C2 vs C0"].iloc[0]
        fig, ax = plt.subplots(figsize=fig_size(1))
        box_points(ax, w, "Raw NASA-TLX (0-100)",
                   contrasts=[(("C0", "C2"), bracket_text(c0))],
                   equivalence=(("C1", "C2"), bracket_text(c1)))
        fig.tight_layout()
        means = ", ".join(f"{c} {w[c].mean():.1f}" for c in CONDITIONS)
        out.fig(fig, "H2", "fig_tlx_endpoint.png",
                f"Raw (unweighted) NASA-TLX at the end of the session, by condition "
                f"($n = {len(w)}$). Points are participants; the diamond marks the mean. "
                f"Means: {means}. Post-hoc paired $t$ tests are Bonferroni-corrected "
                f"across the pair. C2 vs C0 $d_z = {c0['dz']:+.2f}$, "
                f"{apa_p(c0['p_holm'])}. C2 vs C1 shows no significant difference "
                f"($d_z = {c1['dz']:+.2f}$, {apa_p(c1['p_holm'])}); at $n = {len(w)}$ "
                f"this does not establish that the two conditions impose the same "
                f"workload, and no equivalence test is performed. The declared "
                f"$\\pm{METRIC_SPEC['rtlx'][1]:.0f}$-point SESOI is reported as a band "
                f"of practical relevance only.")
        summary.append(h1_row("H2 workload", "rtlx", tab))

        # sensitivity: does the C1-C2 null survive alternative scorings?
        rows = []
        for scoring, sub in (("endpoint", comp[comp.phase == "endpoint"]),
                             ("early", comp[comp.phase == "early"]),
                             ("averaged", comp.groupby(["participant", "condition"],
                                                       as_index=False)["rtlx"].mean())):
            ww = sub.pivot_table(index="participant", columns="condition",
                                 values="rtlx").reindex(columns=CONDITIONS).dropna()
            if {"C1", "C2"} <= set(ww.columns) and len(ww) >= 3:
                d = (ww["C2"] - ww["C1"]).to_numpy(float)
                rows.append(dict(scoring=scoring, n=len(ww), diff=float(d.mean()),
                                 dz=dz(d), hedges_g=hedges_g_paired(d),
                                 test="paired t",
                                 p=float(stats.ttest_rel(ww["C2"], ww["C1"])[1])))
        if rows:
            out.table("H2_scoring_sensitivity", pd.DataFrame(rows), "H2")

    # ---- Fig 4.10  early -> endpoint, tested as an INTERACTION (R7) -------------------
    piv = comp.pivot_table(index="participant", columns=["condition", "phase"],
                           values="rtlx")
    have = [c for c in CONDITIONS
            if (c, "early") in piv.columns and (c, "endpoint") in piv.columns]
    if have:
        deltas = pd.DataFrame({c: piv[(c, "endpoint")] - piv[(c, "early")]
                               for c in have}).dropna()
        inter = rm_anova(deltas.reindex(columns=CONDITIONS)) if len(have) == 3 \
            else dict(F=np.nan, df1=np.nan, df2=np.nan, p=np.nan, np2=np.nan,
                      gg_eps=np.nan, n=len(deltas), test="RM ANOVA")
        rows = []
        for c in have:
            d = deltas[c].dropna().to_numpy(float)
            rows.append(dict(condition=c, n=d.size, mean_change=float(d.mean()),
                             dz=dz(d), p_raw=float(stats.ttest_1samp(d, 0)[1])))
        wt = pd.DataFrame(rows)
        wt["correction"] = "Bonferroni"
        wt["family_size"] = len(wt)
        wt["p_holm"] = bonferroni(wt["p_raw"].tolist())
        wt.insert(0, "interaction_F", inter["F"])
        wt.insert(1, "interaction_p", inter["p"])
        out.table("H2_early_to_endpoint", wt, "H2")
        wt2 = wt.rename(columns={"condition": "contrast", "mean_change": "diff"})
        wt2["verdict"] = np.where(wt2["p_holm"] < .05, "workload fell",
                                  "no reliable change")
        REGISTRY["infer"]["H2 workload -- early to endpoint change"] = (
            wt2, "H2", "h2-change", dict(F=inter["F"], p=inter["p"]),
            ("The omnibus is the condition $\\times$ phase interaction on the change "
             "scores, tested by repeated-measures ANOVA; the per-condition rows are "
             "one-sample $t$ tests against zero, Bonferroni-corrected across the "
             "three, and decompose that omnibus rather than standing as evidence on "
             "their own. Positive $\\Delta$ = workload rose."))
        print(f"\n  early->endpoint change: condition x phase interaction "
              f"{omnibus_text(inter)}, n={inter['n']}")
        for _, r in wt.iterrows():
            print(f"    {r['condition']}: {r['mean_change']:+.1f} points, "
                  f"dz={r['dz']:+.2f}, {plain_p(r['p_holm'])} (Bonferroni)")

        fig, axes = plt.subplots(1, len(have), figsize=fig_size(len(have)),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, c, letter in zip(axes, have, "abc"):
            for pid in deltas.index:
                ax.plot([0, 1], [piv.loc[pid, (c, "early")],
                                 piv.loc[pid, (c, "endpoint")]],
                        color=PALETTE[c], alpha=0.35, lw=1.0, marker=MARKER[c], ms=3.5)
            ax.plot([0, 1], [piv.loc[deltas.index, (c, "early")].mean(),
                             piv.loc[deltas.index, (c, "endpoint")].mean()],
                    color="black", lw=2.2, marker="s", ms=5, zorder=5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["early", "endpoint"])
            ax.set_title(c)
            finish(ax)
            panel_label(ax, letter)
        axes[0].set_ylabel("Raw NASA-TLX (0-100)")
        fig.tight_layout()
        sig = ", ".join(f"{r['condition']} {r['mean_change']:+.1f} "
                        f"($d_z = {r['dz']:+.2f}$, {apa_p(r['p_holm'])})"
                        for _, r in wt.iterrows())
        out.fig(fig, "H2", "fig_tlx_early_endpoint.png",
                f"Raw NASA-TLX for each participant at the familiarisation (early) and "
                f"end-of-session (endpoint) ratings, by condition "
                f"($n = {len(deltas)}$ participants with both ratings). Thin lines are "
                f"participants; the heavy line is the group mean. The claim that the "
                f"conditions differ in how much workload falls is tested as a condition "
                f"$\\times$ phase interaction on the change scores "
                f"({omnibus_text(inter, tex=True)}); the per-condition "
                f"changes ({sig}) are Bonferroni-corrected and are reported only as a "
                f"decomposition of that test, not as evidence in their own right.")

    # ---- Fig 4.11  subscales ----------------------------------------------------------
    sub_end = tlx[tlx.phase == "endpoint"]
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=fig_size(3, 2), sharey=True)
    for ax, sub, letter in zip(axes.ravel(), TLX_SUBS, "abcdef"):
        ws = sub_end[sub_end.subscale == sub].pivot_table(
            index="participant", columns="condition", values="value") \
            .reindex(columns=CONDITIONS).dropna()
        if ws.empty:
            ax.set_visible(False)
            continue
        box_points(ax, ws, "")
        ax.set_title(sub)
        panel_label(ax, letter)
        for a in ("C0", "C1"):
            r = planned_contrast(ws, a, "C2", "rtlx")
            if r:
                r["subscale"] = sub
                rows.append(r)
    axes[0, 0].set_ylabel("Rating (0-100)")
    axes[1, 0].set_ylabel("Rating (0-100)")
    fig.tight_layout()
    if rows:
        st = pd.DataFrame(rows)
        # Bonferroni within the subscale family (all 12 contrasts)
        st["correction"] = "Bonferroni"
        st["family_size"] = len(st)
        st["p_holm"] = bonferroni(st["p_raw"].tolist())
        out.table("H2_subscales", st, "H2")
        n_sig = int(((st.contrast == "C2 vs C1") & (st.p_holm < .05)).sum())
        cap_extra = (f"Subscales are scored on the same continuous 0--100 scale as "
                     f"the composite, so each is tested with a paired $t$; after "
                     f"Bonferroni correction across all {len(st)} subscale contrasts, "
                     f"{n_sig} of the six C1-vs-C2 comparisons differ.")
    else:
        cap_extra = ""
    out.fig(fig, "H2", "fig_tlx_subscales.png",
            f"The six raw NASA-TLX subscales at the end-of-session rating, by condition. "
            f"Points are participants; the diamond marks the mean. {cap_extra} "
            f"These subscale tests are secondary to the composite in "
            f"Fig.~\\ref{{fig:fig_tlx_endpoint}} and are reported for completeness.")
    return pd.DataFrame([s for s in summary if s])


# =============================================================================
# SECTION H3 -- ACCEPTANCE  (Figs 4.12-4.13)
# =============================================================================

def _sheet_block(xlsx, sheet, key_re):
    xl = pd.ExcelFile(xlsx)
    if sheet not in xl.sheet_names:
        return None, None
    raw = xl.parse(sheet, header=None)
    hdr = next((i for i in range(len(raw))
                if sum(bool(key_re.match(str(v))) for v in raw.iloc[i]) >= 2), None)
    return (raw, hdr) if hdr is not None else (None, None)


def extract_preference(xlsx):
    raw, hdr = _sheet_block(xlsx, "Preference", PID_RE)
    if raw is None:
        return {}, {}
    cols = {norm_pid(v): j for j, v in enumerate(raw.iloc[hdr]) if PID_RE.match(str(v))}
    ranks, inten = {}, {}
    for i in range(hdr + 1, len(raw)):
        lab = " ".join(str(x) for x in raw.iloc[i, :2].tolist())
        mr = re.search(r"Rank\s*[—\-]\s*(C\d)", lab)
        mi = re.search(r"Would use again\s*[—\-]\s*(C\d)", lab)
        for pid, j in cols.items():
            v = pd.to_numeric(raw.iloc[i, j], errors="coerce")
            if not np.isfinite(v):
                continue
            if mr:
                ranks.setdefault(pid, {})[mr.group(1)] = float(v)
            elif mi:
                inten.setdefault(pid, {})[mi.group(1)] = float(v)
    return ranks, inten


def extract_sus(xlsx):
    raw, hdr = _sheet_block(xlsx, "SUS", PID_RE)
    if raw is None:
        return {}
    cols = {norm_pid(v): j for j, v in enumerate(raw.iloc[hdr]) if PID_RE.match(str(v))}
    items = {}
    for i in range(hdr + 1, len(raw)):
        a = pd.to_numeric(raw.iloc[i, 0], errors="coerce")
        if np.isfinite(a) and 1 <= int(a) <= 10:
            items[int(a)] = i
    scores = {}
    for pid, j in cols.items():
        vals = []
        for q in range(1, 11):
            if q not in items:
                vals = None
                break
            v = pd.to_numeric(raw.iloc[items[q], j], errors="coerce")
            if not np.isfinite(v):
                vals = None
                break
            vals.append(float(v))
        if not vals:
            continue
        s = sum((vals[i] - 1) if (i + 1) % 2 == 1 else (5 - vals[i]) for i in range(10))
        scores[int(pid)] = s * 2.5
    return scores


def section_h3(ranks, inten, sus, out):
    print("\n=== H3  Acceptance ===")
    summary = []

    wr = pd.DataFrame(ranks).T.reindex(columns=CONDITIONS).dropna() if ranks \
        else pd.DataFrame()
    wi = pd.DataFrame(inten).T.reindex(columns=CONDITIONS).dropna() if inten \
        else pd.DataFrame()

    # ---- Fig 4.12  rank + intention ---------------------------------------------------
    if not wr.empty or not wi.empty:
        fig, axes = plt.subplots(1, 2, figsize=fig_size(2))

        if not wr.empty:
            ax = axes[0]
            bottom = np.zeros(len(CONDITIONS))
            rank_cols = {1: "#009E73", 2: "#E69F00", 3: "#B22222"}
            for rk in (1, 2, 3):
                vals = np.array([(wr[c] == rk).sum() for c in CONDITIONS], float)
                ax.bar(range(len(CONDITIONS)), vals, bottom=bottom, width=0.6,
                       color=rank_cols[rk], edgecolor="white",
                       label=f"rank {rk}")
                bottom += vals
            ax.set_xticks(range(len(CONDITIONS)))
            ax.set_xticklabels([COND_TITLE[c] for c in CONDITIONS])
            ax.set_ylabel("participants")
            ax.legend(frameon=False, loc="upper right", fontsize=7)
            finish(ax)
            panel_label(ax, "a")

        if not wi.empty:
            ax = axes[1]
            box_points(ax, wi, "Would use again (1-7)")
            panel_label(ax, "b")
        fig.tight_layout()

        bits = []
        for name, w, better in (("rank", wr, "lower"), ("intention", wi, "higher")):
            if w.empty:
                continue
            # Policy rule 1: both H3 scales are ordinal by construction (a 1-3
            # forced-choice rank and a 1-7 Likert item), so both are in the
            # nonparametric family -- Friedman omnibus, Wilcoxon signed-rank
            # post-hoc, Bonferroni across the three pairs.
            om = friedman_test(w)
            chi2, p, kw = om["chi2"], om["p"], om["kendalls_w"]
            sesoi_m = METRIC_SPEC.get(name, ("", np.nan, "", ""))[1]
            rows = []
            for a, b in [("C0", "C1"), ("C0", "C2"), ("C1", "C2")]:
                d = (w[a] - w[b])
                dvec = (w[b] - w[a]).to_numpy(float)
                try:
                    res = stats.wilcoxon(w[b], w[a])
                    W, pv = float(res.statistic), float(res.pvalue)
                except ValueError:
                    W, pv = 0.0, 1.0
                lo, hi = paired_ci(dvec)
                r_rb = rank_biserial_paired(dvec)
                rows.append(dict(measure=name, pair=f"{a} vs {b}", n=len(d),
                                 median_diff=float(d.median()),
                                 mean_diff=float(dvec.mean()),
                                 ci_lo=lo, ci_hi=hi, dz=r_rb,
                                 rank_biserial=r_rb, effect_name="r_rb",
                                 t=W, test_stat_name="W", p_raw=pv,
                                 sesoi=sesoi_m))
            pr = pd.DataFrame(rows)
            pr["primary_test"] = "Wilcoxon signed-rank"
            pr["correction"] = "Bonferroni"
            pr["family_size"] = len(pr)
            pr["p_holm"] = bonferroni(pr["p_raw"].tolist())
            pr.insert(0, "friedman_chi2", chi2)
            pr.insert(1, "friedman_p", p)
            pr.insert(2, "kendalls_w", kw)
            out.table(f"H3_{name}", pr, "H3")
            drows = [dict(condition=c, n=int(w[c].notna().sum()),
                          mean=float(w[c].mean()), sd=float(w[c].std(ddof=1)),
                          median=float(w[c].median()), q1=float(w[c].quantile(.25)),
                          q3=float(w[c].quantile(.75))) for c in CONDITIONS]
            REGISTRY["desc"][f"H3 acceptance -- {name}"] = (
                pd.DataFrame(drows), "H3", name, f"h3-{name}",
                ("Forced-choice rank, 1 = most preferred." if name == "rank"
                 else "Intention to use again, 1--7."))
            pr2 = pr.rename(columns={"pair": "contrast", "mean_diff": "diff"})
            pr2["verdict"] = np.where(pr2["p_holm"] < .05, "differs",
                                      "no significant difference")
            REGISTRY["infer"][f"H3 acceptance -- {name}"] = (
                pr2, "H3", f"h3-{name}", om,
                (family_note(len(pr), "H3 acceptance", "Wilcoxon signed-rank", "r_rb")
                 + " This endpoint is ordinal, so it is assigned to the "
                   "nonparametric family by measurement type; the omnibus is a "
                   "Friedman $\\chi^2$, not an $F$."))
            key = pr[pr.pair == "C1 vs C2"].iloc[0]
            bits.append(f"{name}: {omnibus_text(om, tex=True)}; C1 vs C2 "
                        f"$r_{{rb}} = {fmt_r(key['rank_biserial'])}$, "
                        f"{apa_p(key['p_holm'])}")
            print(f"  {name} [nonparametric]: {omnibus_text(om)} n={len(w)}; "
                  f"C1 vs C2 W={key['t']:.1f}, r_rb={key['rank_biserial']:+.2f}, "
                  f"{plain_p(key['p_holm'])} (Bonferroni across "
                  f"{int(key['family_size'])}, Wilcoxon) -> "
                  f"{'differs' if key['p_holm'] < .05 else 'no significant difference'}")
            if name == "intention":
                vd = "differs" if key["p_holm"] < .05 else "no significant difference"
                summary.append(dict(hypothesis="H3 acceptance", endpoint="intention_to_use",
                                    key_contrast="C2 vs C1",
                                    diff=float(key["mean_diff"]),
                                    ci_lo=float(key["ci_lo"]), ci_hi=float(key["ci_hi"]),
                                    dz=float(key["dz"]), effect_name="r_rb",
                                    test="Wilcoxon signed-rank",
                                    p_holm=float(key["p_holm"]),
                                    sesoi=float(key["sesoi"]), verdict=vd))

        first = {c: int((wr[c] == wr.min(axis=1)).sum()) for c in CONDITIONS} \
            if not wr.empty else {}
        out.fig(fig, "H3", "fig_preference.png",
                f"(a) Distribution of forced-choice preference ranks by condition "
                f"($n = {len(wr)}$); first choices were "
                f"{', '.join(f'{c} {v}' for c, v in first.items())}. "
                f"(b) Intention to use again on a 1--7 scale ($n = {len(wi)}$); points "
                f"are participants and the diamond marks the mean. "
                + ". ".join(bits) +
                ". Both scales are ordinal, so each is tested with a Friedman omnibus "
                "and Wilcoxon signed-rank pairwise tests, Bonferroni-corrected across "
                "the three pairs within each measure; the effect size is the "
                "matched-pairs rank-biserial correlation. Contrasts reported as "
                "non-significant are not claims of equivalence. Acceptance separates "
                "robot from no robot, not active from passive.")

    # ---- Fig 4.13  SUS ----------------------------------------------------------------
    if sus:
        v = np.array(sorted(sus.values()), float)
        n = v.size
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        t, p2 = stats.ttest_1samp(v, SUS_BENCHMARK)
        p_t = (p2 / 2) if mean > SUS_BENCHMARK else (1 - p2 / 2)
        d = (mean - SUS_BENCHMARK) / sd if sd > 0 else np.nan
        try:
            p_w = float(stats.wilcoxon(v - SUS_BENCHMARK, alternative="greater").pvalue)
        except Exception:
            p_w = np.nan
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
        above = int((v > SUS_BENCHMARK).sum())
        # PRIMARY TEST: the one-sided one-sample t against the benchmark. SUS is in
        # the parametric family (policy rule 1) -- it is a summated 10-item scale,
        # not a single Likert item, so the ordinal objection that puts the H3 scales
        # in the nonparametric family does not apply, and Lewis & Sauro treat SUS
        # means with interval methods. The BCa bootstrap intervals are RETAINED
        # alongside as the descriptive uncertainty statement, because the
        # normal-theory interval is unreliable at n < 10 (Clark et al. 2021); they
        # are not a second test. The Wilcoxon p is kept in the CSV as a secondary
        # check only.
        bca_lo, bca_hi = bca_ci(v)
        lb_one = bca_lower_bound(v)
        lb_one_t = float(mean - stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n))
        above_benchmark = bool(np.isfinite(p_t) and p_t < .05 and mean > SUS_BENCHMARK)
        p_primary = p_t
        if np.isfinite(p_w) and ((p_w < .05) != (p_t < .05)):
            print(f"    [!] SUS: the secondary Wilcoxon disagrees with the primary "
                  f"one-sided t (W {plain_p(p_w)}, t {plain_p(p_t)}); the t is the "
                  f"declared primary test and stands. Report both.")

        fig, ax = plt.subplots(figsize=fig_size(1))
        rng = np.random.default_rng(RNG_SEED)
        ax.scatter(rng.uniform(-0.10, 0.10, n), v, s=52, marker="^",
                   color=PALETTE["C2"], alpha=0.85, zorder=3, label="participants")
        ax.axhline(SUS_BENCHMARK, color="black", ls="--", lw=1.2,
                   label=f"benchmark {SUS_BENCHMARK:.0f}")
        ax.errorbar(0.28, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="s",
                    color="black", ms=7, capsize=4, lw=1.4, label="mean $\\pm$ 95% CI")
        ax.set_xlim(-0.5, 0.62)
        ax.set_xticks([])
        ax.set_ylabel("SUS (0-100)")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        finish(ax)
        fig.tight_layout()
        out.fig(fig, "H3", "fig_sus.png",
                f"SUS scores for the active condition ($n = {n}$). Points are "
                f"participants; the square and bar show the mean and its 95\\% "
                f"confidence interval; the dashed line is the {SUS_BENCHMARK:.0f}-point "
                f"benchmark. Mean {mean:.1f} (SD {sd:.1f}), above the benchmark for "
                f"{above} of {n} participants. The test is a one-sided one-sample $t$ "
                f"against the benchmark: $t({n - 1}) = {t:.2f}$, $d = {d:.2f}$, "
                f"{apa_p(p_t)}"
                + ("" if p_t < .05 else
                   ", which is non-significant and is therefore not evidence that the "
                   "system sits at the benchmark") +
                f". BCa bootstrap intervals ({N_BOOT} resamples) are reported "
                f"alongside as the descriptive uncertainty statement, being more "
                f"reliable than the normal-theory interval at $n < 10$: two-sided "
                f"[{bca_lo:.1f}, {bca_hi:.1f}], one-sided lower bound {lb_one:.1f}. "
                f"A score of {SUS_BENCHMARK:.0f} is the 50th percentile of the "
                f"Sauro--Lewis curved grading scale, so this supports an average to "
                f"slightly above-average rating rather than a good one.")
        out.table("H3_sus", pd.DataFrame([dict(
            n=n, mean=mean, sd=sd, median=float(np.median(v)),
            benchmark=SUS_BENCHMARK, cohen_d=d, t=float(t), df=float(n - 1),
            p_t_onesided=float(p_t),
            primary_test="one-sample t (one-sided)", p_primary=float(p_primary),
            secondary_test="Wilcoxon signed-rank (one-sided)",
            wilcoxon_p_onesided=p_w,
            ci_lo=float(ci[0]), ci_hi=float(ci[1]),
            bca_lo=float(bca_lo), bca_hi=float(bca_hi),
            ci_lo_onesided=lb_one, ci_lo_onesided_t=lb_one_t,
            interval_method="BCa bootstrap (descriptive)",
            above_benchmark=above_benchmark,
            primary_rule="one-sided one-sample t vs benchmark",
            sesoi=METRIC_SPEC["sus"][1],
            n_above_benchmark=above)]), "H3")
        REGISTRY["desc"]["H3 acceptance -- SUS"] = (
            pd.DataFrame([dict(condition="C2 (active)", n=n, mean=mean, sd=sd,
                               median=float(np.median(v)),
                               q1=float(np.percentile(v, 25)),
                               q3=float(np.percentile(v, 75)))]),
            "H3", "sus", "sus", "System Usability Scale, 0--100, collected for C2 only.")
        REGISTRY["infer"]["H3 acceptance -- SUS vs benchmark"] = (
            pd.DataFrame([dict(contrast=f"C2 vs benchmark {SUS_BENCHMARK:.0f}", n=n,
                               diff=mean - SUS_BENCHMARK, ci_lo=float(ci[0]),
                               ci_hi=float(ci[1]), dz=d, t=float(t),
                               p_raw=float(p_t), p_holm=float(p_t),
                               sesoi=METRIC_SPEC["sus"][1],
                               verdict=("above benchmark" if above_benchmark
                                        else "not above benchmark"))]),
            "H3", "h3-sus", dict(F=np.nan, p=np.nan),
            ("A one-sided one-sample $t$ test against the %.0f-point benchmark. This "
             "is a single comparison, so no multiplicity correction applies and "
             "$p_{\\text{unc}}$ and $p_{\\text{bonf}}$ carry the same value. The CI "
             "is on the mean, not on the difference; BCa bootstrap intervals are "
             "reported in H3\\_sus.csv alongside it as the descriptive uncertainty "
             "statement. A non-significant result would mean the sample does not "
             "place the system above the benchmark, not that it sits at it."
             % SUS_BENCHMARK))
        print(f"  SUS [parametric]: n={n} mean={mean:.1f} SD={sd:.1f} "
              f"t({n - 1})={t:.2f} d={d:.2f} {plain_p(p_t)} (one-sided one-sample t) "
              f"-> {'ABOVE benchmark' if above_benchmark else 'not above benchmark'} "
              f"[BCa {bca_lo:.1f}, {bca_hi:.1f}, one-sided lower bound {lb_one:.1f}; "
              f"t-interval {ci[0]:.1f}, {ci[1]:.1f}; secondary Wilcoxon "
              f"{plain_p(p_w)}]")
        vd = ("above benchmark" if above_benchmark
              else "no significant difference from benchmark")
        summary.append(dict(hypothesis="H3 acceptance", endpoint="SUS vs 68",
                            key_contrast="one-sample", diff=mean - SUS_BENCHMARK,
                            ci_lo=float(bca_lo), ci_hi=float(bca_hi), dz=d,
                            effect_name="d", test="one-sample t (one-sided)",
                            p_holm=float(p_primary), sesoi=METRIC_SPEC["sus"][1],
                            verdict=vd))
    return pd.DataFrame([s for s in summary if s])


# =============================================================================
# SUMMARY  --  the hypothesis map the chapter needs
# =============================================================================

def section_assumptions(trials, singles, tlx, out):
    """Distributional diagnostics for every primary contrast.

    DESCRIPTIVE ONLY. These numbers did not influence the choice of test for any
    endpoint: the test family is fixed in METRIC_FAMILY by measurement type before
    any data are seen (policy rule 1), so nothing computed here can move an endpoint
    from the parametric family to the nonparametric one or back. They are reported
    because a reader is entitled to see the shape of the difference scores behind a
    t test, not because the script consulted them. Conditional pre-testing (Shapiro
    then choose the test) inflates Type I error, and Shapiro-Wilk has almost no
    power at n = 10, which is the second reason not to run the analysis that way."""
    print("\n=== A  Distributional diagnostics (descriptive only; no test was "
          "selected from these) ===")
    frames = []
    if trials is not None and not trials.empty:
        for metric, agg in (("completion_time_s", "mean"),
                            ("median_sync_ms", "median"),
                            ("accuracy", "mean")):
            w = wide_of(trials, metric, agg)
            if w is not None and not w.empty:
                frames.append((metric, w))
    if singles is not None and not singles.empty:
        w = wide_of(singles, "singleton_rate", "mean")
        if w is not None and not w.empty:
            frames.append(("singleton_rate", w))
    if tlx is not None and not tlx.empty and "rtlx" in tlx.columns:
        w = wide_of(tlx, "rtlx", "mean")
        if w is not None and not w.empty:
            frames.append(("rtlx", w))

    rows = []
    for metric, w in frames:
        lab = METRIC_SPEC.get(metric, ("", 0, metric, ""))[2]
        for a in ("C0", "C1"):
            if a not in w.columns or TREATMENT not in w.columns:
                continue
            d = (w[TREATMENT] - w[a]).dropna().to_numpy(float)
            if d.size < 3:
                continue
            try:
                sw, sp = stats.shapiro(d)
            except Exception:
                sw, sp = np.nan, np.nan
            p_t = float(stats.ttest_1samp(d, 0).pvalue)
            try:
                p_w = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                p_w = 1.0
            fam = metric_family(metric)
            agree = (p_t < .05) == (p_w < .05)
            rows.append(dict(metric=metric, outcome=lab, contrast=f"{TREATMENT} vs {a}",
                             declared_family=fam,
                             test_used=FAMILY_TESTS[fam]["posthoc"],
                             n=int(d.size), skew=float(stats.skew(d, bias=False)),
                             kurtosis=float(stats.kurtosis(d, bias=False)),
                             shapiro_W=float(sw), shapiro_p=float(sp),
                             p_t=p_t, p_wilcoxon=p_w,
                             tests_agree=bool(agree),
                             note=("" if agree else
                                   "t and Wilcoxon disagree at alpha=.05; the test "
                                   "reported is the one fixed by the declared family, "
                                   "not the one chosen from this comparison")))
    if not rows:
        return
    tab = pd.DataFrame(rows)
    out.table("A_assumptions", tab, "S")
    n_bad = int((tab.shapiro_p < .05).sum())
    n_dis = int((~tab.tests_agree).sum())
    for _, r in tab.iterrows():
        flag = "  <-- NON-NORMAL" if r["shapiro_p"] < .05 else ""
        flag += "  <-- t/W DISAGREE" if not r["tests_agree"] else ""
        print(f"  {r['outcome']:<24}{r['contrast']:<10} [{r['declared_family']:<13}] "
              f"skew {r['skew']:+.2f}  kurt {r['kurtosis']:+.2f}  "
              f"Shapiro W {r['shapiro_W']:.3f} {plain_p(r['shapiro_p'])}  |  "
              f"t {r['p_t']:.4f} W {r['p_wilcoxon']:.4f}{flag}")
    print(f"  -> {n_bad}/{len(tab)} contrasts non-normal; {n_dis}/{len(tab)} where t "
          f"and Wilcoxon disagree. NO TEST WAS SELECTED FROM THESE DIAGNOSTICS: every "
          f"endpoint's family is fixed in METRIC_FAMILY by measurement type.")

    write_tex_table(
        tab[["outcome", "contrast", "declared_family", "n", "skew", "kurtosis",
             "shapiro_W", "shapiro_p", "p_t", "p_wilcoxon"]],
        os.path.join(out.dir("S", "supp"), "tabA_assumptions.tex"),
        caption=("Distributional diagnostics for the paired difference scores behind "
                 "each primary contrast. Skewness and excess kurtosis are "
                 "bias-corrected; $W$ and its $p$ are Shapiro--Wilk. The last two "
                 "columns give the uncorrected $p$ from the paired $t$ and from "
                 "Wilcoxon signed-rank, for comparison. THESE DIAGNOSTICS ARE "
                 "DESCRIPTIVE AND DID NOT INFLUENCE TEST CHOICE: each endpoint is "
                 "assigned to the parametric or nonparametric family in advance by "
                 "its measurement type (the Family column), and no quantity in this "
                 "table can move it. Conditional pre-testing would inflate the "
                 "Type~I error rate, and Shapiro--Wilk has little power at this $n$."),
        label="assumptions",
        spec=[("outcome", "Outcome", "s"), ("contrast", "Contrast", "s"),
              ("declared_family", "Family", "s"),
              ("n", "$n$", "f0"), ("skew", "Skew", "f2"),
              ("kurtosis", "Kurtosis", "f2"), ("shapiro_W", "$W$", "f2"),
              ("shapiro_p", "$p_W$", "p"),
              ("p_t", "$p_t$", "p"), ("p_wilcoxon", "$p_{\\text{Wilc}}$", "p")],
        note=("The test actually reported for each contrast is the one fixed by the "
              "Family column, whatever the diagnostics show."))


def write_summary(parts, out):
    frames = [p for p in parts if p is not None and not p.empty]
    if not frames:
        return
    tab = pd.concat(frames, ignore_index=True)
    out.table("S_hypothesis_summary", tab, "S")

    lines = ["\\begin{table}[htbp]", "  \\centering",
             "  \\caption{Hypothesis map. One row per pre-declared endpoint, with the "
             "decisive C2-vs-C1 contrast, its 95\\% confidence interval, the effect "
             "size and the Bonferroni-corrected $p$ for its hypothesis family. Each "
             "endpoint is tested by RM ANOVA with paired-$t$ post-hoc tests or by "
             "Friedman with Wilcoxon signed-rank post-hoc tests, fixed in advance by "
             "measurement type; the effect column is Cohen's $d_z$ in the first case "
             "and the matched-pairs rank-biserial $r$ in the second. A contrast that "
             "does not reach $\\alpha$ is reported as showing no significant "
             "difference, which at this $n$ is not evidence that the conditions are "
             "equivalent.}",
             "  \\label{tab:hypothesis-summary}",
             "  \\begin{tabular}{llrrrl}", "    \\toprule",
             "    Hypothesis & Endpoint & $\\Delta$ & $d_z$ / $r_{rb}$ & "
             "$p_{\\text{bonf}}$ & Verdict \\\\", "    \\midrule"]
    for _, r in tab.iterrows():
        def f(v, d=2):
            return "--" if v is None or not np.isfinite(v) else ("%.*f" % (d, v)).replace("-", "$-$")
        lines.append("    %s & %s & %s & %s & %s & %s \\\\" % (
            r["hypothesis"], str(r["endpoint"]).replace("_", "\\_"),
            f(r["diff"]), f(r["dz"]), f(r["p_holm"], 3),
            str(r["verdict"]).replace("_", " ")))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(os.path.join(out.dir("S", "core"), "tab_hypothesis_summary.tex"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n=== Hypothesis summary ===")
    for _, r in tab.iterrows():
        print(f"  {r['hypothesis']:<28s} {str(r['endpoint']):<20s} -> {r['verdict']}")


# =============================================================================
# SELF-TEST  --  synthesises inputs so the pipeline can be verified end to end
# =============================================================================

def selftest(tmpdir):
    rng = np.random.default_rng(7)
    pids = list(range(1, 11))
    rows, srows = [], []
    for p in pids:
        skill = rng.normal(0, 2.5)
        for cond, base_t, base_s, base_sing in (("C0", 29.7, 156, 0.17),
                                                ("C1", 23.2, 125, 0.12),
                                                ("C2", 36.1, 118, 0.13)):
            for k in range(9):
                rows.append(dict(
                    participant=p, condition=cond, block=k // 3 + 1, trial=k + 1,
                    completion_time_s=max(8, rng.normal(base_t + skill, 4.0)),
                    median_sync_ms=max(10, rng.normal(base_s + skill * 4, 35)),
                    settle_time_s=max(0.5, rng.normal(2.5, 0.6)),
                    median_rt_s=max(0.2, rng.normal(0.9, 0.15)),
                    accuracy=min(1.0, rng.normal(0.99, 0.012))))
                srows.append(dict(participant=p, condition=cond, trial=k + 1,
                                  singleton_rate=max(0, rng.normal(base_sing, 0.04))))
    trials = pd.DataFrame(rows)
    # two genuine extremes in C2 (the ones previously deleted)
    idx = trials.index[(trials.condition == "C2") & (trials.participant == 3)][:2]
    trials.loc[idx, "completion_time_s"] = [95.0, 112.0]

    tel = []
    for p in pids[1:]:                       # P1 has no telemetry, as in the real study
        for k in range(9):
            int_rms = rng.normal(24, 5)
            tel.append(dict(
                participant=p, trial=k + 1,
                completion_time_s=float(trials[(trials.participant == p) &
                                               (trials.condition == "C2")]
                                        ["completion_time_s"].iloc[k]),
                int_rms_mm=int_rms, board_rms_mm=int_rms * rng.uniform(0.25, 1.05),
                int_rms_norm=rng.uniform(1.0, 1.75),
                oof_frac=np.clip(rng.normal(0.54, 0.09), 0.2, 0.85),
                pen_oof_events=int(rng.normal(14, 4)), orient_frames=int(rng.normal(600, 90)),
                pen_gap_med_mm=abs(rng.normal(9, 3)),
                pen_angle_mean_deg=rng.normal(52, 6),
                settle_time_s=max(0.5, rng.normal(2.5, 0.6)),
                frac_converged=np.clip(rng.normal(0.70, 0.07), 0.4, 0.95)))
    telem = pd.DataFrame(tel)

    tp = os.path.join(tmpdir, "per_trial_metrics.csv")
    sp = os.path.join(tmpdir, "singleton_by_trial.csv")
    lp = os.path.join(tmpdir, "per_c2_trial.csv")
    trials.to_csv(tp, index=False)
    pd.DataFrame(srows).to_csv(sp, index=False)
    telem.to_csv(lp, index=False)

    xp = os.path.join(tmpdir, "SUS_NASA-TLX.xlsx")
    with pd.ExcelWriter(xp, engine="openpyxl") as xw:
        for cond, base in (("C0", 35.5), ("C1", 19.5), ("C2", 19.3)):
            cols, data = [], []
            for p in pids:
                for ph in (1, 2):
                    cols.append(f"P{p}.{ph}")
            for sub in TLX_SUBS:
                data.append([max(0, min(100, rng.normal(base + (6 if ph == 1 else 0), 12)))
                             for p in pids for ph in (1, 2)])
            df = pd.DataFrame(data, columns=cols)
            df.insert(0, "Subscale", TLX_SUBS)
            df.to_excel(xw, sheet_name=f"NASA-TLX {cond}", index=False)

        pref_rows = [["", "Rank — C0"] + [3] * 10,
                     ["", "Rank — C1"] + list(rng.permutation([1] * 5 + [2] * 5)),
                     ["", "Rank — C2"] + [0] * 10,
                     ["", "Would use again — C0"] + list(rng.integers(1, 4, 10)),
                     ["", "Would use again — C1"] + list(rng.integers(4, 8, 10)),
                     ["", "Would use again — C2"] + list(rng.integers(4, 8, 10))]
        for i in range(10):
            pref_rows[2][2 + i] = 3 - pref_rows[1][2 + i]
        hdr = ["", ""] + [f"P{p}" for p in pids]
        pd.DataFrame([hdr] + pref_rows).to_excel(xw, sheet_name="Preference",
                                                 index=False, header=False)

        sus_hdr = ["#"] + [f"P{p}" for p in pids[:7]]      # SUS on 7 participants
        sus_rows = [[q] + list(rng.integers(2, 6, 7)) for q in range(1, 11)]
        pd.DataFrame([sus_hdr] + sus_rows).to_excel(xw, sheet_name="SUS",
                                                    index=False, header=False)
    return tp, sp, lp, xp


# TOST-derived columns that must not survive anywhere in the emitted tables.
TOST_COLUMNS = {"tost_p", "tost_run", "tost_gate", "equivalent",
                "p_perm", "p_perm_floor", "at_resolution_floor"}


def check_policy(out):
    """Assert that the inference policy actually held, end to end.

    Run after every section, on whatever data the pipeline was given. It checks the
    routing rather than the numbers: a synthetic dataset cannot tell us whether an
    effect is real, but it can tell us whether a parametric endpoint went through an
    RM ANOVA and a paired t, whether an ordinal one went through Friedman and
    Wilcoxon, whether the correction applied was Bonferroni over the family, and
    whether any TOST-derived column survived into a table."""
    checks, fails = [], []
    rg = np.random.default_rng(20240517)

    def ok(name, cond, detail=""):
        checks.append(name)
        if not cond:
            fails.append(f"{name}{': ' + detail if detail else ''}")

    def synth(base, step, sd, n=12, subj_sd=5.0):
        """Within-subject fixture: a real per-participant offset plus independent
        residuals, so the residual variance is non-zero and the covariance matrix is
        non-degenerate. Drawing each column from its own freshly seeded generator
        would make the conditions identical up to a constant, which collapses both
        the ANOVA error term and the paired-difference SD."""
        subj = rg.normal(0, subj_sd, (n, 1))
        return pd.DataFrame(
            {c: (base + step * i) + subj[:, 0] + rg.normal(0, sd, n)
             for i, c in enumerate(CONDITIONS)})

    # ---- (1) parametric endpoint routes to RM ANOVA + paired t ----------------
    w = synth(20.0, 3.0, 4.0)
    ok("parametric family declared",
       metric_family("completion_time_s") == "parametric")
    om = omnibus(w, "completion_time_s")
    ok("parametric omnibus is RM ANOVA", om.get("test") == "RM ANOVA")
    ok("RM ANOVA reports F, both df, p, np2 and GG epsilon",
       all(np.isfinite(om.get(k, np.nan))
           for k in ("F", "df1", "df2", "p", "np2", "gg_eps")),
       str({k: om.get(k) for k in ("F", "df1", "df2", "p", "np2", "gg_eps")}))
    pc = planned_contrast(w, "C1", "C2", "completion_time_s")
    ok("parametric post-hoc is a paired t", pc["test"] == "paired t")
    ok("paired t reports t, df, d_z, Hedges' g and a CI",
       all(np.isfinite(pc[k]) for k in ("t", "df", "dz", "hedges_g",
                                        "ci_lo", "ci_hi")))
    ok("paired t p matches scipy",
       np.isclose(pc["p_raw"], stats.ttest_rel(w["C2"], w["C1"]).pvalue))

    # ---- (2) nonparametric endpoint routes to Friedman + Wilcoxon -------------
    wn = synth(0.95, 0.01, 0.02, subj_sd=0.01).clip(0, 1)
    ok("nonparametric family declared", metric_family("accuracy") == "nonparametric")
    omn = omnibus(wn, "accuracy")
    ok("nonparametric omnibus is Friedman", omn.get("test") == "Friedman")
    ok("Friedman reports chi2, df, p and Kendall's W",
       all(np.isfinite(omn.get(k, np.nan))
           for k in ("chi2", "df", "p", "kendalls_w")))
    pn = planned_contrast(wn, "C1", "C2", "accuracy")
    ok("nonparametric post-hoc is Wilcoxon", pn["test"] == "Wilcoxon signed-rank")
    ok("Wilcoxon reports W and rank-biserial r",
       pn["test_stat_name"] == "W" and np.isfinite(pn["rank_biserial"]))
    ok("Wilcoxon p matches scipy",
       np.isclose(pn["p_raw"], stats.wilcoxon(wn["C2"], wn["C1"]).pvalue))

    # ---- (3) normality never selects the test --------------------------------
    skewed = pd.DataFrame({c: np.exp(rg.normal(0, 1.4, 12)) + 10 * i
                           for i, c in enumerate(CONDITIONS)})
    ok("the skewed fixture really is non-normal (so the check has teeth)",
       stats.shapiro((skewed["C2"] - skewed["C1"]).to_numpy()).pvalue < .05)
    ok("a badly non-normal parametric endpoint still routes to RM ANOVA + t",
       omnibus(skewed, "completion_time_s").get("test") == "RM ANOVA"
       and planned_contrast(skewed, "C1", "C2", "completion_time_s")["test"]
       == "paired t")

    # ---- (4) Bonferroni, applied within family -------------------------------
    ok("bonferroni multiplies by the family size",
       np.allclose(bonferroni([0.01, 0.02]), [0.02, 0.04]))
    ok("bonferroni clips at 1", bonferroni([0.6, 0.9]) == [1.0, 1.0])
    fam_tabs = [(n, df) for n, (df, _, _) in out.tables.items()
                if "p_holm" in getattr(df, "columns", []) and "p_raw" in df.columns
                and "family_size" in df.columns]
    for n, df in fam_tabs:
        sub = df[df["p_raw"].notna() & df["p_holm"].notna()]
        ok(f"Bonferroni applied within family in {n}",
           np.allclose(sub["p_holm"],
                       np.minimum(1.0, sub["family_size"] * sub["p_raw"])),
           f"{len(sub)} rows")
    ok("at least one hypothesis-family table was checked", bool(fam_tabs))

    # ---- (5) no TOST-derived column survives anywhere ------------------------
    for n, (df, _, _) in out.tables.items():
        bad = TOST_COLUMNS & set(getattr(df, "columns", []))
        ok(f"no TOST-derived column in {n}", not bad, ", ".join(sorted(bad)))
    spec_cols = {c for c, _, _ in INFER_SPEC} | {c for c, _, _ in DESC_SPEC}
    ok("no TOST-derived column in any table spec", not (TOST_COLUMNS & spec_cols),
       ", ".join(sorted(TOST_COLUMNS & spec_cols)))
    for nm in ("tost_paired", "tost_one_sample", "apply_tost_gate", "sign_flip_p",
               "perm_floor", "rm_anova_perm", "write_sesoi_table"):
        ok(f"{nm} is gone", nm not in globals())

    print("\n=== Self-test: inference policy ===")
    for f in fails:
        print(f"  [FAIL] {f}")
    print(f"  {len(checks) - len(fails)}/{len(checks)} checks passed")
    return not fails


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", default="per_trial_metrics.csv")
    ap.add_argument("--singletons", default="singleton_by_trial.csv")
    ap.add_argument("--telemetry", default="per_c2_trial.csv")
    ap.add_argument("--workbook", default="SUS_NASA-TLX.xlsx")
    ap.add_argument("--raw", default=None,
                    help="glob for the per-frame ORIENT csv logs (needs gap_mm); "
                         "enables the pen on-point criterion sweep")
    ap.add_argument("--outdir", default="results_out")
    ap.add_argument("--selftest", action="store_true",
                    help="synthesise inputs and run end to end (no real data needed)")
    args = ap.parse_args()

    if args.selftest:
        os.makedirs(args.outdir, exist_ok=True)
        tmp = os.path.join(args.outdir, "_selftest_inputs")
        os.makedirs(tmp, exist_ok=True)
        args.trials, args.singletons, args.telemetry, args.workbook = selftest(tmp)
        print(f"[selftest] synthetic inputs written to {tmp}")

    style()
    out = Out(args.outdir)

    trials = load_trials(args.trials) if os.path.exists(args.trials) else None
    singles = load_singletons(args.singletons) if os.path.exists(args.singletons) else None
    telem = load_telemetry(args.telemetry) if os.path.exists(args.telemetry) else None
    raw = load_raw_frames(args.raw) if args.raw else None
    if args.raw and raw is None:
        print(f"[warn] --raw '{args.raw}' matched no csv with a gap_mm column; "
              "the criterion sweep will be skipped.")
    have_wb = os.path.exists(args.workbook)
    tlx = extract_tlx(args.workbook) if have_wb else None
    ranks, inten = extract_preference(args.workbook) if have_wb else ({}, {})
    sus = extract_sus(args.workbook) if have_wb else {}

    if trials is None:
        print(f"[error] '{args.trials}' not found. Nothing to do.")
        return 1

    print(f"[load] {len(trials)} trials, {trials['participant'].nunique()} participants, "
          f"{trials.groupby('condition').size().to_dict()}")

    participant_flow(trials, singles, telem,
                     tlx if tlx is not None and not tlx.empty else None,
                     sus, list(ranks.keys()), out)

    section_validation(trials, telem, singles, out, raw=raw)
    s1 = section_h1(trials, singles, out)
    section_h1_mechanism(trials, telem, out)
    s2 = section_h2(tlx, out) if tlx is not None and not tlx.empty else pd.DataFrame()
    s3 = section_h3(ranks, inten, sus, out)
    section_assumptions(trials, singles, tlx, out)
    write_summary([s1, s2, s3], out)
    write_appendix_tables(out)

    policy_ok = check_policy(out) if args.selftest else True

    out.write()
    print(f"\nDone. {len(out.figures)} figures, {len(out.tables)} tables in "
          f"'{args.outdir}/'. Figure numbering is in FIGURE_MANIFEST.csv; every figure "
          f"has a .tex caption generated from the data it plots.")
    print(f"[stats] RM ANOVA backend: {ANOVA_BACKEND}; correction: Bonferroni within "
          f"hypothesis family; no equivalence testing. See ANALYSIS_MANIFEST.csv.")
    if not HAVE_SM:
        print("[warn] statsmodels missing -> random-slope LRT and mixed models skipped.")
    if not policy_ok:
        print("[error] inference-policy self-test FAILED (see above).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
