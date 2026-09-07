#!/usr/bin/env python3
"""
analyse_buttons.py - per-face reaction time, synchronicity and singleton rate,
and their relationship to inter-face travel distance.

ALIGNED TO srl_analysis.py. The synchronicity extractor below is that file's
`synchronicity()` reproduced exactly - same co-press detection, same
onset-to-onset offset, same other_A_high / B_carried guards, same device clock,
same one-press-per-presented-face indexing. Verified against
P06_T1_C0_b1_seq2: median_sync_ms = 160, sync_valid_frac = 1.000, 6/8 led by B.

It adds two things srl_analysis.py does not compute:
  * a per-FACE breakdown (that pipeline aggregates straight to trial medians)
  * SINGLETONS, which are not in srl_analysis.py at all

SINGLETON DEFINITION
  An A-channel press that never forms a co-press with B - i.e. one button
  pressed and not followed by its partner. srl_analysis.py absorbs these
  silently, because for each face it keeps only the LONGEST co-press run and
  discards the rest. On P06 that hides three failed attempts at face 8.
  Rate = singleton presses / all A presses. Set --singleton-denominator
  completed to use singletons / completed pairs instead.

AGGREGATION
  Matches srl_analysis.py: median within a trial, then MEAN over that
  condition's trials. Not median-of-all-presses, which would weight trials by
  press count and would not reproduce your published numbers.

    python3 analyse_buttons.py --rec recordings --out button_out
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import statsfuns as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="An input array is constant")

# ============================================================================
# CONSTANTS - mirror srl_common.py
# ============================================================================
PAIR_TO_FACE = {1: 3, 2: 4, 3: 5, 4: 6, 5: 2, 6: 1, 7: 7, 8: 8}
FACE_TO_PAIR = {v: k for k, v in PAIR_TO_FACE.items()}
N_FACES = 8
CONDITIONS = ["C0", "C1", "C2"]
SYNC_THRESH_MS = 100.0

COND_LABEL = {"C0": "C0 (hands)", "C1": "C1 (passive)", "C2": "C2 (active)"}
COND_COLOR = {"C0": "#888888", "C1": "#4C78A8", "C2": "#E45756"}

# Face -> cell on the 3x3 grid, row-major:  0 1 2 / 3 4 5 / 6 7 8
# Default: eight faces round the ring, centre unused. Override with --layout.
FACE_CELLS = [0, 1, 2, 3, 5, 6, 7, 8]

# ============================================================================
# INFERENCE POLICY - must match thesis_results.py exactly
# ============================================================================
# The convention of the human-robot augmentation literature: Huang et al.
# (2020, IEEE T-MRB, doi:10.1109/TMRB.2020.3033137) and Noccaro et al. (2021,
# Sci Rep 11:9511, doi:10.1038/s41598-021-88862-9).
#
#   1. Test family    FIXED PER ENDPOINT by measurement type, before any data
#                     are seen (METRIC_FAMILY below). Nothing observed in the
#                     sample can move an endpoint between families, and no
#                     normality diagnostic selects a test.
#   2. Omnibus        parametric    -> RM ANOVA (uncorrected F, both df, p,
#                                      partial eta-squared, GG epsilon reported
#                                      alongside; no correction applied)
#                     nonparametric -> Friedman (chi-square, df, p, Kendall's W)
#   3. Post-hoc       parametric    -> paired t, effect size Cohen's d_z
#                     nonparametric -> Wilcoxon signed-rank, effect size the
#                                      matched-pairs rank-biserial correlation
#                     Bonferroni-corrected within the family, family size stated.
#   4. Nulls          reported as "no significant difference" and nothing more.
#                     No equivalence testing anywhere in this script.
#
# PERMUTATION IS STILL USED, in two places, and this is NOT an inconsistency
# with the above. Kendall's W is tested by permutation because its chi-square
# form is asymptotic in the number of RATERS and unreliable at n = 10 (Legendre
# 2005); the face-pattern and distance correlations are permuted because that is
# the standard test for a correlation at small n. The policy change was about
# how condition effects on an endpoint are tested, not about either of those.
#
# The functions below duplicate thesis_results.py rather than importing it, so
# this script stays runnable on its own. They are the same implementations.

METRIC_FAMILY = {
    # continuous, unbounded -> parametric
    "rt":               "parametric",
    "sync_ms":          "parametric",
    # rates bounded at zero, with a floor and many ties -> nonparametric
    "singleton":        "nonparametric",
    "singleton_rate":   "nonparametric",
}
# Spread/SD-derived variables in uniformity() are parametric by declaration.
# CAVEAT, and it must be stated wherever singleton_spread is reported: SD is
# coupled to the mean for a count near a floor, so a smaller singleton spread
# under a condition partly restates that condition's lower singleton RATE. That
# is the same confound this script's docstring holds against Kendall's W. It
# does not affect rt_spread or sync_ms_spread, which are away from zero.
METRIC_FAMILY.update({f"{k}_spread": "parametric"
                      for k in ("rt", "sync_ms", "singleton")})

FAMILY_TESTS = {
    "parametric":    dict(omnibus="RM ANOVA", posthoc="paired t",
                          effect="d_z", stat="t"),
    "nonparametric": dict(omnibus="Friedman", posthoc="Wilcoxon signed-rank",
                          effect="r_rb", stat="W"),
}


def metric_family(key):
    """Test family for an endpoint. Unknown endpoints raise rather than
    defaulting: an endpoint with no declared family has not been through the
    policy, and picking one at runtime is what rule 1 forbids."""
    fam = METRIC_FAMILY.get(key)
    if fam is None:
        raise KeyError(
            f"endpoint {key!r} has no declared test family. Add it to "
            f"METRIC_FAMILY as 'parametric' or 'nonparametric' on the basis of "
            f"its measurement type, not its observed distribution.")
    return fam


def bonferroni(pvals):
    """Bonferroni correction, adjusted p in input order. Family size = the
    number of finite p values passed in, which must be stated alongside."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    m = int(ok.sum())
    if m == 0:
        return out.tolist()
    out[ok] = np.minimum(1.0, m * p[ok])
    return out.tolist()


def _orthonormal_contrasts(k):
    """k x (k-1) orthonormal basis of the space orthogonal to the unit vector."""
    C = np.eye(k)[:, :k - 1] - np.ones((k, k - 1)) / k
    Q, _ = np.linalg.qr(C)
    return Q[:, :k - 1]


def gg_epsilon_contrasts(Y, M):
    """Greenhouse-Geisser epsilon for an ARBITRARY contrast space, used for the
    condition x face interaction. Y is subjects x cells, M is cells x df
    orthonormal contrasts; epsilon = tr(S)^2 / (df tr(S S)) on the transformed
    covariance. statsfuns._sphericity gives epsilon for a one-way design only,
    which is why this exists; for one-way cases sf.rm_anova's gg_eps is used
    instead, and the two agree.

    REPORTED ONLY -- no correction is applied to the p beside it."""
    Z = np.asarray(Y, float) @ np.asarray(M, float)
    df = Z.shape[1]
    if Z.shape[0] < 3 or df < 1:
        return np.nan
    S = np.atleast_2d(np.cov(Z, rowvar=False, ddof=1))
    tr, tr2 = np.trace(S), np.trace(S @ S)
    if tr2 <= 0:
        return np.nan
    return float(np.clip((tr ** 2) / (df * tr2), 1.0 / df, 1.0))


def rm_anova_1w(X):
    """One-way RM ANOVA over the columns of X. Thin adapter over sf.rm_anova so
    there is one implementation of the maths in the project; renames its keys to
    the ones this script prints and drops the corrected p-values, which the
    policy does not report."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] < 3 or X.shape[1] < 2:
        return dict(F=np.nan, df1=np.nan, df2=np.nan, p=np.nan, eta_p2=np.nan,
                    eps=np.nan, n=int(X.shape[0]), test="RM ANOVA")
    r = sf.rm_anova(X)
    return dict(F=r["F"], df1=float(r["df1"]), df2=float(r["df2"]), p=r["p"],
                eta_p2=r["eta_p2"], eps=r["gg_eps"], n=int(r["n"]),
                mauchly_p=r.get("mauchly_p", np.nan), test="RM ANOVA")


def friedman_w(X):
    """Friedman with Kendall's W. Thin adapter over sf.friedman."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(axis=1)]
    n, k = X.shape
    if n < 3 or k < 2:
        return dict(chi2=np.nan, df=k - 1, p=np.nan, W=np.nan, n=int(n),
                    test="Friedman")
    try:
        r = sf.friedman(X)
    except Exception:
        return dict(chi2=np.nan, df=k - 1, p=np.nan, W=np.nan, n=int(n),
                    test="Friedman")
    return dict(chi2=r["chi2"], df=float(r["df"]), p=r["p"], W=r["W"],
                n=int(r["n"]), test="Friedman")


def omnibus_by_family(X, key):
    """Dispatch the omnibus on the endpoint's declared family."""
    return rm_anova_1w(X) if metric_family(key) == "parametric" \
        else friedman_w(X)


def omnibus_line(o, tex=False):
    if o.get("test") == "Friedman":
        return (f"Friedman chi2({o['df']:.0f}) = {o['chi2']:.2f}, "
                f"p = {o['p']:.4f}, Kendall's W = {o['W']:.3f}")
    return (f"RM ANOVA F({o['df1']:.0f}, {o['df2']:.0f}) = {o['F']:.2f}, "
            f"p = {o['p']:.4f}, eta_p2 = {o['eta_p2']:.3f}, "
            f"GG eps = {o['eps']:.2f}")


def dz_of(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    s = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / s) if s > 0 else np.nan


def rank_biserial_paired(d):
    """Matched-pairs rank-biserial correlation: (R+ - R-) / (R+ + R-) over the
    signed ranks of the non-zero differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    tot = rp + rn
    return float((rp - rn) / tot) if tot > 0 else np.nan


def paired_ci(d, conf=0.95):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return (np.nan, np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + conf / 2, n - 1)
    return (float(d.mean() - t * se), float(d.mean() + t * se))


def posthoc_by_family(b, a, key):
    """Paired post-hoc b - a, dispatched on the endpoint's DECLARED family.

    Delegates to sf.paired_test / sf.wilcoxon_pair so the underlying maths is
    the project's, and exposes the family's statistic and effect size under
    fixed key names so callers never have to know which test ran. The other
    test is carried as `p_robust` for the CSV only -- the policy forbids
    printing both for the same contrast."""
    b = np.asarray(b, float)
    a = np.asarray(a, float)
    ok = np.isfinite(a) & np.isfinite(b)
    b, a = b[ok], a[ok]
    n = b.size
    fam = metric_family(key)
    row = dict(family=fam, test=FAMILY_TESTS[fam]["posthoc"], n=int(n))
    if n < 3:
        row.update(mdiff=np.nan, ci_lo=np.nan, ci_hi=np.nan, stat=np.nan,
                   stat_name="", df=np.nan, p=np.nan, effect=np.nan,
                   effect_name="", p_robust=np.nan, robust_test="")
        return row
    t = sf.paired_test(b, a)
    w = sf.wilcoxon_pair(b, a)
    row.update(mdiff=t["mdiff"], ci_lo=float(t["ci"][0]), ci_hi=float(t["ci"][1]))
    if fam == "parametric":
        row.update(stat=t["t"], stat_name="t", df=float(t["df"]), p=t["p"],
                   effect=t["dz"], effect_name="d_z",
                   p_robust=w.get("p", np.nan),
                   robust_test="Wilcoxon signed-rank")
    else:
        row.update(stat=w.get("stat", np.nan), stat_name="W", df=np.nan,
                   p=w.get("p", np.nan), effect=w.get("rbc", np.nan),
                   effect_name="r_rb", p_robust=t["p"], robust_test="paired t")
    return row


REPORT: list[str] = []
MACROS: dict[str, str] = {}
_DIGIT = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
          "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def say(*p):
    line = " ".join(str(x) for x in p)
    print(line)
    REPORT.append(line)


def head(t):
    say("\n" + "=" * 76)
    say(t)
    say("=" * 76)


def macro(name, value):
    k = "".join(_DIGIT.get(c, c) for c in str(name) if c.isalnum())
    MACROS["".join(c for c in k if c.isalpha())] = str(value)


# ============================================================================
# srl_analysis.py helpers, reproduced verbatim
# ============================================================================
def _runs(mask):
    mask = np.asarray(mask, bool)
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    return runs


def _block_start(mask, i):
    s = i
    while s - 1 >= 0 and mask[s - 1]:
        s -= 1
    return s


def synchronicity(buttons_path, order):
    """srl_analysis.synchronicity(), unchanged, plus per-press singleton counts.

    For each presented face: find the sustained co-press (B AND A_face); the two
    rising edges that OPEN it give the offset and which button was first.
    Guards: other_A_high (attribution unsafe), B_carried (no fresh B edge).
    """
    b = pd.read_csv(buttons_path)
    t = b["t_device_ms"].to_numpy(float)
    B = b["B"].to_numpy(int) == 1
    Acol = {p: (b[f"A{p}"].to_numpy(int) == 1) for p in range(1, 9)}

    presses = []
    cursor, prev_end = 0, -1
    for step, face in enumerate(order):
        p = FACE_TO_PAIR[face]
        A = Acol[p]
        co = B & A
        runs = [r for r in _runs(co) if r[0] >= cursor]
        if not runs:
            presses.append(dict(step=step + 1, face=face, offset_ms=np.nan,
                                first=None, valid=False, reason="no_copress"))
            continue
        s0, s1 = max(runs, key=lambda r: r[1] - r[0])
        cursor = s1 + 1
        tA = t[_block_start(A, s0)]
        tB = t[_block_start(B, s0)]
        offset = abs(tB - tA)
        first = "B" if tB < tA else ("A" if tA < tB else "sim")
        valid, reason = True, ""
        for p2 in range(1, 9):
            if p2 != p and Acol[p2][s0:s1 + 1].any():
                valid, reason = False, "other_A_high"; break
        if valid and _block_start(B, s0) <= prev_end:
            valid, reason = False, "B_carried"
        prev_end = s1
        presses.append(dict(step=step + 1, face=face, offset_ms=float(offset),
                            first=first, valid=valid, reason=reason))

    valid_off = [x["offset_ms"] for x in presses
                 if x["valid"] and np.isfinite(x["offset_ms"])]
    summ = dict(
        median_sync_ms=float(np.median(valid_off)) if valid_off else np.nan,
        pct_sync_within100=float(np.mean([o <= SYNC_THRESH_MS for o in valid_off]))
        if valid_off else np.nan,
        sync_valid_frac=float(np.mean([x["valid"] for x in presses]))
        if presses else np.nan,
    )
    return summ, presses, singletons(b, t, B, Acol)


def singletons(b, t, B, Acol):
    """
    A-channel presses that never form a co-press with B.

    srl_analysis.py cannot see these: for each face it keeps only the longest
    co-press run, so failed attempts are discarded before any metric is formed.
    Returns one row per A press with a flag.
    """
    rows = []
    for p in range(1, 9):
        A = Acol[p]
        corun = _runs(B & A)
        for s, e in _runs(A):
            joined = any(cs >= s and ce <= e for cs, ce in corun)
            rows.append(dict(pair=p, face=PAIR_TO_FACE[p],
                             t_start_s=t[s] / 1000.0,
                             duration_s=(t[e] - t[s]) / 1000.0,
                             singleton=not joined))
    return rows


# ============================================================================
# discovery, mirroring srl_analysis.discover_trials
# ============================================================================
STEM = re.compile(
    r"^(?P<pid>P\d+)_T(?P<trial>\d+)_(?P<cond>C\d)_b(?P<block>\d+)"
    r"_seq(?P<seq>\d+)_(?P<ts>.+?)_(?P<kind>grid|buttons|events|robot)\.csv$")


def norm_pid(v):
    m = re.search(r"(\d+)", str(v))
    return f"P{int(m.group(1)):02d}" if m else str(v)


def discover(recordings_dir):
    trials = {}
    for path in glob.glob(os.path.join(recordings_dir, "**", "*.csv"),
                          recursive=True):
        m = STEM.match(os.path.basename(path))
        if not m:
            continue
        d = m.groupdict()
        key = (d["pid"], d["trial"], d["cond"], d["block"], d["seq"], d["ts"])
        rec = trials.setdefault(key, dict(
            participant=norm_pid(d["pid"]), trial=int(d["trial"]),
            condition=d["cond"], block=int(d["block"]),
            sequence=int(d["seq"]), paths={}))
        rec["paths"][d["kind"]] = path
    return sorted(trials.values(), key=lambda r: (r["participant"], r["trial"]))


def build_per_press(recordings_dir, denom):
    """One row per presented face, with RT, sync and singleton count."""
    rows = []
    trials = discover(recordings_dir)
    if not trials:
        return pd.DataFrame(), pd.DataFrame()
    say(f"  {len(trials)} trials discovered")

    press_rows, trial_rows = [], []
    for tr in trials:
        paths = tr["paths"]
        if "grid" not in paths:
            continue
        g = pd.read_csv(paths["grid"])
        if g.empty:
            continue
        order = [int(x) for x in g["face"].tolist()]
        meta = {k: tr[k] for k in ("participant", "trial", "condition",
                                   "block", "sequence")}

        summ, presses, singles = ({}, [], [])
        if "buttons" in paths:
            summ, presses, singles = synchronicity(paths["buttons"], order)

        pmap = {p["face"]: p for p in presses}
        sing = pd.DataFrame(singles)
        for _, r in g.iterrows():
            face = int(r["face"])
            pr = pmap.get(face, {})
            if len(sing):
                sub = sing[sing.face == face]
                n_press, n_single = len(sub), int(sub.singleton.sum())
            else:
                n_press = n_single = np.nan
            press_rows.append(dict(**meta, step=int(r["step"]), face=face,
                                   reaction_time_s=float(r["reaction_time_s"]),
                                   wrong_presses=int(r["wrong_presses"]),
                                   sync_ms=pr.get("offset_ms", np.nan),
                                   sync_valid=pr.get("valid", np.nan),
                                   first=pr.get("first"),
                                   n_press=n_press, n_singleton=n_single))

        n_s = int(sing.singleton.sum()) if len(sing) else np.nan
        n_p = len(sing) if len(sing) else np.nan
        n_ok = (n_p - n_s) if len(sing) else np.nan
        trial_rows.append(dict(**meta, **summ,
                               median_rt_s=float(g["reaction_time_s"].median()),
                               singleton_presses=n_s, n_presses=n_p,
                               n_completed_pairs=n_ok,
                               singleton_rate=(n_s / n_p if denom == "all"
                                               else (n_s / n_ok if n_ok else np.nan))))

    return pd.DataFrame(press_rows), pd.DataFrame(trial_rows)


# ============================================================================
def distance_matrix(pitch=1.0):
    pos = {f: np.array([((c % 3) - 1) * pitch, (1 - c // 3) * pitch], float)
           for f, c in zip(range(1, N_FACES + 1), FACE_CELLS)}
    faces = sorted(pos)
    M = pd.DataFrame(index=faces, columns=faces, dtype=float)
    for a in faces:
        for bb in faces:
            M.loc[a, bb] = float(np.linalg.norm(pos[a] - pos[bb]))
    return M


def reconcile(trial_df):
    """Reproduce the srl_analysis participant x condition aggregate."""
    head("1  Reconciliation with srl_analysis.py / run_analysis.py")
    cols = [c for c in ("median_sync_ms", "pct_sync_within100",
                        "sync_valid_frac", "median_rt_s") if c in trial_df]
    ppc = (trial_df.groupby(["participant", "condition"], as_index=False)[cols]
           .mean(numeric_only=True))
    say("  aggregation: median within trial, then MEAN over the condition's "
        "trials (as run_analysis.py does)")
    for cond in CONDITIONS:
        d = ppc[ppc.condition == cond]
        if d.empty:
            continue
        bits = [f"{c} {d[c].mean():.3f}" for c in cols if d[c].notna().any()]
        say(f"    {COND_LABEL[cond]:16s} " + "  |  ".join(bits))
        if "median_sync_ms" in d:
            macro(f"sync{cond}", f"{d['median_sync_ms'].mean():.0f}")
    say("  Compare these against per_participant_condition.csv - they should "
        "match to rounding. If they do not, the recordings folder differs.")
    return ppc


def singleton_report(trial_df, denom):
    head("2  Singletons  (not computed by srl_analysis.py)")
    if "singleton_rate" not in trial_df or trial_df.singleton_rate.isna().all():
        say("  [skipped] no button files")
        return
    say(f"  denominator: {'all A presses' if denom=='all' else 'completed pairs'}")
    ppc = (trial_df.groupby(["participant", "condition"], as_index=False)
           [["singleton_rate", "singleton_presses", "n_presses"]]
           .mean(numeric_only=True))
    for cond in CONDITIONS:
        d = ppc[ppc.condition == cond]
        if d.empty:
            continue
        say(f"    {COND_LABEL[cond]:16s} rate {d.singleton_rate.mean():.3f}  "
            f"({d.singleton_presses.mean():.2f} of {d.n_presses.mean():.2f} "
            f"presses per trial)")
        macro(f"single{cond}", f"{d.singleton_rate.mean():.3f}")
    w = ppc.pivot_table(index="participant", columns="condition",
                        values="singleton_rate").reindex(
        columns=CONDITIONS).dropna()
    if len(w) >= 3:
        f = sf.friedman(w.values)
        say(f"  Friedman chi2({f['df']}) = {f['chi2']:.2f}, p = {f['p']:.4f} "
            f"(bounded proportion, so rank-based)")
        macro("singlechi", f"{f['chi2']:.2f}")
        macro("singlep", f"{f['p']:.3f}")
    say("  NOTE: srl_analysis.py keeps only the LONGEST co-press run per face, "
        "so these failed attempts are invisible to your existing pipeline.")


def per_face_report(press_df, out):
    head("3  Per-face breakdown")
    if press_df.empty:
        return pd.DataFrame()
    pf = (press_df.groupby(["participant", "condition", "face"], as_index=False)
          .agg(rt=("reaction_time_s", "median"),
               sync_ms=("sync_ms", "median"),
               singleton=("n_singleton", "mean"),
               n=("face", "size")))
    for cond in CONDITIONS:
        d = pf[pf.condition == cond]
        if d.empty:
            continue
        say(f"  {COND_LABEL[cond]:16s} RT {d.rt.median():.2f} s  |  "
            f"sync {d.sync_ms.median():.0f} ms  |  "
            f"singletons/face {d.singleton.mean():.2f}")
    say("\n  Kendall's W across the 8 faces: how consistently participants "
        "agree on the face ordering.")
    say("  Tie-corrected (Siegel & Castellan 1988) and tested by permutation "
        "rather than the chi-square approximation, which is asymptotic in the "
        "number of raters and unreliable at n = 10 (Legendre 2005).")
    say(f"  Chance floor: under H0, E[W] = 1/n = {1/max(pf.participant.nunique(),1):.2f}. "
        "W near that value is not evidence of a weak effect, it is chance.")
    wrows = []
    for col, lab in (("rt", "reaction time"), ("sync_ms", "synchronicity"),
                     ("singleton", "singletons per face")):
        if col not in pf.columns:
            continue
        say(f"  {lab}:")
        for cond in CONDITIONS:
            w = (pf[pf.condition == cond].pivot_table(
                index="participant", columns="face", values=col)
                .dropna(how="any"))
            if w.shape[0] < 3 or w.shape[1] < 3:
                say(f"    {COND_LABEL[cond]:16s} [too few complete cases]")
                continue
            r = sf.kendall_w_full(w.values)
            tie = ("" if not np.isfinite(r["tie_inflation"])
                   or abs(r["tie_inflation"] - 1) < .01
                   else f"  (ties inflated W by x{r['tie_inflation']:.2f})")
            say(f"    {COND_LABEL[cond]:16s} W = {r['W']:.3f} "
                f"(adj {r['W_adj']:+.3f}), permutation p = {r['p_perm']:.4f}, "
                f"chi2({r['df']}) = {r['chi2']:.2f}, p_chi2 = "
                f"{r['p_chi2']:.4f}{tie}")
            macro(f"kw{cond}{col}", f"{r['W']:.3f}")
            macro(f"kwp{cond}{col}", f"{r['p_perm']:.3f}")
            wrows.append(dict(measure=lab, key=col, condition=cond, **r))
    if wrows:
        wdf = pd.DataFrame(wrows)
        # Bonferroni within each measure's family of 3 conditions, matching the
        # policy in thesis_results.py. The permutation p being corrected is the
        # right p to use here (see the policy note at the top of this file).
        wdf["p_bonf"] = np.nan
        for key, sub in wdf.groupby("key"):
            wdf.loc[sub.index, "p_bonf"] = bonferroni(sub["p_perm"].tolist())
            wdf.loc[sub.index, "family_size"] = len(sub)
        wdf.to_csv(out / "kendall_w.csv", index=False)
        n_lost = int(((wdf["p_perm"] < .05) & (wdf["p_bonf"] >= .05)).sum())
        say(f"\n  Bonferroni within each measure's family of 3 conditions "
            f"({len(wdf)} W tests total)"
            + (f"; correction removes {n_lost}." if n_lost else "."))
    say("  W and the spread in section 5 answer DIFFERENT questions and can "
        "disagree: W is agreement about the ORDER of faces, spread is the "
        "MAGNITUDE of the differences. High agreement about small differences "
        "gives high W and low spread. Report both.")

    pf.to_csv(out / "per_face_summary.csv", index=False)
    return pf


def per_face_contrasts(pf, out):
    """
    C2 vs C1 at each face, Holm-corrected within each measure's family of 8.

    This is 8 paired comparisons per measure on n = 10, so it is exploratory:
    the design detects d_z >= 1.0 at 80% power, and a per-face test has no more
    data than the whole-condition test it decomposes. Effect sizes carry the
    interpretation; the p-values locate where to look.
    """
    head("4  C2 vs C1 at each face")
    if pf.empty:
        say("  [skipped]")
        return pd.DataFrame()

    mdz = sf.min_detectable_dz(pf.participant.nunique())
    say(f"  n = {pf.participant.nunique()} participants; a per-face contrast "
        f"detects d_z >= {mdz:.2f} at 80% power. This is a design sensitivity "
        f"statement computed from n alone, not observed power.")
    say("  Bonferroni within each measure across the 8 faces. The test is fixed "
        "by the endpoint's declared family, not chosen per contrast: "
        + ", ".join(f"{k} -> {FAMILY_TESTS[metric_family(k)]['posthoc']}"
                    for k in ("rt", "sync_ms", "singleton")))

    rows = []
    for col, lab, nd in (("rt", "reaction time (s)", 2),
                         ("sync_ms", "synchronicity (ms)", 0),
                         ("singleton", "singletons per face", 2)):
        if col not in pf or pf[col].notna().sum() < 10:
            continue
        fam = metric_family(col)
        faces = sorted(pf.face.dropna().unique())
        recs = []
        for face in faces:
            w = (pf[pf.face == face]
                 .pivot_table(index="participant", columns="condition",
                              values=col))
            if not {"C1", "C2"}.issubset(w.columns):
                continue
            w = w[["C1", "C2"]].dropna()
            if len(w) < 4:
                continue
            r = posthoc_by_family(w["C2"], w["C1"], col)
            recs.append(dict(measure=lab, key=col, face=int(face), n=len(w),
                             C1=float(w["C1"].mean()), C2=float(w["C2"].mean()),
                             diff=r["mdiff"], effect=r["effect"],
                             effect_name=r["effect_name"], stat=r["stat"],
                             stat_name=r["stat_name"], p_raw=r["p"],
                             test=r["test"],
                             p_robust=r["p_robust"],       # CSV only
                             robust_test=r.get("robust_test", "")))
        if not recs:
            continue
        for r, pb in zip(recs, bonferroni([r["p_raw"] for r in recs])):
            r["p_bonf"] = pb
            r["family_size"] = len(recs)
            r["sig"] = pb < .05
            r["sig_uncorrected"] = (r["p_raw"] < .05) and not r["sig"]

        eff_hdr = recs[0]["effect_name"]
        say(f"\n  {lab}  [{fam}, {recs[0]['test']}, Bonferroni across "
            f"{len(recs)} faces]:")
        say(f"    {'face':>4s} {'C1':>8s} {'C2':>8s} {'diff':>8s} "
            f"{eff_hdr:>6s} {'p':>7s} {'p_bonf':>7s}   n")
        any_sig = False
        for r in recs:
            flag = ""
            if r["sig"]:
                flag = "  *  C2 " + ("LOWER" if r["diff"] < 0 else "HIGHER")
                any_sig = True
            elif r["sig_uncorrected"]:
                flag = "  (uncorrected only)"
            say(f"    {r['face']:4d} {r['C1']:8.{nd}f} {r['C2']:8.{nd}f} "
                f"{r['diff']:+8.{nd}f} {r['effect']:+6.2f} {r['p_raw']:7.4f} "
                f"{r['p_bonf']:7.4f}  {r['n']:2d}{flag}")
        if not any_sig:
            say("    -> no face differs after Bonferroni correction. This is a "
                "null, not evidence that the faces behave identically.")
            n_unc = sum(r["sig_uncorrected"] for r in recs)
            if n_unc:
                say(f"    -> {n_unc} face(s) significant uncorrected; with 8 "
                    "tests at alpha = .05 you expect 0.4 by chance, so do not "
                    "report these as findings.")
        rows.extend(recs)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "per_face_contrasts.csv", index=False)
        for _, r in df[df.sig].iterrows():
            macro(f"sigface{r.key}{int(r.face)}", f"{r.effect:.2f}")
    return df


def uniformity(pf, out, spread="sd"):
    """
    Does a condition make performance MORE UNIFORM across the eight faces?

    Positive-evidence version of the Kendall's-W observation. For each
    participant x condition the spread across faces (SD, or IQR with
    --spread iqr) becomes a single dependent variable, compared across
    conditions. A smaller spread under C2 is direct evidence of equalised
    access, rather than an inference from a non-significant W.

    Why not W: W has a floor of 1/n (0.10 at n = 10), so a low value is close
    to chance rather than to zero; ties deflate it, and C2 has more tied
    zero-singleton faces precisely BECAUSE its rate is lower, so a low W is
    partly a restatement of the rate; and reading a non-significant W as
    'no face effect' is accepting the null.

    CAVEAT ON THE SINGLETON SPREAD: the mean-restatement objection above applies
    to the SD too, because SD is coupled to the mean for a count near a floor. A
    smaller singleton spread under C2 is not cleanly separable from C2's lower
    singleton rate at this n. rt and sync_ms are away from zero and unaffected.
    """
    head("5  Uniformity across faces  (spread as the dependent variable)")
    if pf.empty:
        say("  [skipped]")
        return pd.DataFrame()

    agg = (lambda x: x.std(ddof=1)) if spread == "sd" else \
          (lambda x: x.quantile(.75) - x.quantile(.25))
    say(f"  spread = {'SD' if spread == 'sd' else 'IQR'} across the 8 faces, "
        "one value per participant x condition")
    say("  lower = more uniform. Friedman first, then the planned C2 vs C1 "
        "contrast.")

    rows = []
    for col, lab, nd in (("rt", "reaction time (s)", 3),
                         ("sync_ms", "synchronicity (ms)", 1),
                         ("singleton", "singletons per face", 3)):
        if col not in pf or pf[col].notna().sum() < 10:
            continue
        w = (pf.pivot_table(index="participant", columns="condition",
                            values=col, aggfunc=agg)
             .reindex(columns=CONDITIONS).dropna(how="any"))
        if len(w) < 4:
            say(f"\n  {lab}: [too few complete cases]")
            continue

        say(f"\n  {lab}:")
        for c in CONDITIONS:
            d = sf.describe(w[c])
            say(f"    {COND_LABEL[c]:16s} spread {d['mean']:.{nd}f} "
                f"(SD {d['sd']:.{nd}f})  median {d['median']:.{nd}f}")
            macro(f"spread{c}{col}", f"{d['mean']:.{nd}f}")

        # The spread variable has its own declared family (see METRIC_FAMILY).
        skey = f"{col}_spread"
        fam = metric_family(skey)
        om = omnibus_by_family(w.values, skey)
        say(f"    [{fam}]  {omnibus_line(om)}   n = {len(w)}")
        macro(f"unif{col}stat", f"{om.get('F', om.get('chi2')):.2f}")
        macro(f"unif{col}p", f"{om['p']:.3f}")

        res = []
        for hi, lo in (("C1", "C0"), ("C2", "C1")):
            r = posthoc_by_family(w[hi], w[lo], skey)
            r["contrast"] = f"{hi} vs {lo}"
            res.append(r)
        for r, pb in zip(res, bonferroni([x["p"] for x in res])):
            r["p_bonf"] = pb
            r["family_size"] = len(res)
        for r in res:
            verdict = ""
            if r["p_bonf"] < .05:
                verdict = ("  *  MORE uniform" if r["mdiff"] < 0
                           else "  *  LESS uniform")
            say(f"    {r['contrast']:10s} diff {r['mdiff']:+.{nd}f} "
                f"CI [{r['ci_lo']:+.{nd}f}, {r['ci_hi']:+.{nd}f}], "
                f"{r['effect_name']} = {r['effect']:+.2f}, {r['test']} "
                f"p_bonf = {r['p_bonf']:.4f}  (n = {r['n']}){verdict}")
            tag = r["contrast"].replace(" vs ", "")
            macro(f"unif{col}{tag}dz", f"{r['effect']:.2f}")
            macro(f"unif{col}{tag}p", f"{r['p_bonf']:.4f}")
            rows.append(dict(measure=lab, key=col, family=fam,
                             **{k: r[k] for k in
                                ("contrast", "n", "mdiff", "ci_lo", "ci_hi",
                                 "effect", "effect_name", "stat", "stat_name",
                                 "p", "p_bonf", "test", "p_robust")}))

        c2c1 = [r for r in res if r["contrast"] == "C2 vs C1"][0]
        if c2c1["p_bonf"] >= .05:
            mdz = sf.min_detectable_dz(len(w))
            say(f"    -> no significant difference in uniformity between C2 and "
                f"C1. As a design sensitivity statement, n = {len(w)} detects "
                f"d_z >= {mdz:.2f} at 80% power and the observed effect is "
                f"{abs(c2c1['effect']):.2f}; this does not establish that the "
                f"conditions are equally uniform.")
        if col == "singleton":
            say("    -> NOTE: singleton spread is coupled to the singleton rate "
                "(SD scales with the mean near a floor), so a uniformity "
                "difference here partly restates the rate difference.")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "uniformity.csv", index=False)
    say("\n  Reporting note: quote the spread contrast, not Kendall's W. W has "
        "a 1/n floor, is deflated by ties, and a non-significant W cannot "
        "support 'the face effect disappeared'. Section 8 gives the magnitude "
        "of the face effect itself, which is the number to quote when the "
        "question is 'how different are the buttons'.")
    return df


# ============================================================================
# 8  FACE EFFECT  -  do the eight buttons differ, and by how much?
# ============================================================================
# The question this answers is not "does C2 change performance" (sections 1-5)
# but "is the board itself uneven, and does the exoskeleton flatten it". Unit of
# analysis throughout is participant x face: the participant's median over that
# participant's trials for that face, which is what per_face_report already
# builds. Every test is routed through METRIC_FAMILY.

FACE_ENDPOINTS = (("rt", "reaction time (s)", 3),
                  ("sync_ms", "synchronicity (ms)", 1),
                  ("singleton", "singletons per face", 3))


def _face_wide(pf, cond, col):
    """participant x face matrix for one condition, complete cases only."""
    w = (pf[pf.condition == cond]
         .pivot_table(index="participant", columns="face", values=col)
         .dropna(how="any"))
    return w.reindex(sorted(w.columns), axis=1)


def corner_edge_split():
    """Faces classified from FACE_CELLS: corners are cells 0,2,6,8 of the 3x3
    grid, edges are 1,3,5,7. Returns (corner_faces, edge_faces)."""
    corners, edges = [], []
    for face, cell in zip(range(1, N_FACES + 1), FACE_CELLS):
        (corners if cell in (0, 2, 6, 8) else edges).append(face)
    return corners, edges


def _perm_spearman(x, y, n_perm=20000, seed=0):
    """Spearman rho with a permutation p. With 8 faces there are 8! = 40320
    orderings, so this enumerates exactly when it can and samples otherwise.
    Permutation, not the asymptotic p, because n = 8 pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, int(x.size)
    obs = float(stats.spearmanr(x, y).statistic)
    from itertools import permutations
    if x.size <= 8:
        idx = list(permutations(range(x.size)))
        stat = np.array([stats.spearmanr(x, y[list(i)]).statistic for i in idx])
        p = float((np.abs(stat) >= abs(obs) - 1e-12).mean())
    else:
        rng = np.random.default_rng(seed)
        cnt = sum(abs(float(stats.spearmanr(x, rng.permutation(y)).statistic))
                  >= abs(obs) - 1e-12 for _ in range(n_perm))
        p = (cnt + 1) / (n_perm + 1)
    return obs, float(p), int(x.size)


def interaction_condition_face(pf, col):
    """Two-way RM ANOVA, condition x face interaction, participant as the unit.

    The SS decomposition is sf.rm_anova_2way (both factors within-subject, error
    term for the interaction is the subject x condition x face residual). Only
    the AB term is reported. GG epsilon is added here because sf.rm_anova_2way
    does not return one; it is computed on the interaction contrast space, the
    Kronecker product of the condition and face orthonormal contrasts, and is
    REPORTED rather than applied."""
    sub = pf[pf[col].notna()]
    piv = sub.pivot_table(index="participant", columns=["condition", "face"],
                          values=col)
    conds = [c for c in CONDITIONS if c in piv.columns.get_level_values(0)]
    faces = sorted(set(piv.columns.get_level_values(1)))
    want = [(c, f) for c in conds for f in faces]
    if len(conds) < 2 or len(faces) < 2 or not set(want).issubset(piv.columns):
        return None
    piv = piv[want].dropna(how="any")
    n, C, F = len(piv), len(conds), len(faces)
    if n < 3:
        return None
    cube = piv.to_numpy(float).reshape(n, C, F)
    ab = sf.rm_anova_2way(cube)["AB"]
    Mi = np.kron(_orthonormal_contrasts(C), _orthonormal_contrasts(F))
    return dict(key=col, n=int(n), n_conditions=C, n_faces=F,
                F=float(ab["F"]), df1=float(ab["df1"]), df2=float(ab["df2"]),
                p=float(ab["p"]), eta_p2=float(ab["eta_p2"]),
                eps=gg_epsilon_contrasts(piv.to_numpy(float), Mi),
                test="RM ANOVA (condition x face interaction)")


def face_effect(pf, out):
    head("8  Face effect  -  do the eight buttons differ, and by how much?")
    if pf.empty:
        say("  [skipped] no per-face data")
        return {}
    say("  Unit of analysis: participant x face (that participant's median over "
        "their trials). Test family fixed by endpoint: "
        + ", ".join(f"{k} -> {FAMILY_TESTS[metric_family(k)]['omnibus']}"
                    for k, _, _ in FACE_ENDPOINTS))

    om_rows, mag_rows, out_rows, sim_rows, geo_rows = [], [], [], [], []
    summary = {}

    # ---- 8.1 omnibus + 8.2 magnitude ------------------------------------
    for col, lab, nd in FACE_ENDPOINTS:
        if col not in pf or pf[col].notna().sum() < 6:
            continue
        fam = metric_family(col)
        say(f"\n  {lab}  [{fam}]")
        recs = []
        for cond in CONDITIONS:
            w = _face_wide(pf, cond, col)
            if w.shape[0] < 3 or w.shape[1] < 3:
                say(f"    {COND_LABEL[cond]:16s} [n = {w.shape[0]} participants "
                    f"x {w.shape[1]} faces -- too few complete cases]")
                continue
            o = omnibus_by_family(w.values, col)
            means = w.mean(axis=0)
            best, worst = means.idxmin(), means.idxmax()
            rng_ = float(means.max() - means.min())
            gm = float(means.mean())
            recs.append(dict(condition=cond, key=col, measure=lab,
                             n_participants=int(w.shape[0]),
                             n_faces=int(w.shape[1]), **o))
            mag_rows.append(dict(
                condition=cond, key=col, measure=lab,
                n_participants=int(w.shape[0]), n_faces=int(w.shape[1]),
                best_face=int(best), best_mean=float(means.min()),
                worst_face=int(worst), worst_mean=float(means.max()),
                range=rng_, range_pct_of_mean=(100 * rng_ / gm) if gm else np.nan,
                between_face_sd=float(means.std(ddof=1)), condition_mean=gm))
        if not recs:
            continue
        # Bonferroni across the three conditions within this endpoint
        for r, pb in zip(recs, bonferroni([r["p"] for r in recs])):
            r["p_bonf"] = pb
            r["family_size"] = len(recs)
        for r in recs:
            m = next(x for x in mag_rows
                     if x["condition"] == r["condition"] and x["key"] == col)
            verdict = ("faces DIFFER" if r["p_bonf"] < .05
                       else "no significant difference between faces")
            say(f"    {COND_LABEL[r['condition']]:16s} {omnibus_line(r)}"
                f"   n = {r['n_participants']}")
            say(f"    {'':16s} best face {m['best_face']} "
                f"{m['best_mean']:.{nd}f}, worst face {m['worst_face']} "
                f"{m['worst_mean']:.{nd}f}, range {m['range']:.{nd}f} "
                f"({m['range_pct_of_mean']:.0f}% of the condition mean), "
                f"between-face SD {m['between_face_sd']:.{nd}f}")
            say(f"    {'':16s} p_bonf = {r['p_bonf']:.4f} across "
                f"{r['family_size']} conditions -> {verdict}")
            macro(f"facerange{r['condition']}{col}", f"{m['range']:.{nd}f}")
            macro(f"facerangepct{r['condition']}{col}",
                  f"{m['range_pct_of_mean']:.0f}")
            summary[(col, r["condition"])] = dict(
                differ=bool(r["p_bonf"] < .05), range=m["range"],
                pct=m["range_pct_of_mean"], nd=nd, label=lab)
        om_rows.extend(recs)

        # ---- 8.3 which faces differ, vs the condition grand mean ---------
        for r in recs:
            if r["p_bonf"] >= .05:
                continue
            cond = r["condition"]
            w = _face_wide(pf, cond, col)
            gmean = w.mean(axis=1)                     # participant grand mean
            sub = []
            for face in w.columns:
                d = (w[face] - gmean).to_numpy(float)
                if np.count_nonzero(np.isfinite(d)) < 3:
                    continue
                if fam == "parametric":
                    t, p = stats.ttest_1samp(d, 0.0)
                    stat, sname, eff, ename = float(t), "t", dz_of(d), "d_z"
                else:
                    try:
                        res = stats.wilcoxon(d)
                        stat, p = float(res.statistic), float(res.pvalue)
                    except ValueError:
                        stat, p = 0.0, 1.0
                    sname, eff, ename = "W", rank_biserial_paired(d), "r_rb"
                sub.append(dict(condition=cond, key=col, measure=lab,
                                face=int(face), n=int(np.isfinite(d).sum()),
                                mean_dev=float(np.nanmean(d)), stat=stat,
                                stat_name=sname, effect=eff, effect_name=ename,
                                p_raw=float(p), test=FAMILY_TESTS[fam]["posthoc"]))
            if not sub:
                continue
            for x, pb in zip(sub, bonferroni([x["p_raw"] for x in sub])):
                x["p_bonf"] = pb
                x["family_size"] = len(sub)
            say(f"    {COND_LABEL[cond]:16s} faces vs the condition mean "
                f"(Bonferroni across {len(sub)}; the 8 deviations sum to zero "
                f"so they are not independent):")
            flagged = [x for x in sub if x["p_bonf"] < .05]
            for x in sorted(sub, key=lambda z: z["mean_dev"]):
                mark = ("  <-- " + ("SLOWER/WORSE" if x["mean_dev"] > 0
                                    else "FASTER/BETTER")) if x["p_bonf"] < .05 else ""
                say(f"      face {x['face']}  dev {x['mean_dev']:+.{nd}f}  "
                    f"{x['effect_name']} {x['effect']:+.2f}  "
                    f"p_bonf {x['p_bonf']:.4f}  n {x['n']}{mark}")
            if not flagged:
                say("      -> omnibus fired but no single face is an outlier "
                    "against the mean; the effect is spread across faces.")
            out_rows.extend(sub)

    # ---- 8.4a is the same face pattern present in every condition? -------
    say("\n  Pattern similarity: Spearman correlation of the eight face means "
        "between conditions (permutation p, exact over 8! orderings).")
    say("  A high correlation means the same faces are hard in every condition, "
        "i.e. difficulty is geometric rather than induced by the condition.")
    for col, lab, nd in FACE_ENDPOINTS:
        if col not in pf or pf[col].notna().sum() < 6:
            continue
        prof = {}
        for cond in CONDITIONS:
            w = _face_wide(pf, cond, col)
            if w.shape[0] >= 3 and w.shape[1] >= 3:
                prof[cond] = w.mean(axis=0)
        pairs = [(a, b) for i, a in enumerate(CONDITIONS)
                 for b in CONDITIONS[i + 1:] if a in prof and b in prof]
        if not pairs:
            continue
        say(f"    {lab}:")
        recs = []
        for a, b in pairs:
            faces = sorted(set(prof[a].index) & set(prof[b].index))
            rho, p, nn = _perm_spearman(prof[a][faces].values,
                                        prof[b][faces].values)
            recs.append(dict(key=col, measure=lab, pair=f"{a}-{b}",
                             spearman_rho=rho, p_perm=p, n_faces=nn))
        for r, pb in zip(recs, bonferroni([r["p_perm"] for r in recs])):
            r["p_bonf"] = pb
            r["family_size"] = len(recs)
        for r in recs:
            say(f"      {r['pair']:8s} rho = {r['spearman_rho']:+.3f}, "
                f"permutation p = {r['p_perm']:.4f}, p_bonf = "
                f"{r['p_bonf']:.4f}  ({r['n_faces']} faces)")
        sim_rows.extend(recs)

    # ---- 8.4b condition x face interaction -------------------------------
    say("\n  Condition x face interaction (two-way RM ANOVA). A significant "
        "interaction is the direct evidence that the condition changes WHICH "
        "faces are hard, not merely the overall level.")
    inter_rows = []
    for col, lab, nd in FACE_ENDPOINTS:
        if metric_family(col) != "parametric":
            say(f"    {lab}: [not run -- nonparametric endpoint, and there is no "
                f"standard rank-based interaction test]")
            continue
        r = interaction_condition_face(pf, col)
        if r is None:
            say(f"    {lab}: [too few complete participant x condition x face "
                f"cases]")
            continue
        say(f"    {lab}: F({r['df1']:.0f}, {r['df2']:.0f}) = {r['F']:.2f}, "
            f"p = {r['p']:.4f}, eta_p2 = {r['eta_p2']:.3f}, GG eps = "
            f"{r['eps']:.2f}, n = {r['n']}")
        say(f"      -> {'the face pattern DIFFERS across conditions' if r['p'] < .05 else 'no significant condition x face interaction'}")
        macro(f"interact{col}p", f"{r['p']:.4f}")
        macro(f"interact{col}F", f"{r['F']:.2f}")
        inter_rows.append(r)
        summary[("interaction", col)] = dict(p=r["p"], F=r["F"])

    # ---- 8.5 corner vs edge ---------------------------------------------
    corners, edges = corner_edge_split()
    say(f"\n  Corner vs edge faces (from FACE_CELLS): corners {corners}, "
        f"edges {edges}. Paired at participant level within each condition.")
    for col, lab, nd in FACE_ENDPOINTS:
        if col not in pf or pf[col].notna().sum() < 6:
            continue
        say(f"    {lab}:")
        recs = []
        for cond in CONDITIONS:
            w = _face_wide(pf, cond, col)
            cf = [f for f in corners if f in w.columns]
            ef = [f for f in edges if f in w.columns]
            if w.shape[0] < 3 or not cf or not ef:
                say(f"      {COND_LABEL[cond]:16s} [n = {w.shape[0]}, "
                    f"{len(cf)} corner / {len(ef)} edge faces -- skipped]")
                continue
            r = posthoc_by_family(w[cf].mean(axis=1), w[ef].mean(axis=1), col)
            r.update(condition=cond, key=col, measure=lab,
                     n_corner_faces=len(cf), n_edge_faces=len(ef),
                     corner_mean=float(w[cf].mean(axis=1).mean()),
                     edge_mean=float(w[ef].mean(axis=1).mean()))
            recs.append(r)
        for r, pb in zip(recs, bonferroni([r["p"] for r in recs])):
            r["p_bonf"] = pb
            r["family_size"] = len(recs)
        for r in recs:
            verdict = ""
            if r["p_bonf"] < .05:
                verdict = ("  *  corners WORSE" if r["mdiff"] > 0
                           else "  *  corners BETTER")
            say(f"      {COND_LABEL[r['condition']]:16s} corner "
                f"{r['corner_mean']:.{nd}f} vs edge {r['edge_mean']:.{nd}f}, "
                f"diff {r['mdiff']:+.{nd}f} CI [{r['ci_lo']:+.{nd}f}, "
                f"{r['ci_hi']:+.{nd}f}], {r['effect_name']} "
                f"{r['effect']:+.2f}, {r['test']} p_bonf = {r['p_bonf']:.4f} "
                f"(n = {r['n']}){verdict}")
            macro(f"corneredge{r['condition']}{col}", f"{r['effect']:.2f}")
        geo_rows.extend(recs)

    # ---- 8.6 extreme-face sensitivity ------------------------------------
    say("\n  Extreme-face sensitivity: each condition-level mean recomputed with "
        "that condition's single worst face dropped, and the C2-vs-C1 contrast "
        "re-run. This guards against one pathological face carrying a headline "
        "result in thesis_results.py.")
    for col, lab, nd in FACE_ENDPOINTS:
        if col not in pf or pf[col].notna().sum() < 6:
            continue
        full, drop = {}, {}
        for cond in CONDITIONS:
            w = _face_wide(pf, cond, col)
            if w.shape[0] < 3 or w.shape[1] < 3:
                continue
            worst = w.mean(axis=0).idxmax()
            full[cond] = w.mean(axis=1)
            drop[cond] = w.drop(columns=[worst]).mean(axis=1)
        if not {"C1", "C2"} <= set(full):
            say(f"    {lab}: [C1/C2 unavailable]")
            continue
        idx = full["C1"].index.intersection(full["C2"].index)
        if len(idx) < 3:
            say(f"    {lab}: [n = {len(idx)} -- too few paired participants]")
            continue
        a = posthoc_by_family(full["C2"][idx], full["C1"][idx], col)
        wf = _face_wide(pf, "C2", col).mean(axis=0).idxmax()
        b = posthoc_by_family(drop["C2"][idx], drop["C1"][idx], col)
        changed = (a["p"] < .05) != (b["p"] < .05)
        say(f"    {lab}: C2 vs C1 all faces {a['effect_name']} "
            f"{a['effect']:+.2f}, p = {a['p']:.4f}  |  worst C2 face ({wf}) "
            f"dropped: {b['effect_name']} {b['effect']:+.2f}, p = {b['p']:.4f} "
            f"(n = {len(idx)})"
            + ("   <-- VERDICT CHANGES" if changed else "   verdict unchanged"))
        if changed:
            say("      -> a single face is carrying this result. Report the "
                "sensitivity analysis alongside the headline number.")

    for name, rows_ in (("face_omnibus", om_rows), ("face_magnitude", mag_rows),
                        ("face_outliers", out_rows),
                        ("face_pattern_similarity", sim_rows),
                        ("face_geometry", geo_rows)):
        if rows_:
            pd.DataFrame(rows_).to_csv(out / f"{name}.csv", index=False)
    if inter_rows:
        pd.DataFrame(inter_rows).to_csv(out / "face_interaction.csv", index=False)
    return summary


def face_summary(summary):
    """The short verdict block asked for at the end of the run."""
    head("9  Face effect -- summary")
    if not summary:
        say("  [no face-level results]")
        return
    say(f"  {'endpoint':<22s}{'condition':<10s}{'faces differ?':<16s}"
        f"{'largest gap':>14s}{'% of mean':>11s}")
    for (col, cond), v in summary.items():
        if col == "interaction":
            continue
        nd = v["nd"]
        say(f"  {v['label']:<22s}{cond:<10s}"
            f"{('YES' if v['differ'] else 'no'):<16s}"
            f"{v['range']:>14.{nd}f}{v['pct']:>10.0f}%")
    say("")
    for key, v in summary.items():
        if key[0] != "interaction":
            continue
        say(f"  condition x face interaction on {key[1]}: F = {v['F']:.2f}, "
            f"p = {v['p']:.4f} -> "
            + ("the pattern of face difficulty CHANGES across conditions"
               if v["p"] < .05 else
               "no evidence the pattern of face difficulty changes across "
               "conditions"))
    say("\n  Read the pattern-similarity correlations in section 8 alongside "
        "this: a high between-condition rho with a null interaction means the "
        "same faces are hard everywhere, i.e. the unevenness is a property of "
        "the board geometry and the exoskeleton does not redistribute it.")


def distance_report(press_df, out, pitch):
    head("6  Synchronicity and singletons vs inter-face travel distance")
    M = distance_matrix(pitch)
    unit = "mm" if pitch != 1.0 else "grid units"
    say(f"  distances in {unit}; Spearman is scale-invariant so rho does not "
        f"depend on the pitch. Layout: faces 1-8 -> cells {FACE_CELLS}")
    if press_df.empty:
        return None

    d = press_df.sort_values(["participant", "condition", "trial", "step"]).copy()
    d["prev_face"] = d.groupby(["participant", "condition", "trial"])["face"].shift(1)
    d = d.dropna(subset=["prev_face"])
    d["dist"] = [M.loc[int(a), int(b)] for a, b in zip(d.prev_face, d.face)]
    say(f"  {len(d)} transitions, distances "
        f"{', '.join(f'{v:.2f}' for v in sorted(d.dist.unique()))}")

    rows = []
    for cond in CONDITIONS:
        sub = d[d.condition == cond]
        if len(sub) < 8:
            continue
        say(f"\n  {COND_LABEL[cond]}  ({len(sub)} transitions, "
            f"{sub.participant.nunique()} participants)")
        for col, lab in (("reaction_time_s", "reaction time"),
                         ("sync_ms", "synchronicity"),
                         ("n_singleton", "singletons")):
            if col not in sub or sub[col].notna().sum() < 8:
                continue
            rs = []
            for _, gg in sub.groupby("participant"):
                gg = gg[["dist", col]].dropna()
                if len(gg) >= 4 and gg.dist.nunique() > 1 and gg[col].nunique() > 1:
                    r = stats.spearmanr(gg.dist, gg[col]).statistic
                    if np.isfinite(r):
                        rs.append(float(np.clip(r, -.999, .999)))
            within = float(np.tanh(np.mean(np.arctanh(rs)))) if rs else np.nan
            pooled = stats.spearmanr(sub.dist, sub[col], nan_policy="omit")
            pw = (float(stats.ttest_1samp(np.arctanh(rs), 0.0).pvalue)
                  if len(rs) >= 3 else np.nan)
            say(f"    {lab:14s} within rho = {sf.fmt(within,3):>7s} "
                f"(n = {len(rs)}, p = {sf.fmt(pw,4)})  |  pooled rho = "
                f"{pooled.statistic:+.3f} (p = {pooled.pvalue:.4f})")
            if np.isfinite(within) and abs(within) > .1 and \
                    abs(pooled.statistic) > .1 and \
                    np.sign(within) != np.sign(pooled.statistic):
                say("      >>> SIGN FLIP pooled vs within - report the within "
                    "estimate.")
            rows.append(dict(condition=cond, measure=lab, within=within,
                             p_within=pw, pooled=pooled.statistic,
                             p_pooled=pooled.pvalue, n_participants=len(rs)))
            macro(f"dist{cond}{col}", sf.fmt(within, 2))
    if rows:
        pd.DataFrame(rows).to_csv(out / "distance_correlations.csv", index=False)
        say("\n  Sensitivity: at 10 participants the Fisher-z test needs a mean "
            "|rho| of roughly 0.3-0.4 for p < .05; smaller values are "
            "descriptive only.")
    d.to_csv(out / "transitions.csv", index=False)
    return d


# ============================================================================
def figures(pf, trans, out, pitch, contrasts=None, unif=None):
    head("7  Figures")
    if not pf.empty:
        cols = [(c, l, u) for c, l, u in (("rt", "Reaction time", "s"),
                                          ("sync_ms", "Synchronicity", "ms"),
                                          ("singleton", "Singletons per face", ""))
                if c in pf and pf[c].notna().any()]
        if cols:
            fig, axes = plt.subplots(1, len(cols),
                                     figsize=(2.7 * len(cols) + .6, 2.5))
            axes = np.atleast_1d(axes)
            faces = sorted(pf.face.unique())
            w = .26
            for ax, (col, lab, unit) in zip(axes, cols):
                for k, cond in enumerate(CONDITIONS):
                    dd = pf[pf.condition == cond]
                    if dd.empty:
                        continue
                    med = [dd.loc[dd.face == f, col].median() for f in faces]
                    se = [dd.loc[dd.face == f, col].sem() for f in faces]
                    ax.bar(np.arange(len(faces)) + (k - 1) * w, med, w, yerr=se,
                           capsize=1.4, color=COND_COLOR[cond], alpha=.85,
                           label=COND_LABEL[cond],
                           error_kw=dict(lw=.6, ecolor="#444"))
                # mark faces where C2 differs from C1
                if contrasts is not None and not contrasts.empty:
                    sub = contrasts[contrasts.key == col]
                    ymax = ax.get_ylim()[1]
                    for _, r in sub.iterrows():
                        if not (r.sig or r.sig_uncorrected):
                            continue
                        xi = faces.index(r.face)
                        col_vals = [pf.loc[(pf.face == r.face) &
                                           (pf.condition == c), col].median()
                                    for c in ("C1", "C2")]
                        y = np.nanmax(col_vals) if np.isfinite(
                            np.nanmax(col_vals)) else ymax * .8
                        ax.text(xi + w / 2, y + .06 * ymax,
                                "*" if r.sig else "\u00b7",
                                ha="center", va="bottom",
                                fontsize=9 if r.sig else 11,
                                color="#c0392b" if r.sig else "#999")
                ax.set_xticks(range(len(faces)))
                ax.set_xticklabels([int(f) for f in faces], fontsize=6.5)
                ax.set_xlabel("face", fontsize=7)
                ax.set_ylabel(lab + (f" ({unit})" if unit else ""), fontsize=7)
                ax.tick_params(labelsize=6.5, length=2.5, width=.6)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            axes[0].legend(frameon=False, fontsize=6)
            if contrasts is not None and not contrasts.empty:
                fig.text(0.5, -0.04,
                         "* C2 differs from C1, Bonferroni-corrected across the 8 "
                         "faces   \u00b7 uncorrected only",
                         ha="center", fontsize=6, color="#555")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out / f"fig_by_face.{ext}", dpi=300,
                            bbox_inches="tight")
            plt.close(fig)
            say("  fig_by_face.pdf")

    if trans is not None and not trans.empty:
        unit = "mm" if pitch != 1.0 else "grid units"
        cols = [(c, l, u) for c, l, u in
                (("reaction_time_s", "Reaction time", "s"),
                 ("sync_ms", "Synchronicity", "ms"),
                 ("n_singleton", "Singletons", ""))
                if c in trans and trans[c].notna().sum() > 8]
        if cols:
            fig, axes = plt.subplots(1, len(cols),
                                     figsize=(2.7 * len(cols) + .6, 2.5))
            axes = np.atleast_1d(axes)
            for ax, (col, lab, u) in zip(axes, cols):
                for cond in CONDITIONS:
                    s = trans[trans.condition == cond].dropna(subset=["dist", col])
                    if len(s) < 4:
                        continue
                    m = s.groupby("dist")[col].median()
                    e = s.groupby("dist")[col].sem()
                    ax.errorbar(m.index, m.values, yerr=e.values, fmt="o-",
                                ms=3.2, lw=1.0, capsize=1.5, elinewidth=.6,
                                color=COND_COLOR[cond], label=COND_LABEL[cond])
                ax.set_xlabel(f"travel distance ({unit})", fontsize=7)
                ax.set_ylabel(lab + (f" ({u})" if u else ""), fontsize=7)
                ax.tick_params(labelsize=6.5, length=2.5, width=.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            axes[0].legend(frameon=False, fontsize=6)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out / f"fig_vs_distance.{ext}", dpi=300,
                            bbox_inches="tight")
            plt.close(fig)
            say("  fig_vs_distance.pdf")

    # --- uniformity: spread across faces, per participant ------------------
    if not pf.empty:
        cols = [(c, l, u) for c, l, u in (("rt", "Reaction time", "s"),
                                          ("sync_ms", "Synchronicity", "ms"),
                                          ("singleton", "Singletons per face", ""))
                if c in pf and pf[c].notna().any()]
        if cols:
            fig, axes = plt.subplots(1, len(cols),
                                     figsize=(2.5 * len(cols) + .6, 2.6))
            axes = np.atleast_1d(axes)
            for ax, (col, lab, unit) in zip(axes, cols):
                w = (pf.pivot_table(index="participant", columns="condition",
                                    values=col, aggfunc=lambda x: x.std(ddof=1))
                     .reindex(columns=CONDITIONS).dropna(how="any"))
                if w.empty:
                    ax.set_visible(False)
                    continue
                for i, c in enumerate(CONDITIONS, start=1):
                    ax.scatter(np.full(len(w), i) +
                               (np.random.default_rng(0).random(len(w)) - .5) * .16,
                               w[c], s=16, color=COND_COLOR[c], alpha=.75,
                               edgecolor="white", linewidth=.4, zorder=3)
                for _, row in w.iterrows():
                    ax.plot(range(1, 4), row.values, color="#bbb", lw=.5,
                            alpha=.6, zorder=1)
                ax.plot(range(1, 4), [w[c].mean() for c in CONDITIONS], "-o",
                        color="#222", lw=1.4, ms=4, zorder=4)
                ax.set_xticks([1, 2, 3])
                ax.set_xticklabels(CONDITIONS, fontsize=6.5)
                ax.set_ylabel(f"SD across faces\n{lab}"
                              + (f" ({unit})" if unit else ""), fontsize=6.8)
                ax.tick_params(labelsize=6.5, length=2.5, width=.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            fig.text(0.5, -0.04, "lower = more uniform across the eight faces",
                     ha="center", fontsize=6, color="#555")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out / f"fig_uniformity.{ext}", dpi=300,
                            bbox_inches="tight")
            plt.close(fig)
            say("  fig_uniformity.pdf")


    # --- 8-face profile: mean +/- 95% CI per face, three conditions overlaid --
    if not pf.empty:
        cols = [(c, l, u) for c, l, u in (("rt", "Reaction time", "s"),
                                          ("sync_ms", "Synchronicity", "ms"),
                                          ("singleton", "Singletons per face", ""))
                if c in pf and pf[c].notna().any()]
        if cols:
            fig, axes = plt.subplots(1, len(cols),
                                     figsize=(2.7 * len(cols) + .6, 2.5))
            axes = np.atleast_1d(axes)
            faces = sorted(pf.face.dropna().unique())
            for ax, (col, lab, unit) in zip(axes, cols):
                for cond in CONDITIONS:
                    dd = pf[pf.condition == cond]
                    if dd.empty:
                        continue
                    mu, lo, hi = [], [], []
                    for f in faces:
                        d = sf.describe(dd.loc[dd.face == f, col])
                        mu.append(d["mean"])
                        lo.append(d["ci"][0])
                        hi.append(d["ci"][1])
                    mu = np.asarray(mu, float)
                    err = np.vstack([mu - np.asarray(lo, float),
                                     np.asarray(hi, float) - mu])
                    err[~np.isfinite(err)] = 0.0
                    ax.errorbar(np.arange(len(faces)), mu, yerr=err, fmt="o-",
                                ms=3.2, lw=1.0, capsize=1.5, elinewidth=.6,
                                color=COND_COLOR[cond], label=COND_LABEL[cond])
                ax.set_xticks(range(len(faces)))
                ax.set_xticklabels([int(f) for f in faces], fontsize=6.5)
                ax.set_xlabel("face", fontsize=7)
                ax.set_ylabel(lab + (f" ({unit})" if unit else ""), fontsize=7)
                ax.tick_params(labelsize=6.5, length=2.5, width=.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            axes[0].legend(frameon=False, fontsize=6)
            fig.text(0.5, -0.04,
                     "mean \u00b1 95% CI per face; a flat line means the eight "
                     "buttons behave alike in that condition",
                     ha="center", fontsize=6, color="#555")
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(out / f"fig_face_profile.{ext}", dpi=300,
                            bbox_inches="tight")
            plt.close(fig)
            say("  fig_face_profile.pdf")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  python3 analyse_buttons.py --rec recordings "
               "--out button_out --pitch 45")
    ap.add_argument("--rec", default="recordings",
                    help="recordings folder (searched recursively)")
    ap.add_argument("--out", default="./button_out")
    ap.add_argument("--pitch", type=float, default=1.0,
                    help="face spacing in mm, for axis labels only")
    ap.add_argument("--layout", default=None,
                    help="comma-separated 3x3 cell indices for faces 1-8")
    ap.add_argument("--spread", choices=["sd", "iqr"], default="sd",
                    help="spread statistic for the uniformity analysis "
                         "(default sd; iqr is more robust at n=8 faces)")
    ap.add_argument("--singleton-denominator", choices=["all", "completed"],
                    default="all",
                    help="'all' = singletons / all A presses (default); "
                         "'completed' = singletons / completed pairs")
    args = ap.parse_args()

    global FACE_CELLS
    if args.layout:
        FACE_CELLS = [int(x) for x in args.layout.split(",")]
        assert len(FACE_CELLS) == N_FACES, "--layout needs 8 cell indices"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    head("0  Loading")
    press_df, trial_df = build_per_press(args.rec, args.singleton_denominator)
    if press_df.empty:
        sys.exit(f"No trials found under {args.rec!r}. Expected files named "
                 "P*_T*_C*_b*_seq*_*_grid.csv and _buttons.csv")
    say(f"  {len(press_df)} presented faces, "
        f"{press_df.participant.nunique()} participants")
    press_df.to_csv(out / "per_press.csv", index=False)
    trial_df.to_csv(out / "per_trial.csv", index=False)

    reconcile(trial_df)
    singleton_report(trial_df, args.singleton_denominator)
    pf = per_face_report(press_df, out)
    contrasts = per_face_contrasts(pf, out)
    unif = uniformity(pf, out, args.spread)
    trans = distance_report(press_df, out, args.pitch)
    figures(pf, trans, out, args.pitch, contrasts, unif)
    fsum = face_effect(pf, out)
    face_summary(fsum)

    lines = ["% Auto-generated by analyse_buttons.py", ""]
    for k, v in sorted(MACROS.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (out / "button_numbers.tex").write_text("\n".join(lines))
    (out / "button_report.txt").write_text("\n".join(REPORT))
    print(f"\nAll outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
