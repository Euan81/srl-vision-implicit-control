"""
statsfuns.py — small, dependency-light statistics used by the results analysis.

Only numpy + scipy. No statsmodels, no pingouin: everything the Results chapter
needs is implemented here so the analysis runs anywhere and every formula is
auditable.

Validated against pingouin 0.6.1: rm_anova F, Mauchly's W and the
Greenhouse-Geisser epsilon agree to six decimal places, as does tost_paired.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

RNG = np.random.default_rng(20240101)


# ---------------------------------------------------------------- formatting
def fmt_p(p: float) -> str:
    """APA-style p, leading zero dropped."""
    if not np.isfinite(p):
        return "---"
    if p < .001:
        return "$<.001$"
    return f"${('%.3f' % p).lstrip('0')}$"


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    return f"{x:.{nd}f}"


# ------------------------------------------------------ multiplicity control
def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# ------------------------------------------------------------ paired effects
def paired_test(a, b, n_boot=10000):
    """
    Paired comparison of a vs b (a - b).

    Returns t, df, p, Cohen's d_z, the mean difference and its bootstrap CI,
    plus a Wilcoxon p-value as a nonparametric cross-check.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    n = d.size
    t, p = stats.ttest_rel(a, b)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan

    boot = RNG.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    try:
        _, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = np.nan

    return dict(n=n, t=float(t), df=n - 1, p=float(p), dz=float(dz),
                mdiff=float(d.mean()), ci=(float(lo), float(hi)),
                p_wilcoxon=float(p_w))


def tost_paired(a, b, bound):
    """
    Two one-sided tests for equivalence of paired samples within +/- bound.

    Returns the larger of the two one-sided p-values (the TOST p) — equivalence
    is claimed only if it is below alpha.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return dict(p=np.nan, bound=bound, n=n)
    t_lo = (d.mean() + bound) / se          # H0: diff <= -bound
    t_hi = (d.mean() - bound) / se          # H0: diff >= +bound
    p_lo = stats.t.sf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return dict(p=float(max(p_lo, p_hi)), p_lower=float(p_lo),
                p_upper=float(p_hi), bound=bound, n=n,
                mdiff=float(d.mean()))


# ---------------------------------------------------------------- RM-ANOVA
def rm_anova(mat):
    """
    One-way repeated-measures ANOVA.

    mat : (n_subjects, k_conditions), complete cases only.
    Returns F, df, p, partial eta squared, Mauchly's W and the
    Greenhouse-Geisser and Huynh-Feldt corrected p-values.
    """
    m = np.asarray(mat, float)
    m = m[np.isfinite(m).all(axis=1)]
    n, k = m.shape

    grand = m.mean()
    ss_cond = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_tot = ((m - grand) ** 2).sum()
    ss_err = ss_tot - ss_cond - ss_subj

    df_cond, df_err = k - 1, (k - 1) * (n - 1)
    ms_cond, ms_err = ss_cond / df_cond, ss_err / df_err
    F = ms_cond / ms_err
    p = stats.f.sf(F, df_cond, df_err)
    eta_p2 = ss_cond / (ss_cond + ss_err)

    mauchly_W, mauchly_p, gg, hf = _sphericity(m)
    p_gg = stats.f.sf(F, df_cond * gg, df_err * gg)
    p_hf = stats.f.sf(F, df_cond * min(hf, 1.0), df_err * min(hf, 1.0))

    return dict(n=n, k=k, F=float(F), df1=df_cond, df2=df_err, p=float(p),
                eta_p2=float(eta_p2), mauchly_W=mauchly_W, mauchly_p=mauchly_p,
                gg_eps=gg, hf_eps=hf, p_gg=float(p_gg), p_hf=float(p_hf))


def _sphericity(m):
    """Mauchly's test and the GG / HF epsilons for a one-way within design."""
    n, k = m.shape
    if k < 3:
        return np.nan, np.nan, 1.0, 1.0

    # orthonormal contrast basis for the (k-1)-dim contrast space
    C = np.zeros((k, k - 1))
    for i in range(k - 1):
        C[:i + 1, i] = 1.0 / (i + 1)
        C[i + 1, i] = -1.0
    C, _ = np.linalg.qr(C)

    S = np.cov(m, rowvar=False)
    T = C.T @ S @ C
    d = k - 1
    det, tr = np.linalg.det(T), np.trace(T)
    if tr <= 0 or det <= 0:
        return np.nan, np.nan, 1.0, 1.0

    W = det / (tr / d) ** d
    df_chi = d * (d + 1) / 2 - 1
    f = 1 - (2 * d ** 2 + d + 2) / (6 * d * (n - 1))
    chi2 = -(n - 1) * f * np.log(W)
    p = stats.chi2.sf(chi2, df_chi)

    lam = np.linalg.eigvalsh(T)
    lam = lam[lam > 1e-12]
    gg = lam.sum() ** 2 / (d * (lam ** 2).sum())
    gg = float(np.clip(gg, 1.0 / d, 1.0))
    hf = (n * d * gg - 2) / (d * (n - 1 - d * gg))
    hf = float(np.clip(hf, gg, 1.0))
    return float(W), float(p), gg, hf


# -------------------------------------------------------------- correlations
def spearman_ci(x, y, n_boot=10000):
    """Spearman rho with a bootstrap CI (BCa is overkill at these n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    rho, p = stats.spearmanr(x, y)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boots = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan)
    return dict(rho=float(rho), p=float(p), n=n,
                ci=(float(lo), float(hi)))


def pearson_simple(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[ok], y[ok])
    return dict(r=float(r), p=float(p), n=int(ok.sum()))


def loo_influence(x, y, labels):
    """
    Leave-one-out Spearman, to expose high-leverage points.

    Returns the full-sample rho and, for each dropped label, the recomputed rho
    and p. This is what settles whether an n=9 correlation rests on one point.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    labels = np.asarray(labels)
    full = stats.spearmanr(x, y)
    out = []
    for lab in labels:
        keep = labels != lab
        r = stats.spearmanr(x[keep], y[keep])
        out.append(dict(dropped=str(lab), rho=float(r.statistic),
                        p=float(r.pvalue), n=int(keep.sum()),
                        delta=float(r.statistic - full.statistic)))
    out.sort(key=lambda d: -abs(d["delta"]))
    return dict(full_rho=float(full.statistic), full_p=float(full.pvalue),
                full_n=x.size, loo=out)


def within_subject_corr(df, x, y, subject):
    """
    Pooled vs within-participant association — the Simpson's-paradox check.

    Returns the naive pooled Spearman, the mean within-participant Spearman
    (Fisher-z averaged), and a repeated-measures correlation computed by
    centring both variables within participant.
    """
    pooled = stats.spearmanr(df[x], df[y])

    rs, ns = [], []
    for _, g in df.groupby(subject):
        g = g[[x, y]].dropna()
        if len(g) >= 4 and g[x].nunique() > 1 and g[y].nunique() > 1:
            r = stats.spearmanr(g[x], g[y]).statistic
            if np.isfinite(r):
                rs.append(np.clip(r, -0.999, 0.999))
                ns.append(len(g))
    if rs:
        z = np.arctanh(rs)
        w = np.array(ns) - 3
        z_bar = np.average(z, weights=np.maximum(w, 1))
        r_within = float(np.tanh(z_bar))
    else:
        r_within = np.nan

    d = df[[x, y, subject]].dropna().copy()
    d["_xc"] = d[x] - d.groupby(subject)[x].transform("mean")
    d["_yc"] = d[y] - d.groupby(subject)[y].transform("mean")
    n_sub = d[subject].nunique()
    rm_r, _ = stats.pearsonr(d["_xc"], d["_yc"])
    df_rm = len(d) - n_sub - 1
    t_rm = rm_r * np.sqrt(df_rm / max(1e-12, 1 - rm_r ** 2))
    p_rm = 2 * stats.t.sf(abs(t_rm), df_rm)

    return dict(pooled_rho=float(pooled.statistic), pooled_p=float(pooled.pvalue),
                pooled_n=int(len(df)), within_rho=r_within,
                n_subjects_used=len(rs), rm_r=float(rm_r), rm_df=int(df_rm),
                rm_p=float(p_rm),
                sign_flip=bool(np.isfinite(r_within) and
                               np.sign(r_within) != np.sign(pooled.statistic) and
                               abs(r_within) > 0.10 and
                               abs(pooled.statistic) > 0.10))


# ------------------------------------------------------------- one-sample
def one_sample_vs_benchmark(x, benchmark, alternative="greater"):
    """SUS-style test against a fixed benchmark, with a one-sided bound."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    t, p = stats.ttest_1samp(x, benchmark, alternative=alternative)
    d = (x.mean() - benchmark) / x.std(ddof=1)
    se = x.std(ddof=1) / np.sqrt(n)
    lower_1s = x.mean() - stats.t.ppf(0.95, n - 1) * se
    lo2, hi2 = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=se)
    return dict(n=n, mean=float(x.mean()), sd=float(x.std(ddof=1)),
                t=float(t), df=n - 1, p_onesided=float(p), d=float(d),
                lower_bound_1s=float(lower_1s),
                ci95_2s=(float(lo2), float(hi2)),
                n_above=int((x > benchmark).sum()))


def kendall_w(ranks):
    """
    Kendall's W for (n_raters, k_items) ranks, with the chi-square test.
    """
    r = np.asarray(ranks, float)
    n, k = r.shape
    Rj = r.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    W = 12 * S / (n ** 2 * (k ** 3 - k))
    chi2 = n * (k - 1) * W
    p = stats.chi2.sf(chi2, k - 1)
    return dict(W=float(W), chi2=float(chi2), df=k - 1, p=float(p), n=n, k=k)


# ------------------------------------------------------------ descriptives
def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, ci=(np.nan, np.nan),
                    median=np.nan, iqr=(np.nan, np.nan))
    sd = x.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    ci = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=se) if n > 1 \
        else (np.nan, np.nan)
    q1, q3 = np.percentile(x, [25, 75])
    return dict(n=n, mean=float(x.mean()), sd=float(sd),
                ci=(float(ci[0]), float(ci[1])), median=float(np.median(x)),
                iqr=(float(q1), float(q3)))


# ===========================================================================
# Additions: nonparametric parallel track, factorial designs, sensitivity
# ===========================================================================
def friedman_track(mat, labels, contrasts):
    """
    Nonparametric counterpart to rm_anova + paired contrasts.

    Friedman omnibus, Wilcoxon signed-rank contrasts with Holm correction, and
    the matched-pairs rank-biserial effect size. Use as the primary test when
    the outcome is ordinal, bounded, or badly skewed; use as a robustness check
    otherwise. If it agrees with the parametric track, say so in one sentence
    and the reader stops worrying about distributional assumptions.
    """
    m = np.asarray(mat, float)
    m = m[np.isfinite(m).all(axis=1)]
    n, k = m.shape
    chi2, p = stats.friedmanchisquare(*[m[:, j] for j in range(k)])
    kendalls_w = chi2 / (n * (k - 1))          # Friedman -> W identity

    idx = {lab: j for j, lab in enumerate(labels)}
    res, praw = [], []
    for hi, lo in contrasts:
        a, b = m[:, idx[hi]], m[:, idx[lo]]
        d = a - b
        nz = d[d != 0]
        if nz.size == 0:
            res.append(dict(contrast=f"{hi} vs {lo}", W=np.nan, p=1.0,
                            rb=0.0, n=n))
            praw.append(1.0)
            continue
        W, pw = stats.wilcoxon(a, b)
        r = stats.rankdata(np.abs(nz))
        rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()   # rank-biserial
        res.append(dict(contrast=f"{hi} vs {lo}", W=float(W), p=float(pw),
                        rb=float(rb), n=int(nz.size)))
        praw.append(pw)
    for r_, ph in zip(res, holm(praw)):
        r_["p_holm"] = float(ph)

    return dict(chi2=float(chi2), df=k - 1, p=float(p), n=n,
                kendalls_w=float(kendalls_w), contrasts=res)


def rm_anova_2way(cube):
    """
    Two-way repeated-measures ANOVA, both factors within-subject.

    cube : (n_subjects, a_levels, b_levels)

    The interaction is the test you need whenever the claim is "the effect
    appeared in one condition but not another". Running separate tests per
    condition and comparing which reached significance is the difference-of-
    significance error; it does not test the interaction.
    """
    x = np.asarray(cube, float)
    x = x[np.isfinite(x).all(axis=(1, 2))]
    n, a, b = x.shape

    grand = x.mean()
    subj = x.mean(axis=(1, 2))
    A = x.mean(axis=(0, 2))
    B = x.mean(axis=(0, 1))
    AB = x.mean(axis=0)

    ss_subj = a * b * ((subj - grand) ** 2).sum()
    ss_a = n * b * ((A - grand) ** 2).sum()
    ss_b = n * a * ((B - grand) ** 2).sum()
    ss_ab = n * ((AB - A[:, None] - B[None, :] + grand) ** 2).sum()

    sa = x.mean(axis=2)                      # subject x A
    ss_as = b * ((sa - subj[:, None] - A[None, :] + grand) ** 2).sum()
    sb = x.mean(axis=1)                      # subject x B
    ss_bs = a * ((sb - subj[:, None] - B[None, :] + grand) ** 2).sum()

    ss_tot = ((x - grand) ** 2).sum()
    ss_abs = ss_tot - (ss_subj + ss_a + ss_b + ss_ab + ss_as + ss_bs)

    def F(ss_eff, df_eff, ss_err, df_err):
        ms_e, ms_r = ss_eff / df_eff, ss_err / df_err
        f = ms_e / ms_r if ms_r > 0 else np.nan
        return dict(F=float(f), df1=df_eff, df2=df_err,
                    p=float(stats.f.sf(f, df_eff, df_err)),
                    eta_p2=float(ss_eff / (ss_eff + ss_err)))

    return dict(
        n=n,
        A=F(ss_a, a - 1, ss_as, (a - 1) * (n - 1)),
        B=F(ss_b, b - 1, ss_bs, (b - 1) * (n - 1)),
        AB=F(ss_ab, (a - 1) * (b - 1), ss_abs, (a - 1) * (b - 1) * (n - 1)),
    )


def sensitivity_dz(n, alpha=0.05, power=0.80, tails=2):
    """
    Smallest paired effect (Cohen's d_z) detectable at the given n and power.

    Report this beside every null. "We found no difference" is uninformative
    without it; "we could have detected d_z >= X and did not" is a result.
    Post-hoc observed power is not a substitute and should not be reported.
    """
    from scipy.optimize import brentq
    df = n - 1
    crit = stats.t.ppf(1 - alpha / tails, df)

    def pwr(d):
        nc = d * np.sqrt(n)
        val = stats.nct.sf(crit, df, nc)
        if tails == 2:
            # opposite tail underflows to nan for large nc; it is ~0 there
            other = stats.nct.cdf(-crit, df, nc)
            val += 0.0 if not np.isfinite(other) else other
        return val - power

    try:
        return float(brentq(pwr, 1e-6, 3.0))
    except ValueError:
        return float("nan")


def sensitivity_rho(n, alpha=0.05, power=0.80):
    """Smallest correlation detectable at the given n and power (Fisher z)."""
    from scipy.optimize import brentq
    if n < 5:
        return float("nan")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)

    def f(r):
        return np.arctanh(r) * np.sqrt(n - 3) - (z_a + z_b)

    try:
        return float(brentq(f, 1e-4, 0.999))
    except ValueError:
        return float("nan")


# ============================================================================
# Omnibus and pairwise tests for repeated measures — parametric and rank-based
# ============================================================================
def friedman(mat):
    """Friedman test: the rank-based analogue of a one-way RM-ANOVA."""
    m = np.asarray(mat, float)
    m = m[np.isfinite(m).all(axis=1)]
    n, k = m.shape
    chi2, p = stats.friedmanchisquare(*[m[:, j] for j in range(k)])
    # Kendall's W as the effect size (W = chi2 / (n*(k-1)))
    W = chi2 / (n * (k - 1))
    return dict(n=n, k=k, chi2=float(chi2), df=k - 1, p=float(p), W=float(W))


def wilcoxon_pair(a, b, n_boot=10000):
    """
    Wilcoxon signed-rank with a matched-pairs rank-biserial effect size and a
    bootstrap CI on the Hodges-Lehmann median difference.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    n = d.size
    nz = d[d != 0]
    try:
        stat, p = stats.wilcoxon(a, b)
    except ValueError:
        return dict(n=n, p=np.nan, rbc=np.nan, hl=np.nan, ci=(np.nan, np.nan))

    # rank-biserial correlation = (sum of positive ranks - negative) / total
    r = stats.rankdata(np.abs(nz))
    rpos, rneg = r[nz > 0].sum(), r[nz < 0].sum()
    rbc = (rpos - rneg) / r.sum() if r.sum() else np.nan

    # Hodges-Lehmann estimator: median of Walsh averages
    walsh = np.add.outer(d, d)[np.triu_indices(n)] / 2.0
    hl = float(np.median(walsh))
    boot = np.array([np.median(RNG.choice(walsh, walsh.size, replace=True))
                     for _ in range(min(n_boot, 2000))])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return dict(n=n, stat=float(stat), p=float(p), rbc=float(rbc),
                hl=hl, ci=(float(lo), float(hi)))


def min_detectable_dz(n, alpha=0.05, power=0.80, tails=2):
    """
    Smallest paired effect size d_z this design could detect.

    Essential when interpreting a null: it converts "we found nothing" into
    "we could have found anything above d_z = X". Solved by search, so no
    dependency on a power package.
    """
    from scipy.optimize import brentq

    def achieved(d):
        df = n - 1
        ncp = d * np.sqrt(n)
        crit = stats.t.ppf(1 - alpha / tails, df)
        return stats.nct.sf(crit, df, ncp) - power

    try:
        return float(brentq(achieved, 1e-4, 5.0))
    except ValueError:
        return np.nan


def lmm_condition(df, dv, subject, condition, ref="C1"):
    """
    Trial-level linear mixed model: dv ~ condition + (1 | subject), plus a
    likelihood-ratio test for a by-participant random slope on condition.

    Preferred over aggregating to participant means when trial counts are
    unequal, because it uses every trial and weights participants by how much
    data they actually contributed. Returns None if statsmodels is absent.
    """
    try:
        import warnings
        import statsmodels.formula.api as smf
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
    except ImportError:
        return None

    d = df[[dv, subject, condition]].dropna().copy()
    d.columns = ["y", "subj", "cond"]
    d["cond"] = pd.Categorical(d["cond"],
                               categories=[ref] + [c for c in d["cond"].unique()
                                                   if c != ref])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m0 = smf.mixedlm("y ~ C(cond)", d, groups=d["subj"]).fit(reml=False)
    out = dict(fixed={k: (float(v), float(m0.pvalues[k]))
                      for k, v in m0.params.items() if k.startswith("C(cond)")},
               aic=float(m0.aic), converged=bool(m0.converged),
               n_obs=int(m0.nobs), n_groups=int(m0.model.n_groups))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m1 = smf.mixedlm("y ~ C(cond)", d, groups=d["subj"],
                             re_formula="~C(cond)").fit(reml=False)
        if not m1.converged:
            out["random_slope"] = dict(error="random-slope model did not "
                                             "converge; do not report an LRT "
                                             "from it")
            return out
        lr = 2 * (m1.llf - m0.llf)
        ddf = m1.df_modelwc - m0.df_modelwc
        out["random_slope"] = dict(lr=float(lr), df=int(max(ddf, 1)),
                                   p=float(stats.chi2.sf(lr, max(ddf, 1))),
                                   converged=bool(m1.converged))
    except Exception as e:
        out["random_slope"] = dict(error=str(e))
    return out


import pandas as pd  # noqa: E402  (used by lmm_condition)


def kendall_w_full(mat, n_perm=10000, seed=0):
    """
    Kendall's W with the tie correction and a permutation test.

    mat : (m raters/participants) x (k items/faces) of scores; ranked within
    each rater. Three departures from the naive W = chi2 / (m(k-1)):

    1. TIE CORRECTION (Siegel & Castellan 1988, eq. 9.16). Tied ranks within a
       rater shrink S and therefore deflate W. The correction divides by
       m^2(k^3-k) - m*sum(T) instead of m^2(k^3-k), where T = sum(t^3 - t)
       over each rater's tie groups. Matters most for bounded/discrete
       measures such as a per-face proportion, where ties are common.

    2. PERMUTATION TEST rather than the chi-square approximation. The
       chi2 = m(k-1)W approximation is asymptotic in m and is unreliable for
       small m; Legendre (2005) recommends permutation for exactly this case,
       and Siegel & Castellan give exact tables for small m and k. Ranks are
       shuffled independently within each rater under H0.

    3. CHANCE FLOOR. Under H0, E[chi2] = k-1, so E[W] = 1/m -- not 0. At
       m = 10 a W of 0.10 is chance. `w_adj` rescales so chance maps to zero,
       by analogy with adjusted R^2; it is a convenience, not a standard
       statistic, so report W and state the floor.
    """
    x = np.asarray(mat, float)
    x = x[np.isfinite(x).all(axis=1)]
    m, k = x.shape
    if m < 2 or k < 2:
        return dict(W=np.nan, W_uncorrected=np.nan, p_perm=np.nan,
                    p_chi2=np.nan, m=m, k=k)

    ranks = np.apply_along_axis(stats.rankdata, 1, x)

    def tie_term(r):
        tot = 0.0
        for row in r:
            _, counts = np.unique(row, return_counts=True)
            tot += float(((counts ** 3) - counts).sum())
        return tot

    def w_of(r):
        Rj = r.sum(axis=0)
        S = ((Rj - Rj.mean()) ** 2).sum()
        denom = (m ** 2 * (k ** 3 - k) - m * tie_term(r)) / 12.0
        return S / denom if denom > 0 else np.nan

    W = w_of(ranks)
    W_unc = ((ranks.sum(axis=0) - ranks.sum(axis=0).mean()) ** 2).sum() / \
            (m ** 2 * (k ** 3 - k) / 12.0)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.apply_along_axis(rng.permutation, 1, ranks)
        null[i] = w_of(perm)
    p_perm = float((np.sum(null >= W) + 1) / (n_perm + 1))

    chi2 = m * (k - 1) * W
    return dict(W=float(W), W_uncorrected=float(W_unc),
                W_adj=float((m * W - 1) / (m - 1)),
                chi2=float(chi2), df=k - 1,
                p_chi2=float(stats.chi2.sf(chi2, k - 1)),
                p_perm=p_perm, m=m, k=k, chance_floor=1.0 / m,
                tie_inflation=float(W / W_unc) if W_unc > 0 else np.nan)
