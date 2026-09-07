#!/usr/bin/env python3
"""
srl_analysis.py
=====================================================================
Analysis library for the C0/C1/C2 SRL / shared-control soldering study.

Reads a recordings/ folder written by session_runner.py + trial_logger.py and
computes, per trial, every metric defined in the study plan:

  Behavioural (all conditions, from grid + button CSVs)
    completion_time_s, median_rt_s, error_rate, accuracy,
    median_sync_ms, pct_sync_within100, sync_valid_frac
  Robot / fluency (C2 only, from robot + events CSVs)
    reposition_correctness, repo_pos_err_mm, repo_rot_err_deg,
    functional_delay_s, robot_active_frac, robot_idle_frac,
    freeze_count, box_count, assist_time_s

Then aggregates to one value per participant x condition (mean over blocks),
merges questionnaire scores (RTLX, fluency, agency), and produces descriptive
summaries with bootstrap 95% CIs. NO significance testing here by design — this
is the extraction + descriptives + figures layer (see run_analysis.py).

Design constants and the synchronicity guards live in srl_common.py.
"""

from __future__ import annotations
import os
import glob
import json
import numpy as np
import pandas as pd

from srl_common import (parse_stem, norm_pid, FACE_TO_PAIR, N_FACES, CONDITIONS,
                        SYNC_THRESH_MS, SETTLE_ROT_DEG)


# ===========================================================================
# Discovery
# ===========================================================================
def discover_trials(recordings_dir):
    """Group every per-trial CSV by its filename stem. Returns a list of dicts:
    {participant, trial, condition, block, seq, ts, paths:{kind:path}}."""
    trials = {}
    for path in glob.glob(os.path.join(recordings_dir, "**", "*.csv"), recursive=True):
        meta = parse_stem(os.path.basename(path))
        if meta is None:
            continue
        key = (meta["pid"], meta["trial"], meta["cond"], meta["block"], meta["seq"], meta["ts"])
        rec = trials.setdefault(key, {"participant": meta["pid"], "trial": meta["trial"],
                                      "condition": meta["cond"], "block": meta["block"],
                                      "sequence": meta["seq"], "ts": meta["ts"], "paths": {}})
        rec["paths"][meta["kind"]] = path
    return sorted(trials.values(), key=lambda r: (r["participant"], r["trial"]))


# ===========================================================================
# Small array helpers for the synchronicity edge logic
# ===========================================================================
def _runs(mask):
    """List of (start, end_inclusive) index ranges of contiguous True in `mask`."""
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
    """Start index of the contiguous True run of `mask` containing index i."""
    s = i
    while s - 1 >= 0 and mask[s - 1]:
        s -= 1
    return s


# ===========================================================================
# Per-trial metric extraction
# ===========================================================================
def _behavioural(grid_path):
    g = pd.read_csv(grid_path)
    if g.empty:
        return {}
    completion = float(g["pressed_clock"].max() - g["shown_clock"].min())
    return dict(
        completion_time_s=completion,
        median_rt_s=float(g["reaction_time_s"].median()),
        error_rate=float(g["wrong_presses"].sum()) / N_FACES,
        accuracy=float((g["wrong_presses"] == 0).mean()),
        order=[int(x) for x in g["face"].tolist()],
    )


def synchronicity(buttons_path, order):
    """Guarded per-press synchronicity from the raw button stream.

    For each face: find the sustained co-press (B AND A_face); the two rising
    edges that OPEN it give the offset and which button was first. Guards:
      * other_A_high  - another pair active during the interval (attribution unsafe)
      * B_carried     - B never dropped since the previous press (no fresh edge)
    Returns per-press records and a summary. Offsets use the DEVICE clock (ms)."""
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
        s0, s1 = max(runs, key=lambda r: r[1] - r[0])     # longest run = the hold
        cursor = s1 + 1
        tA = t[_block_start(A, s0)]
        tB = t[_block_start(B, s0)]
        offset = abs(tB - tA)
        first = "B" if tB < tA else ("A" if tA < tB else "sim")
        valid, reason = True, ""
        for p2 in range(1, 9):                            # one-face check
            if p2 != p and Acol[p2][s0:s1 + 1].any():
                valid, reason = False, "other_A_high"; break
        if valid and _block_start(B, s0) <= prev_end:     # B carried from prior press
            valid, reason = False, "B_carried"
        prev_end = s1
        presses.append(dict(step=step + 1, face=face, offset_ms=float(offset),
                            first=first, valid=valid, reason=reason))
    valid_off = [x["offset_ms"] for x in presses if x["valid"] and np.isfinite(x["offset_ms"])]
    summ = dict(
        median_sync_ms=float(np.median(valid_off)) if valid_off else np.nan,
        pct_sync_within100=float(np.mean([o <= SYNC_THRESH_MS for o in valid_off]))
        if valid_off else np.nan,
        sync_valid_frac=float(np.mean([x["valid"] for x in presses])) if presses else np.nan,
    )
    return summ, presses


def _events_df(events_path):
    e = pd.read_csv(events_path, dtype=str).fillna("")
    return e


def _reposition(events_path):
    e = _events_df(events_path)
    rows = e[e["kind"] == "PAIR_AT_POSE"]
    if rows.empty:
        return {}
    corr, pe, re_ = [], [], []
    for blob in rows["data"]:
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if d.get("correct") is not None:
            corr.append(bool(d["correct"]))
        if d.get("pos_err_m") is not None:
            pe.append(float(d["pos_err_m"]) * 1000.0)
        if d.get("rot_err_deg") is not None:
            re_.append(float(d["rot_err_deg"]))
    return dict(
        reposition_correctness=float(np.mean(corr)) if corr else np.nan,
        repo_pos_err_mm=float(np.mean(pe)) if pe else np.nan,
        repo_rot_err_deg=float(np.mean(re_)) if re_ else np.nan,
    )


def _fluency(robot_path, events_path, grid_path):
    out = {}
    e = _events_df(events_path)
    out["freeze_count"] = int((e["kind"] == "FREEZE").sum())
    out["box_count"] = int((e["kind"] == "BOX_LIMIT").sum())
    if robot_path is None or not os.path.exists(robot_path):
        return out
    r = pd.read_csv(robot_path)
    if r.empty:
        return out
    tr = r["t_lsl"].to_numpy(float)
    rot = r["rot_err_deg"].to_numpy(float)
    active = (r["state"] == "ORIENT") & (r["frozen"].astype(int) == 0)
    idle = r["frozen"].astype(int) == 1
    dur = float(tr.max() - tr.min()) if len(tr) > 1 else np.nan
    out["robot_active_frac"] = float(active.mean())
    out["robot_idle_frac"] = float(idle.mean())
    out["assist_time_s"] = float(active.mean() * dur) if np.isfinite(dur) else np.nan
    # functional delay: press time - last time the arm was settled (rot < threshold)
    g = pd.read_csv(grid_path)
    settled_t = tr[rot < SETTLE_ROT_DEG]
    fds = []
    for tp in g["pressed_clock"].to_numpy(float):
        earlier = settled_t[settled_t <= tp]
        if earlier.size:
            fds.append(tp - earlier.max())
    out["functional_delay_s"] = float(np.median(fds)) if fds else np.nan
    return out


def trial_metrics(trial):
    """All metrics for one discovered trial (dict from discover_trials)."""
    paths = trial["paths"]
    m = {k: trial[k] for k in ("participant", "trial", "condition", "block", "sequence")}
    beh = _behavioural(paths["grid"]) if "grid" in paths else {}
    order = beh.pop("order", None)
    m.update(beh)
    if "buttons" in paths and order:
        summ, _ = synchronicity(paths["buttons"], order)
        m.update(summ)
    if trial["condition"] == "C2":
        if "events" in paths:
            m.update(_reposition(paths["events"]))
            m.update(_fluency(paths.get("robot"), paths["events"], paths.get("grid")))
    return m


def build_per_trial(recordings_dir):
    """DataFrame: one row per trial with all metrics + metadata."""
    rows = [trial_metrics(t) for t in discover_trials(recordings_dir)]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["participant"] = df["participant"].map(norm_pid)
        df = df.sort_values(["participant", "trial"]).reset_index(drop=True)
    return df


# ===========================================================================
# Questionnaires
# ===========================================================================
TLX_SUBS = ["tlx_mental", "tlx_physical", "tlx_temporal",
            "tlx_performance", "tlx_effort", "tlx_frustration"]


def load_questionnaires(path):
    """Load questionnaires.csv and compute Raw-TLX (mean of the 6 subscales)."""
    if path is None or not os.path.exists(path):
        return None
    q = pd.read_csv(path)
    q["participant"] = q["participant"].map(norm_pid)
    present = [c for c in TLX_SUBS if c in q.columns]
    if present:
        q["rtlx"] = q[present].mean(axis=1)
    return q


# ===========================================================================
# Aggregation + descriptives
# ===========================================================================
METRIC_COLS = [
    "completion_time_s", "median_rt_s", "error_rate", "accuracy",
    "median_sync_ms", "pct_sync_within100", "sync_valid_frac",
    "reposition_correctness", "repo_pos_err_mm", "repo_rot_err_deg",
    "functional_delay_s", "robot_active_frac", "robot_idle_frac",
    "freeze_count", "box_count", "assist_time_s",
]


def aggregate_participant_condition(per_trial, questionnaires=None, exclude_block1=False):
    """Mean each metric over blocks -> one row per participant x condition; merge
    questionnaire scores (RTLX, fluency_total, agency) if provided."""
    df = per_trial.copy()
    if exclude_block1:
        df = df[df["block"] != 1]
    metrics = [c for c in METRIC_COLS if c in df.columns]
    ppc = (df.groupby(["participant", "condition"], as_index=False)[metrics]
             .mean(numeric_only=True))
    if questionnaires is not None:
        qcols = [c for c in ["rtlx", "fluency_total", "agency"] if c in questionnaires.columns]
        ppc = ppc.merge(questionnaires[["participant", "condition"] + qcols],
                        on=["participant", "condition"], how="left")
    ppc["condition"] = pd.Categorical(ppc["condition"], categories=list(CONDITIONS), ordered=True)
    return ppc.sort_values(["participant", "condition"]).reset_index(drop=True)


def _bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=0):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def condition_descriptives(ppc):
    """Per condition x metric: n, mean, sd, median, IQR, bootstrap 95% CI of mean."""
    metrics = [c for c in METRIC_COLS + ["rtlx", "fluency_total", "agency"] if c in ppc.columns]
    out = []
    for metric in metrics:
        for cond in CONDITIONS:
            v = ppc.loc[ppc["condition"] == cond, metric].dropna().to_numpy(float)
            if v.size == 0:
                continue
            lo, hi = _bootstrap_ci(v)
            q1, q3 = (np.percentile(v, [25, 75]) if v.size > 1 else (v[0], v[0]))
            out.append(dict(metric=metric, condition=cond, n=int(v.size),
                            mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
                            median=float(np.median(v)), iqr=float(q3 - q1),
                            ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(out)
