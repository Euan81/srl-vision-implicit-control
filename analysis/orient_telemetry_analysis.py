#!/usr/bin/env python3
"""
orient_telemetry_analysis.py
=====================================================================
Evidence analysis for the C2 (robot-active) condition of the SRL study.

It answers "why did some participants do worse in C2?" by reading the robot's
own telemetry from the recorded XDF and relating it to task performance:

  * INTERSECTION MOTION  — how much the pen-line intersection the board chases
    wandered during each C2 trial (RMS about its centroid + total path length).
    A jumpy intersection means the board kept being re-commanded, so the face is
    harder to work on.  (From the `OrientTelemetry` stream / `TRIAL_SUMMARY`.)
  * PEN OUT-OF-FRAME     — how often a pen marker dropped out (frames with < 2
    pens, and the number of dropout episodes).  Dropouts freeze / retract the arm.
  * FUNCTIONAL DELAY     — robot responsiveness per face (fluency markers).
  * PERFORMANCE          — task completion time + freeze count, derived from the
    LoggerMarkers/GridControl press+STOP markers in the SAME XDF.

Everything here is C2-only: the robot only orients (and only streams telemetry)
on C2 trials, so every row is a C2 trial.

Inputs are one or more .xdf files (as written by LabRecorder alongside
session_runner_hold.py + orient_experiment.py).

Usage
    python3 orient_telemetry_analysis.py --xdf recordings/*.xdf --out c2_evidence
    python3 orient_telemetry_analysis.py --xdf recordings_dir  --out c2_evidence
    python3 orient_telemetry_analysis.py --selftest            --out c2_selftest

Outputs (under --out)
    per_c2_trial.csv        one row per C2 trial (wander, OOF, fluency, perf)
    per_participant.csv     mean over C2 trials, per participant
    correlations.csv        r / p for each wander|OOF metric vs performance
    figures/fig_wander_vs_performance.png
    figures/fig_pen_dropout_vs_performance.png
    figures/fig_participant_profile.png
    figures/fig_intersection_traces.png
"""
from __future__ import annotations
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- presentation-quality defaults (match release_analysis.py) -------------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
PART_CMAP = plt.get_cmap("tab10")

# Telemetry channel order MUST match ORIENT_TLM_CH in orient_experiment.py.
TLM_CH = ["n_pens", "penA_seen", "penB_seen", "Xint_x", "Xint_y", "Xint_z",
          "gap_mm", "tgt_x", "tgt_y", "tgt_z", "pos_err_mm", "rot_err_deg", "trial"]


# =============================================================================================
# XDF loading + stream access
# =============================================================================================
def load_xdf_streams(path):
    """Return the list of stream dicts from one .xdf (lazy pyxdf import)."""
    try:
        import pyxdf
    except Exception as e:                                          # noqa: BLE001
        raise SystemExit("pyxdf is required to read .xdf files.  Install it with:\n"
                         "    pip install pyxdf --break-system-packages\n"
                         f"(import error: {e})")
    streams, _header = pyxdf.load_xdf(path, dejitter_timestamps=True)
    return streams


def _sname(st):
    try:
        return str(st["info"]["name"][0])
    except Exception:                                              # noqa: BLE001
        return ""


def _stype(st):
    try:
        return str(st["info"]["type"][0])
    except Exception:                                              # noqa: BLE001
        return ""


def find_stream(streams, name=None, stype=None):
    for st in streams:
        if name is not None and _sname(st) != name:
            continue
        if stype is not None and _stype(st) != stype:
            continue
        return st
    return None


def marker_pairs(st):
    """(timestamp, string) list for a string-marker stream (robust to shapes)."""
    if st is None:
        return []
    ts = np.asarray(st["time_stamps"], float)
    out = []
    for t, row in zip(ts, st["time_series"]):
        if isinstance(row, (list, tuple, np.ndarray)):
            s = row[0] if len(row) else ""
        else:
            s = row
        out.append((float(t), str(s)))
    return out


# =============================================================================================
# marker parsing
# =============================================================================================
def parse_kv(msg):
    """Parse 'KEY=VALUE' tokens into a dict, numeric where possible. Tokens are separated by
    whitespace OR ';' — START markers are ';'-separated (START:3;P=7;TRIAL=14;...) while
    TRIAL_SUMMARY / fluency markers are space-separated. int_std_mm=[a,b,c] stays one token
    (no spaces or ';' inside the brackets) and is returned as a list of floats."""
    d = {}
    for tok in re.split(r"[;\s]+", msg):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if v.startswith("[") and v.endswith("]"):
            try:
                d[k] = [float(x) for x in v[1:-1].split(",") if x != ""]
            except Exception:                                     # noqa: BLE001
                d[k] = v
            continue
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    return d


def _first_int(s):
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def participant_from_path(path):
    b = os.path.basename(path)
    m = re.search(r"[Pp](\w+?)[_\.\-]", b)
    return m.group(1) if m else os.path.splitext(b)[0]


# =============================================================================================
# per-file extraction: build one row per C2 trial
# =============================================================================================
def _settle_time(times, events):
    """SETTLE TIME per face: mean seconds from each button press (PAIR) — or START for the first face —
    to the following FREEZE:1. This is the per-face cost of getting the next face ready: the robot
    reorienting to the new intersection PLUS the operator repositioning and judging it ready enough to
    lock. Averaged over the trial. (Replaces functional delay, which in a continuous-servo system is
    ~0 and meaningless — the arm starts correcting immediately.)"""
    t = np.asarray(times, float)
    ev = [str(e) for e in events]
    fz = [t[i] for i, e in enumerate(ev) if e.startswith("FREEZE:1")]
    anchors = sorted(t[i] for i, e in enumerate(ev) if e.startswith("START") or e.startswith("PAIR"))
    if not fz or not anchors:
        return np.nan
    durs = [f - [a for a in anchors if a < f][-1] for f in fz if any(a < f for a in anchors)]
    return float(np.mean(durs)) if durs else np.nan


def _freeze_to_press(times, events):
    """Mean seconds from each FREEZE:1 to the FOLLOWING button press (PAIR) — the post-freeze
    working time the operator spends completing the bimanual press once the workpiece is locked."""
    t = np.asarray(times, float)
    ev = [str(e) for e in events]
    fz = [t[i] for i, e in enumerate(ev) if e.startswith("FREEZE:1")]
    pr = sorted(t[i] for i, e in enumerate(ev) if e.startswith("PAIR"))
    durs = [min(p for p in pr if p > f) - f for f in fz if any(p > f for p in pr)]
    return float(np.mean(durs)) if durs else np.nan


def extract_file(streams, path):
    """Return (trials_df, telemetry_df) for one XDF.  trials_df = per C2 trial metrics;
    telemetry_df = the raw per-frame OrientTelemetry (for trace plots)."""
    logger = marker_pairs(find_stream(streams, name="LoggerMarkers")
                           or find_stream(streams, stype="Markers"))
    grid   = marker_pairs(find_stream(streams, name="GridControl"))
    robot  = marker_pairs(find_stream(streams, name="RobotMarkers"))

    # --- telemetry numeric stream -> tidy DataFrame ---
    tlm = find_stream(streams, name="OrientTelemetry")
    if tlm is not None and len(tlm["time_stamps"]):
        arr = np.asarray(tlm["time_series"], float)
        cols = TLM_CH[:arr.shape[1]] if arr.ndim == 2 else TLM_CH
        tdf = pd.DataFrame(arr, columns=cols)
        tdf.insert(0, "t", np.asarray(tlm["time_stamps"], float))
        tdf["file"] = os.path.basename(path)
    else:
        tdf = pd.DataFrame(columns=["t"] + TLM_CH + ["file"])

    file_part = participant_from_path(path)

    # --- START markers define C2 trial windows; STOP closes them ---
    starts = []
    for t, msg in logger:
        if msg.upper().startswith("START"):
            kv = parse_kv(msg.upper())
            starts.append({
                "t_start": t,
                "trial": int(kv.get("TRIAL")) if "TRIAL" in kv else _first_int(msg[5:]),
                "cond": kv.get("COND", "C2"),
                "block": int(kv["BLOCK"]) if "BLOCK" in kv else None,
                "participant": str(kv.get("P", file_part)),
            })
    stops = sorted(t for t, m in grid if m.upper().startswith("STOP"))
    # robot TRIAL_SUMMARY keyed by trial id
    summaries = {}
    for t, msg in robot:
        if msg.upper().startswith("TRIAL_SUMMARY"):
            kv = parse_kv(msg)
            tr = int(kv["trial"]) if str(kv.get("trial", "")).lstrip("-").isdigit() else _first_int(msg)
            summaries[tr] = (t, kv)

    rows = []
    for i, s in enumerate(starts):
        t0 = s["t_start"]
        t_next = starts[i + 1]["t_start"] if i + 1 < len(starts) else np.inf
        # first STOP strictly after t0 and before the next START -> trial end
        t1 = next((ts for ts in stops if ts > t0 and ts < t_next), None)
        if t1 is None:
            t1 = min(t_next, (tdf["t"].max() if not tdf.empty else t0 + 1e6))

        # performance from LoggerMarkers presses within (t0, t1]
        win = [(t, m) for t, m in logger if t0 < t <= t1]
        pairs   = [t for t, m in win if m.upper().startswith("PAIR")]
        freezes = [t for t, m in win if m.upper().startswith("FREEZE:1")]
        press_span = (max(pairs) - min(pairs)) if len(pairs) >= 2 else np.nan
        row = {
            "file": os.path.basename(path), "participant": s["participant"],
            "trial": s["trial"], "cond": s["cond"], "block": s["block"],
            "t_start": t0, "t_stop": t1,
            "start_stop_s": t1 - t0,
            "completion_time_s": press_span,               # participant press span (task time)
            "n_press": len(pairs), "n_freeze": len(freezes),
            "settle_time_s": _settle_time(
                [t0] + [t for t, _ in win], ["START"] + [m for _, m in win]),
            "freeze_to_press_s": _freeze_to_press(
                [t for t, _ in win], [m for _, m in win]),
        }

        # wander / OOF / fluency: prefer the controller's TRIAL_SUMMARY, else recompute from stream
        kv = summaries.get(s["trial"], (None, None))[1]
        if kv:
            std = kv.get("int_std_mm", [np.nan] * 3)
            row.update({
                "int_rms_mm": kv.get("int_rms_mm", np.nan),
                "int_std_x_mm": std[0] if len(std) > 0 else np.nan,
                "int_std_y_mm": std[1] if len(std) > 1 else np.nan,
                "int_std_z_mm": std[2] if len(std) > 2 else np.nan,
                "int_pathlen_mm": kv.get("int_pathlen_mm", np.nan),
                "int_n": kv.get("int_n", np.nan),
                "pen_oof_frames": kv.get("pen_oof_frames", np.nan),
                "pen_oof_events": kv.get("pen_oof_events", np.nan),
                "penA_miss": kv.get("penA_miss", np.nan),
                "penB_miss": kv.get("penB_miss", np.nan),
                "orient_frames": kv.get("orient_frames", np.nan),
                "oof_frac": kv.get("oof_frac", np.nan),
                "func_delay_mean_s": kv.get("func_delay_mean", np.nan),
                "present_time_mean_s": kv.get("present_time_mean", np.nan),
                "nfaces": kv.get("nfaces", np.nan),
                "src": "TRIAL_SUMMARY",
            })
        else:
            row.update(_wander_from_stream(tdf, t0, t1))
            row["src"] = "stream"
        # dynamics from the telemetry segment (LSL stream has no board pose / cmd, so those are NaN)
        seg = tdf[(tdf["t"] > t0) & (tdf["t"] <= t1)] if not tdf.empty else tdf
        if not seg.empty:
            row.update(compute_dynamics(
                seg["t"].to_numpy(float), seg[["Xint_x", "Xint_y", "Xint_z"]].to_numpy(float),
                Xboard=None, pos_err_mm=seg.get("pos_err_mm"), rot_err_deg=seg.get("rot_err_deg")))
            row.update(pen_usage_metrics(seg))         # gap from the stream (pen_angle absent -> NaN)
        rows.append(row)

    trials_df = pd.DataFrame(rows)
    return trials_df, tdf


def extract_csv_file(path):
    """Parse one flat 'orient_log_*.csv' (written directly by orient_experiment.py, no XDF).
    Returns (trials_df, telemetry_df) with the SAME schema as extract_file(), so the rest of the
    pipeline (figures, correlations) is identical. Functional delay isn't in the flat CSV, so
    func_delay_mean_s / present_time_mean_s are left NaN."""
    df = pd.read_csv(path)
    df["event"] = df.get("event", "").fillna("").astype(str)
    for c in ["t_lsl", "trial", "n_pens", "penA_seen", "penB_seen",
              "Xint_x", "Xint_y", "Xint_z", "gap_mm", "pen_angle_deg", "func_delay_s",
              "present_time_s", "board_x", "board_y", "board_z", "pos_err_mm", "rot_err_deg",
              "cmd_lin_mm_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    base = os.path.basename(path)

    # telemetry frames (drop pure event rows without a pose sample is unnecessary; trace fig filters)
    tdf = pd.DataFrame({
        "t": df["t_lsl"],
        "n_pens": df.get("n_pens"), "penA_seen": df.get("penA_seen"), "penB_seen": df.get("penB_seen"),
        "Xint_x": df.get("Xint_x"), "Xint_y": df.get("Xint_y"), "Xint_z": df.get("Xint_z"),
        "gap_mm": df.get("gap_mm")})
    tdf["file"] = base

    rows = []
    for tr, g in df.dropna(subset=["trial"]).groupby("trial"):
        g = g.sort_values("t_lsl")
        t0 = float(g["t_lsl"].min()); t1 = float(g["t_lsl"].max())
        ev = g["event"].astype(str)
        pair_t = g.loc[ev.str.startswith("PAIR"), "t_lsl"].to_numpy(float)
        n_freeze = int(ev.eq("FREEZE:1").sum())
        press_span = (pair_t.max() - pair_t.min()) if len(pair_t) >= 2 else np.nan
        part = str(g["participant"].dropna().iloc[0]) if g["participant"].notna().any() else participant_from_path(path)
        cond = str(g["cond"].dropna().iloc[0]) if "cond" in g and g["cond"].notna().any() else "C2"
        block = g["block"].dropna().iloc[0] if "block" in g and g["block"].notna().any() else None
        row = {"file": base, "participant": part, "trial": int(tr), "cond": cond,
               "block": int(block) if pd.notna(block) else None,
               "t_start": t0, "t_stop": t1, "start_stop_s": t1 - t0,
               "completion_time_s": press_span, "n_press": int(len(pair_t)), "n_freeze": n_freeze,
               "settle_time_s": _settle_time(g["t_lsl"].to_numpy(float), ev.tolist()),
               "freeze_to_press_s": _freeze_to_press(g["t_lsl"].to_numpy(float), ev.tolist())}
        row.update(_wander_from_frames(g))
        # distance-normalised wander: factor out how far apart the buttons/faces are
        tt, mib, nft = _target_travel_mm(g)
        row["interbutton_travel_mm"] = tt          # total straight-line travel the task requires
        row["interbutton_mean_mm"] = mib           # mean spacing between successive faces
        row["nfaces_tgt"] = nft
        if np.isfinite(row.get("int_pathlen_mm", np.nan)) and tt and tt > 1e-6:
            row["int_pathlen_norm"] = row["int_pathlen_mm"] / tt   # travelled ÷ required (~1 ideal)
        else:
            row["int_pathlen_norm"] = np.nan
        if np.isfinite(row.get("int_rms_mm", np.nan)) and mib and mib > 1e-6:
            row["int_rms_norm"] = row["int_rms_mm"] / mib          # RMS relative to button spacing
        else:
            row["int_rms_norm"] = np.nan
        # dynamics: mean velocity, ellipse area, SPARC, jitter attenuation, tracking lag, etc.
        _xb = (g[["board_x", "board_y", "board_z"]].to_numpy(float)
               if {"board_x", "board_y", "board_z"} <= set(g.columns) else None)
        row.update(compute_dynamics(
            g["t_lsl"].to_numpy(float), g[["Xint_x", "Xint_y", "Xint_z"]].to_numpy(float),
            Xboard=_xb, pos_err_mm=g.get("pos_err_mm"), rot_err_deg=g.get("rot_err_deg"),
            cmd_lin_mm_s=g.get("cmd_lin_mm_s")))
        row.update(pen_usage_metrics(g))               # pen convergence / parallel / skew
        # functional delay / present time are logged per FACE on the MOVE_START / READY frames
        if "func_delay_s" in g:
            fd = g["func_delay_s"].dropna()
            row["func_delay_mean_s"] = float(fd.mean()) if len(fd) else np.nan
            row["nfaces"] = int(len(fd))
        if "present_time_s" in g:
            pt = g["present_time_s"].dropna()
            row["present_time_mean_s"] = float(pt.mean()) if len(pt) else np.nan
        row["src"] = "csv"
        rows.append(row)
    return pd.DataFrame(rows), tdf


def _target_travel_mm(seg, face_change_mm=5.0):
    """Straight-line travel REQUIRED between the successive button/face targets (mm) — the
    minimum distance the workpiece has to move — plus the mean inter-button spacing.
    Uses the logged target intersection tgt_x/y/z; collapses runs of the same target so only
    genuine face CHANGES (>face_change_mm) count. Lets us express wander as a distance-normalised
    ratio (path actually travelled ÷ path required), so 'further buttons need more travel' is
    factored out. Returns (total_mm, mean_leg_mm, n_faces)."""
    cols = ["tgt_x", "tgt_y", "tgt_z"]
    if not set(cols) <= set(seg.columns):
        return np.nan, np.nan, 0
    T = seg[cols].to_numpy(float)
    T = T[np.all(np.isfinite(T), axis=1)]
    if len(T) < 2:
        return np.nan, np.nan, 0
    keep = [T[0]]
    for p in T[1:]:
        if np.linalg.norm(p - keep[-1]) * 1000.0 > face_change_mm:
            keep.append(p)
    keep = np.asarray(keep)
    if len(keep) < 2:
        return np.nan, np.nan, len(keep)
    legs = np.linalg.norm(np.diff(keep, axis=0), axis=1) * 1000.0
    return float(legs.sum()), float(legs.mean()), int(len(keep))


def _wander_from_stream(tdf, t0, t1):
    """Fallback: recompute wander + OOF from the raw OrientTelemetry stream in a window."""
    if tdf.empty:
        return _blank_wander()
    seg = tdf[(tdf["t"] > t0) & (tdf["t"] <= t1)]
    return _wander_from_frames(seg)


def _blank_wander():
    return {k: np.nan for k in
            ["int_rms_mm", "int_std_x_mm", "int_std_y_mm", "int_std_z_mm", "int_pathlen_mm",
             "int_n", "pen_oof_frames", "pen_oof_events", "penA_miss", "penB_miss",
             "orient_frames", "oof_frac", "func_delay_mean_s", "present_time_mean_s", "nfaces"]}


def _despike_mask(X, k=6.0, abs_cap_mm=1000.0):
    """Keep-mask over rows of X (N,3, metres) that drops lone tracking glitches — frames where the
    intersection teleports away and back in a single sample (a mis-track), which would otherwise
    inflate path length / RMS / velocity / jerk. A sample is a spike if the step INTO it and the
    step OUT of it are both robust outliers (> median + k*1.4826*MAD of step sizes), or if either
    step exceeds a hard non-physical cap. Genuine face-change ramps (several consecutive moderate
    steps) are NOT flagged. Returns a boolean mask aligned to X (non-finite rows already False)."""
    X = np.asarray(X, float)
    keep = np.all(np.isfinite(X), axis=1)
    idx = np.where(keep)[0]
    if len(idx) < 4:
        return keep
    d = np.linalg.norm(np.diff(X[idx], axis=0), axis=1) * 1000.0     # mm steps between kept frames
    med = np.median(d); mad = np.median(np.abs(d - med))
    thr = med + k * 1.4826 * mad if mad > 0 else np.inf
    big = (d > thr) | (d > abs_cap_mm)
    for j in range(1, len(idx) - 1):
        if big[j - 1] and big[j]:            # jumped away (step j-1) then back (step j) = isolated spike
            keep[idx[j]] = False
    # also drop an isolated huge single step at the very ends
    if len(idx) >= 2:
        if d[0] > abs_cap_mm:
            keep[idx[0]] = False
        if d[-1] > abs_cap_mm:
            keep[idx[-1]] = False
    return keep


# Trial-level extreme exclusion is applied ONLY to TIME metrics: a stalled / paused / mis-logged
# trial yields one corrupt scalar with no per-frame fix, so we guard it here. Wander / RMS / path /
# jerk are NOT trial-excluded — their lone tracking glitches are already removed per-frame by
# _despike_mask, and a genuinely high-wander trial is real signal (a struggling operator), not error.
OUTLIER_METRICS = ["completion_time_s", "settle_time_s", "freeze_to_press_s"]


def _mask_extreme(values, second_max_mult=10.0, mad_k=None):
    """Keep-mask over a 1-D column of TRIAL values. Flags a value as a likely error (high side only)
    if it exceeds second_max_mult x the next-largest remaining value — a lone spike, applied
    iteratively so several spikes are caught (the '10x the second maximum' rule). If mad_k is given,
    a robust median + mad_k*1.4826*MAD bound is added as a backstop (off by default)."""
    v = np.asarray(values, float)
    keep = np.isfinite(v)
    # iterative lone-spike rule: current max > mult x second max => the max is an error
    while keep.sum() >= 2:
        cur = np.where(keep)[0]
        s = np.sort(v[cur])
        if s[-2] > 0 and s[-1] > second_max_mult * s[-2]:
            keep[cur[np.argmax(v[cur])]] = False
        else:
            break
    if mad_k:                                     # optional robust backstop
        fin = v[keep]
        if len(fin) >= 4:
            med = np.median(fin); mad = np.median(np.abs(fin - med))
            if mad > 0:
                keep &= ~(np.isfinite(v) & (v > med + mad_k * 1.4826 * mad))
    return keep


def _clean_trial_outliers(df, cols=OUTLIER_METRICS, verbose=True):
    """NaN out trial-level TIME values flagged by _mask_extreme (10x-second-max). Returns df (copy)."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        keep = _mask_extreme(col.to_numpy(float))
        drop = np.isfinite(col.to_numpy(float)) & ~keep
        if drop.any():
            if verbose:
                bad = col[drop]
                print("[outlier] %-20s dropped %d trial value(s) as errors: %s"
                      % (c, int(drop.sum()), ", ".join("%.1f" % x for x in bad)))
            df.loc[drop, c] = np.nan
    return df


# A pen is only meaningfully "out of view" while the system is ACTIVELY TRYING to track it. During a
# FREEZE the operator has parked the pens to press the buttons; button_hold is the post-press pause;
# slow_home is the give-up retreat. Counting those as dropouts inflates the fraction, so OOF / wander
# are computed over active-tracking frames only (watch = seeking pens, tracking = servoing, and
# pen_lost_hold = a genuine loss while tracking).
ACTIVE_TRACK_PHASES = {"watch", "tracking", "pen_lost_hold"}


def _wander_from_frames(seg):
    """Compute wander (RMS / path length) + pen out-of-frame counts from a set of telemetry
    frames (a DataFrame with n_pens, penA_seen, penB_seen, Xint_x/y/z). If a per-frame 'phase'
    column is present, only ACTIVE_TRACK_PHASES frames are used (frozen / button-hold / slow-home
    are excluded, since the pens are deliberately away then, not lost)."""
    out = _blank_wander()
    if not seg.empty and "phase" in seg.columns:
        seg = seg[seg["phase"].astype(str).str.lower().isin(ACTIVE_TRACK_PHASES)]
    seg = seg[np.isfinite(seg["n_pens"].to_numpy(float))] if not seg.empty else seg
    if seg.empty:
        return out
    npens = seg["n_pens"].to_numpy(float)
    out["orient_frames"] = int(len(seg))
    out["pen_oof_frames"] = int(np.sum(npens < 2))
    full = npens >= 2
    out["pen_oof_events"] = int(np.sum((~full) & np.concatenate([[False], full[:-1]])))
    out["penA_miss"] = int(np.sum(seg["penA_seen"].to_numpy(float) < 0.5))
    out["penB_miss"] = int(np.sum(seg["penB_seen"].to_numpy(float) < 0.5))
    out["oof_frac"] = out["pen_oof_frames"] / max(1, out["orient_frames"])
    X = seg[["Xint_x", "Xint_y", "Xint_z"]].to_numpy(float)
    X = X[np.all(np.isfinite(X), axis=1)]
    X = X[_despike_mask(X)]                     # drop lone tracking-glitch teleports
    if len(X):
        out["int_n"] = int(len(X))
        c = X.mean(axis=0)
        out["int_rms_mm"] = float(np.sqrt(((X - c) ** 2).sum(axis=1).mean())) * 1000.0
        sd = X.std(axis=0) * 1000.0
        out["int_std_x_mm"], out["int_std_y_mm"], out["int_std_z_mm"] = map(float, sd)
        out["int_pathlen_mm"] = float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum()) * 1000.0
    return out


# =============================================================================================
# dynamics metrics (velocity, 95% ellipse, SPARC smoothness, jitter attenuation, tracking lag)
# NOTE ON SAMPLING RATE: the camera runs ~5-7 fps, so the Nyquist limit is ~2.5-3.5 Hz.
# Physiological tremor (6-15 Hz) is therefore UNOBSERVABLE / aliased at this rate — so there is
# deliberately NO tremor-band spectral metric here. Everything below is valid for the gross,
# low-frequency motion the camera can actually resolve.
# =============================================================================================
def _resample_uniform(t, X):
    """Interpolate irregularly-timed samples onto a uniform grid at fs = 1/median(dt).
    Returns (fs, t_uniform, X_uniform) or (None, None, None) if too few samples."""
    t = np.asarray(t, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    good_t = np.isfinite(t)
    t, X = t[good_t], X[good_t]
    if len(t) < 4:
        return None, None, None
    order = np.argsort(t); t, X = t[order], X[order]
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None, None, None
    fs = 1.0 / dt
    tu = np.arange(t[0], t[-1], dt)
    if len(tu) < 4:
        return None, None, None
    Xu = np.empty((len(tu), X.shape[1]))
    for j in range(X.shape[1]):
        col = X[:, j]; ok = np.isfinite(col)
        Xu[:, j] = np.interp(tu, t[ok], col[ok]) if ok.sum() >= 2 else np.nan
    return fs, tu, Xu


def _speed_profile(Xu, fs):
    """Speed magnitude (m/s) of a uniformly sampled position track."""
    if Xu is None or len(Xu) < 3 or not np.all(np.isfinite(Xu)):
        return None
    return np.linalg.norm(np.gradient(Xu, axis=0), axis=1) * fs


def sparc(speed, fs, fc=10.0, amp_th=0.05, padlevel=4):
    """Spectral ARC length smoothness (Balasubramanian, Melendez-Calderon & Burdet 2012).
    More negative = less smooth. Operates on a speed profile. fc is capped to <Nyquist here."""
    if speed is None or len(speed) < 4 or np.all(speed == 0):
        return np.nan
    fc = min(fc, 0.49 * fs)                                   # can't exceed Nyquist at this fps
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    if Mf.max() <= 0:
        return np.nan
    Mf = Mf / Mf.max()
    inx = np.where(f <= fc)[0]
    if len(inx) < 3:
        return np.nan
    f_sel, Mf_sel = f[inx], Mf[inx]
    above = np.where(Mf_sel >= amp_th)[0]
    if len(above) < 2:
        return np.nan
    f_sel = f_sel[above[0]:above[-1] + 1]; Mf_sel = Mf_sel[above[0]:above[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    dM = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))


def ellipse_area_95(X):
    """95% confidence-ellipse area (mm^2) of a point cloud, on its 2 principal axes
    (posturography-style sway area). chi2(0.95, 2 dof) = 5.991."""
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 3:
        return np.nan
    C = np.cov((X - X.mean(0)).T)
    w = np.sort(np.clip(np.linalg.eigvalsh(C), 0, None))[::-1][:2]
    return float(np.pi * 5.991 * np.sqrt(w[0] * w[1])) * 1e6   # m^2 -> mm^2


def xcorr_lag_ms(a, b, fs):
    """Lag (ms) of b relative to a at peak cross-correlation. Positive = b lags a
    (e.g. board lags the intersection command)."""
    if a is None or b is None or len(a) != len(b) or len(a) < 4:
        return np.nan
    a = np.nan_to_num(a - np.mean(a)); b = np.nan_to_num(b - np.mean(b))
    if np.all(a == 0) or np.all(b == 0):
        return np.nan
    n = len(a)
    corr = np.correlate(b, a, mode="full")
    lag = np.arange(-n + 1, n)[int(np.argmax(corr))]
    return float(lag) / fs * 1000.0


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # NumPy 2 renamed trapz->trapezoid


INT_FASTJUMP_MM_S = 200.0     # intersection speed above this = a "fast jump" the robot shouldn't chase


def _jitter_metrics(t, X, fast_mm_s=INT_FASTJUMP_MM_S, smooth_w=5):
    """How JUMPY is the ESTIMATED INTERSECTION itself — the fast, large excursions a naive controller
    would chase if it followed the raw target. This is a property of the pen-intersection ESTIMATE
    (a bad pen frame makes the target leap), sampled at the frame rate the robot actually acts on —
    NOT human physiological tremor. Computed on the RAW intersection (the spikes are the signal, so
    it must NOT be despiked).
      int_vel_p95_mm_s / int_vel_max_mm_s — 95th-percentile & peak intersection speed
      int_fastjump_events / _rate_hz      — count / per-second rate of frames faster than fast_mm_s
      int_jitter_rms_mm                   — high-frequency jitter amplitude: RMS of the intersection
                                            minus a short moving-average (i.e. the part that is NOT
                                            the intended slow follow)
      int_jerk_rms_mm_s3                  — RMS jerk (sharp accelerations) of the intersection
    Larger values = a jumpier target ⇒ the argument for rate/step-limiting the robot's follow."""
    out = {"int_vel_p95_mm_s": np.nan, "int_vel_max_mm_s": np.nan,
           "int_fastjump_events": np.nan, "int_fastjump_rate_hz": np.nan,
           "int_jitter_rms_mm": np.nan, "int_jerk_rms_mm_s3": np.nan}
    t = np.asarray(t, float); X = np.asarray(X, float)
    m = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
    t, X = t[m], X[m]
    if len(X) < 4:
        return out
    dt = np.diff(t)
    step_mm = np.linalg.norm(np.diff(X, axis=0), axis=1) * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = step_mm / dt                                  # intersection speed, mm/s
    spd = spd[np.isfinite(spd) & (dt > 0)]
    if len(spd):
        out["int_vel_p95_mm_s"] = float(np.percentile(spd, 95))
        out["int_vel_max_mm_s"] = float(spd.max())
        nfast = int(np.sum(spd > fast_mm_s))
        out["int_fastjump_events"] = nfast
        dur = t[-1] - t[0]
        out["int_fastjump_rate_hz"] = nfast / dur if dur > 0 else np.nan
    # high-frequency jitter = residual after a short moving-average low-pass (isolates fast wiggle
    # from the intended slow follow)
    w = min(smooth_w, len(X))
    if w >= 3:
        k = np.ones(w) / w
        Xs = np.column_stack([np.convolve(X[:, j], k, mode="same") for j in range(3)])
        res_mm = np.linalg.norm(X - Xs, axis=1) * 1000.0
        out["int_jitter_rms_mm"] = float(np.sqrt(np.mean(res_mm ** 2)))
    # RMS jerk of the raw intersection (sharp accelerations)
    if len(X) >= 5 and np.all(dt > 0):
        fs = 1.0 / np.median(dt)
        vel = np.gradient(X, 1.0 / fs, axis=0)
        jrk = np.gradient(np.gradient(vel, 1.0 / fs, axis=0), 1.0 / fs, axis=0)
        out["int_jerk_rms_mm_s3"] = float(np.sqrt(np.mean(np.linalg.norm(jrk, axis=1) ** 2))) * 1000.0
    return out


def compute_dynamics(t, Xint, Xboard=None, pos_err_mm=None, rot_err_deg=None, cmd_lin_mm_s=None):
    """All dynamic metrics for one trial. Xint / Xboard are (N,3) arrays (metres); the rest are
    (N,) arrays. Anything missing comes back NaN."""
    out = {"int_mean_vel_mm_s": np.nan, "int_ellipse_area_mm2": np.nan, "int_sparc": np.nan,
           "int_vel_p95_mm_s": np.nan, "int_vel_max_mm_s": np.nan, "int_fastjump_events": np.nan,
           "int_fastjump_rate_hz": np.nan, "int_jitter_rms_mm": np.nan, "int_jerk_rms_mm_s3": np.nan,
           "board_rms_mm": np.nan, "atten_ratio": np.nan, "sparc_gain": np.nan,
           "track_lag_ms": np.nan, "track_err_mean_mm": np.nan, "track_err_rot_deg": np.nan,
           "robot_active_frac": np.nan}
    t = np.asarray(t, float)
    Xi = np.asarray(Xint, float)
    fin_raw = np.all(np.isfinite(Xi), axis=1) & np.isfinite(t)
    out.update(_jitter_metrics(t[fin_raw], Xi[fin_raw]))   # intersection jitter — on the RAW target
    fin = fin_raw & _despike_mask(Xi)            # despike ONLY for the clean-movement metrics below
    if fin.sum() >= 3:
        ti, Xi_f = t[fin], Xi[fin]
        dur = ti[-1] - ti[0]
        path_mm = float(np.linalg.norm(np.diff(Xi_f, axis=0), axis=1).sum()) * 1000.0
        out["int_mean_vel_mm_s"] = path_mm / dur if dur > 0 else np.nan
        out["int_ellipse_area_mm2"] = ellipse_area_95(Xi_f)
        fs_i, _, Xu_i = _resample_uniform(ti, Xi_f)
        sp_i = _speed_profile(Xu_i, fs_i) if fs_i else None
        out["int_sparc"] = sparc(sp_i, fs_i) if fs_i else np.nan
    else:
        Xu_i = sp_i = fs_i = None
    # board vs intersection: jitter attenuation + tracking lag + smoothness gain
    if Xboard is not None:
        Xb = np.asarray(Xboard, float)
        fb = np.all(np.isfinite(Xb), axis=1) & np.isfinite(t)
        if fb.sum() >= 3:
            Xb_f = Xb[fb]
            out["board_rms_mm"] = float(np.sqrt(((Xb_f - Xb_f.mean(0)) ** 2).sum(1).mean())) * 1000.0
            if fin.sum() >= 3:
                int_rms = float(np.sqrt(((Xi[fin] - Xi[fin].mean(0)) ** 2).sum(1).mean())) * 1000.0
                if int_rms > 1e-9:
                    out["atten_ratio"] = out["board_rms_mm"] / int_rms
            # resample BOTH onto a shared grid over the overlap for lag + sparc gain
            lo = max(t[fin][0], t[fb][0]); hi = min(t[fin][-1], t[fb][-1])
            if hi > lo:
                fs2, tu2, _ = _resample_uniform(t[fin], Xi[fin])
                if fs2:
                    tu = np.arange(lo, hi, 1.0 / fs2)
                    if len(tu) >= 4:
                        Ii = np.column_stack([np.interp(tu, t[fin], Xi[fin][:, j]) for j in range(3)])
                        Bb = np.column_stack([np.interp(tu, t[fb], Xb[fb][:, j]) for j in range(3)])
                        spi = _speed_profile(Ii, fs2); spb = _speed_profile(Bb, fs2)
                        out["track_lag_ms"] = xcorr_lag_ms(spi, spb, fs2)
                        sg_b = sparc(spb, fs2); sg_i = sparc(spi, fs2)
                        if np.isfinite(sg_b) and np.isfinite(sg_i):
                            out["sparc_gain"] = sg_b - sg_i          # >0 = board smoother than intent
    if pos_err_mm is not None:
        v = np.asarray(pos_err_mm, float); out["track_err_mean_mm"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    if rot_err_deg is not None:
        v = np.asarray(rot_err_deg, float); out["track_err_rot_deg"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    if cmd_lin_mm_s is not None:
        v = np.asarray(cmd_lin_mm_s, float); v = v[np.isfinite(v)]
        if len(v):
            out["robot_active_frac"] = float(np.mean(v > 1.0))       # >1 mm/s = commanding motion
    return out


# =============================================================================================
# pen-usage metrics — is the operator actually CONVERGING the pens on a point, or holding them
# parallel / skew (out-of-plane)?  gap_mm = distance between the two pen LINES (intersection sits
# gap/2 from each line); pen_angle_deg = acute angle between the pens.
# =============================================================================================
PEN_CONVERGE_MM = 15.0    # gap below this = pens genuinely meeting on a point ("uses the intersection")
PEN_PARALLEL_DEG = 20.0   # pen angle below this = near-parallel, intersection ill-defined


def pen_usage_metrics(df):
    out = {"pen_gap_med_mm": np.nan, "pen_gap_mean_mm": np.nan, "pen_angle_mean_deg": np.nan,
           "frac_converged": np.nan, "frac_parallel": np.nan}
    if "gap_mm" in df:
        g = pd.to_numeric(df["gap_mm"], errors="coerce").dropna()
        if len(g):
            out["pen_gap_med_mm"] = float(g.median())
            out["pen_gap_mean_mm"] = float(g.mean())
            out["frac_converged"] = float((g < PEN_CONVERGE_MM).mean())   # fraction of frames on-point
    if "pen_angle_deg" in df:
        a = pd.to_numeric(df["pen_angle_deg"], errors="coerce").dropna()
        if len(a):
            out["pen_angle_mean_deg"] = float(a.mean())
            out["frac_parallel"] = float((a < PEN_PARALLEL_DEG).mean())   # fraction of frames near-parallel
    return out


# =============================================================================================
# statistics
# =============================================================================================
def corr_with_p(x, y, method="pearson", n_perm=20000, seed=0):
    """Pearson or Spearman r with a permutation two-sided p-value (dependency-free).
    Returns (r, p, n)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan, n)
    if method == "spearman":
        x = pd.Series(x).rank().to_numpy(); y = pd.Series(y).rank().to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = 1
    for _ in range(n_perm):
        rp = float(np.corrcoef(x, rng.permutation(y))[0, 1])
        if abs(rp) >= abs(r) - 1e-12:
            cnt += 1
    return (r, cnt / (n_perm + 1), n)


# =============================================================================================
# figures
# =============================================================================================
def _scatter_reg(ax, df, xcol, ycol, xlabel, ylabel):
    if xcol not in df.columns or ycol not in df.columns:
        ax.set_visible(False); return None
    sub = df[[xcol, ycol, "participant"]].dropna()
    if len(sub) < 2:
        ax.set_visible(False); return None
    parts = sorted(sub["participant"].unique())
    for i, p in enumerate(parts):
        s = sub[sub["participant"] == p]
        ax.scatter(s[xcol], s[ycol], s=40, alpha=0.8, color=PART_CMAP(i % 10),
                   edgecolor="white", linewidth=0.6, label=f"P{p}", zorder=3)
    x = sub[xcol].to_numpy(float); y = sub[ycol].to_numpy(float)
    if np.std(x) > 0:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "-", color="#333", linewidth=2, zorder=2)
    r, p, n = corr_with_p(x, y, "pearson")
    rs, ps, _ = corr_with_p(x, y, "spearman")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"r={r:+.2f} (p={p:.3f}, n={n})\nSpearman ρ={rs:+.2f} (p={ps:.3f})",
                 fontsize=10, loc="left")
    return (xcol, ycol, r, p, rs, ps, n)


def fig_wander_vs_performance(trials, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    stats = []
    stats.append(_scatter_reg(axes[0], trials, "int_rms_mm", "completion_time_s",
                              "Intersection RMS wander (mm)", "Completion time (s)"))
    stats.append(_scatter_reg(axes[1], trials, "int_pathlen_mm", "completion_time_s",
                              "Intersection path length (mm)", "Completion time (s)"))
    stats.append(_scatter_reg(axes[2], trials, "int_rms_mm", "settle_time_s",
                              "Intersection RMS wander (mm)", "Settle time (s)"))
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(10, len(l)), fontsize=8,
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Intersection instability vs. C2 performance (each point = one C2 trial)",
                 y=1.06, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_wander_vs_performance.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, [s for s in stats if s]


def fig_dropout_vs_performance(trials, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    stats = []
    stats.append(_scatter_reg(axes[0], trials, "pen_oof_events", "completion_time_s",
                              "Pen dropout episodes (count)", "Completion time (s)"))
    stats.append(_scatter_reg(axes[1], trials, "oof_frac", "completion_time_s",
                              "Frames with a pen out of view (fraction)", "Completion time (s)"))
    fig.suptitle("Pen marker dropouts vs. C2 performance (each point = one C2 trial)",
                 y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_pen_dropout_vs_performance.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, [s for s in stats if s]


def fig_participant_profile(trials, figdir):
    """Per-participant means: who had the noisiest intersection / most dropouts / slowest trials."""
    g = trials.groupby("participant", as_index=False).agg(
        rms=("int_rms_mm", "mean"), path=("int_pathlen_mm", "mean"),
        oof=("oof_frac", "mean"), ct=("completion_time_s", "mean"),
        fd=("settle_time_s", "mean"), n=("trial", "count"))
    if g.empty:
        return None
    g = g.sort_values("ct", ascending=True)
    parts = g["participant"].astype(str).to_list()
    xpos = np.arange(len(parts))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, col, ttl, ylab in [
        (axes[0], "rms", "Mean intersection RMS wander", "mm"),
        (axes[1], "oof", "Mean pen-out-of-frame fraction", "fraction"),
        (axes[2], "ct", "Mean completion time", "s")]:
        cols = [PART_CMAP(i % 10) for i in range(len(parts))]
        ax.bar(xpos, g[col].to_numpy(float), color=cols, edgecolor="white")
        ax.set_xticks(xpos); ax.set_xticklabels([f"P{p}" for p in parts], rotation=0)
        ax.set_title(ttl, fontsize=11); ax.set_ylabel(ylab)
    fig.suptitle("Per-participant C2 profile (sorted by completion time)",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_participant_profile.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_assistance(trials, figdir):
    """Does the SRL STEADY the workpiece? board RMS vs intersection RMS (points below y=x are
    attenuated), the attenuation-ratio distribution, and mean-velocity vs completion time."""
    if not {"board_rms_mm", "atten_ratio"}.issubset(trials.columns) \
            or trials["atten_ratio"].notna().sum() == 0:
        return None, []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    stats = []
    # A: board RMS vs intersection RMS with unity line
    sub = trials[["int_rms_mm", "board_rms_mm", "participant"]].dropna()
    if len(sub) >= 1:
        for i, p in enumerate(sorted(sub["participant"].unique())):
            s = sub[sub["participant"] == p]
            axes[0].scatter(s["int_rms_mm"], s["board_rms_mm"], s=40, alpha=0.8,
                            color=PART_CMAP(i % 10), edgecolor="white", linewidth=0.6)
        m = float(np.nanmax([sub["int_rms_mm"].max(), sub["board_rms_mm"].max()]))
        axes[0].plot([0, m], [0, m], "--", color="#888", linewidth=1.5)
        axes[0].set_xlabel("Intersection RMS (mm)"); axes[0].set_ylabel("Board RMS (mm)")
        axes[0].set_title("Board vs intent wander\n(below dashed = SRL steadies it)", fontsize=10, loc="left")
    # B: attenuation-ratio distribution with reference at 1
    v = trials["atten_ratio"].dropna().to_numpy(float)
    axes[1].boxplot([v], widths=0.5, showmeans=True, medianprops=dict(color="black"))
    jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.15
    axes[1].scatter(np.ones(v.size) + jit, v, s=30, color="#4C78A8", edgecolor="white", zorder=3)
    axes[1].axhline(1.0, color="#E45756", linestyle="--", linewidth=1.5)
    axes[1].set_xticks([1]); axes[1].set_xticklabels(["C2"])
    axes[1].set_ylabel("attenuation ratio  (board RMS / intersection RMS)")
    axes[1].set_title(f"Jitter attenuation  (median {np.median(v):.2f})\n<1 = damps operator wander",
                      fontsize=10, loc="left")
    # C: mean intersection velocity vs completion time
    stats.append(_scatter_reg(axes[2], trials, "int_mean_vel_mm_s", "completion_time_s",
                              "Intersection mean velocity (mm/s)", "Completion time (s)"))
    fig.suptitle("Assistance quality — does the robot steady the workpiece?",
                 y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_assistance.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, [s for s in stats if s]


def _labelled_scatter(ax, m, xcol, xlabel, ylabel):
    """Participant-level scatter (labelled points + OLS fit + Pearson/Spearman title). m has columns
    xcol, 'y', 'pid'. Returns a stats row or None."""
    m = m.dropna(subset=[xcol, "y"])
    if len(m) < 2:
        ax.set_visible(False); return None
    x = m[xcol].to_numpy(float); y = m["y"].to_numpy(float)
    for k, (_, r) in enumerate(m.iterrows()):
        ax.scatter(r[xcol], r["y"], s=60, color=PART_CMAP(k % 10),
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(f"P{r['pid']}", (r[xcol], r["y"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    if np.std(x) > 0:
        b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, "-", color="#333", linewidth=2, zorder=2)
    rr, pp, nn = corr_with_p(x, y, "pearson"); rs, ps, _ = corr_with_p(x, y, "spearman")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"r={rr:+.2f} (p={pp:.3f}, n={nn})  ρ={rs:+.2f} (p={ps:.3f})", loc="left", fontsize=9)
    return (f"{xcol} vs {ylabel.split('[')[0].strip()} (participant)", "", rr, pp, rs, ps, nn)


def _grid_vs_outcomes(trials, figdir, preds, fname, suptitle, sync_csv=None, singleton_csv=None):
    """Generic participant-level grid: each predictor (row) vs every outcome (column). Outcomes are
    the shared set — completion time, settle time, time-after-freeze (telemetry) + bimanual offset
    (synchronicity) and singleton rate (singularities) from the behavioural CSVs. `preds` is a list
    of (column, x-label, participant-aggregation 'mean'|'median')."""
    preds = [(c, lab, how) for c, lab, how in preds
             if c in trials.columns and trials[c].notna().sum() > 0]
    if not preds:
        return None, []
    g = trials.groupby("participant")
    base = pd.DataFrame({"participant": list(g.groups.keys())})
    base["pid"] = base["participant"].map(_pid_key)
    for c, _, how in preds:
        agg = g[c].median() if how == "median" else g[c].mean()
        base[c] = base["participant"].map(agg)
    outcomes = []
    for c, lab, how in [("completion_time_s", "Completion time (s)", "mean"),
                        ("settle_time_s", "Settle time (s)", "mean"),
                        ("freeze_to_press_s", "Time after freeze (s)", "mean")]:
        if c in trials.columns and trials[c].notna().sum() > 0:
            agg = g[c].median() if how == "median" else g[c].mean()
            base[c] = base["participant"].map(agg)
            outcomes.append((c, lab))
    sync = _load_behavior_c2(sync_csv, "median_sync_ms")
    singl = _load_behavior_c2(singleton_csv, "singleton_rate")
    if sync:
        base["sync_ms"] = base["pid"].map(sync); outcomes.append(("sync_ms", "Bimanual offset (ms) [lower=better]"))
    if singl:
        base["singleton_rate"] = base["pid"].map(singl); outcomes.append(("singleton_rate", "Singleton rate [lower=better]"))
    if not outcomes:
        return None, []
    nrow, ncol = len(preds), len(outcomes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.5 * nrow), squeeze=False)
    stats = []
    for i, (pcol, plabel, _) in enumerate(preds):
        for j, (ycol, ylabel) in enumerate(outcomes):
            m = base[[pcol, ycol, "pid"]].rename(columns={ycol: "y"})
            s = _labelled_scatter(axes[i][j], m, pcol, plabel, ylabel)
            if s:
                stats.append(s)
    fig.suptitle(suptitle, y=1.005, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, fname)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, [s for s in stats if s]


def fig_wander_ratio(trials, figdir, sync_csv=None, singleton_csv=None):
    """Distance-normalised wander (path ratio + RMS/spacing) vs every outcome, participant-level —
    button spacing factored out so 'further buttons need more travel' is not confounded."""
    return _grid_vs_outcomes(
        trials, figdir,
        [("int_pathlen_norm", "path ratio (travelled / required)", "mean"),
         ("int_rms_norm", "RMS wander / button spacing", "mean")],
        "fig_wander_ratio.png",
        "Distance-normalised wander vs all outcomes (participant-level; button spacing factored out)",
        sync_csv, singleton_csv)


def fig_intersection_distance_vs_outcomes(trials, figdir, sync_csv=None, singleton_csv=None):
    """AVERAGE INTERSECTION DISTANCE (mean pen line-to-line gap: how far the two pen lines pass from
    each other; the estimated intersection sits half this from each line) vs every outcome —
    completion time, synchronicity (bimanual offset) and singularities (singleton rate), plus settle
    and freeze→press. A larger gap means the operator converged the pens less well on a point, so the
    intended target was noisier ⇒ expect worse (higher) outcomes."""
    return _grid_vs_outcomes(
        trials, figdir,
        [("pen_gap_mean_mm", "average intersection distance (line-to-line gap, mm)", "mean")],
        "fig_intersection_distance_vs_outcomes.png",
        "Average intersection distance vs all outcomes (participant-level)",
        sync_csv, singleton_csv)


def fig_dropout_vs_coordination(trials, figdir, sync_csv=None, singleton_csv=None):
    """Pen out-of-frame (now active-tracking frames only) vs every outcome, participant-level:
    does losing the pen markers cost coordination (higher bimanual offset, more singletons) as well
    as time? Predictors = out-of-view fraction and dropout-episode count."""
    return _grid_vs_outcomes(
        trials, figdir,
        [("oof_frac", "pen out-of-view fraction (tracking only)", "mean"),
         ("pen_oof_events", "pen dropout episodes (count)", "mean")],
        "fig_dropout_vs_coordination.png",
        "Pen dropouts vs all outcomes (participant-level; out-of-view = active-tracking frames only)",
        sync_csv, singleton_csv)


def fig_jitter(trials, figdir):
    """How JUMPY is the estimated intersection (the target the robot follows) — and does the robot
    chase it? Jitter is a property of the pen-intersection ESTIMATE, not human tremor.
      A: high-frequency jitter amplitude per participant (RMS of intersection minus its slow follow)
      B: fast-jump rate per participant (frames faster than the INT_FASTJUMP_MM_S threshold)
      C: intersection jitter vs board wander — does the arm follow the fast bad movements?"""
    if "int_jitter_rms_mm" not in trials.columns or trials["int_jitter_rms_mm"].notna().sum() == 0:
        return None, []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    stats = []
    # A: HF jitter amplitude per participant
    g = trials.groupby("participant", as_index=False).agg(
        j=("int_jitter_rms_mm", "mean")).dropna().sort_values("j")
    if len(g):
        parts = g["participant"].astype(str).to_list(); xpos = np.arange(len(parts))
        axes[0].bar(xpos, g["j"].to_numpy(float),
                    color=[PART_CMAP(i % 10) for i in range(len(parts))], edgecolor="white")
        axes[0].set_xticks(xpos); axes[0].set_xticklabels([f"P{p}" for p in parts])
        axes[0].set_ylabel("high-frequency jitter RMS (mm)")
        axes[0].set_title("Intersection jitter per participant\n(fast wiggle, not the intended follow)",
                          fontsize=10, loc="left")
    # B: fast-jump rate per participant
    if trials["int_fastjump_rate_hz"].notna().sum():
        g2 = trials.groupby("participant", as_index=False).agg(
            r=("int_fastjump_rate_hz", "mean")).dropna().sort_values("r")
        parts = g2["participant"].astype(str).to_list(); xpos = np.arange(len(parts))
        axes[1].bar(xpos, g2["r"].to_numpy(float),
                    color=[PART_CMAP(i % 10) for i in range(len(parts))], edgecolor="white")
        axes[1].set_xticks(xpos); axes[1].set_xticklabels([f"P{p}" for p in parts])
        axes[1].set_ylabel(f"fast-jump rate (events/s, >{INT_FASTJUMP_MM_S:.0f} mm/s)")
        axes[1].set_title("How often the target lurches\n(movements the robot should not chase)",
                          fontsize=10, loc="left")
    else:
        axes[1].set_visible(False)
    # C: does the robot chase the jitter? intersection HF jitter vs board RMS
    if {"board_rms_mm"}.issubset(trials.columns) and trials["board_rms_mm"].notna().sum():
        sub = trials[["int_jitter_rms_mm", "board_rms_mm", "participant"]].dropna()
        if len(sub) >= 2:
            for i, pp in enumerate(sorted(sub["participant"].unique())):
                s = sub[sub["participant"] == pp]
                axes[2].scatter(s["int_jitter_rms_mm"], s["board_rms_mm"], s=40, alpha=0.8,
                                color=PART_CMAP(i % 10), edgecolor="white", linewidth=0.6)
            m = float(np.nanmax([sub["int_jitter_rms_mm"].max(), sub["board_rms_mm"].max()]))
            axes[2].plot([0, m], [0, m], "--", color="#888", linewidth=1.5)
            axes[2].set_xlabel("intersection HF jitter RMS (mm)")
            axes[2].set_ylabel("board RMS (mm)")
            axes[2].set_title("Does the arm chase jitter?\n(below dashed = it does NOT follow)",
                              fontsize=10, loc="left")
            r, pv, n = corr_with_p(sub["int_jitter_rms_mm"], sub["board_rms_mm"], "pearson")
            stats.append(("int_jitter_rms_mm", "board_rms_mm", r, pv, np.nan, np.nan, n))
        else:
            axes[2].set_visible(False)
    else:
        axes[2].set_visible(False)
    fig.suptitle("Intersection jitter — the jumpy target, and whether the robot follows it",
                 y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_jitter.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, [s for s in stats if s]


OOF_FIGSIZE = (4.8, 4.8)   # shared size so fig_oof_burden and the oof_vs_* scatters publish together


def fig_oof_burden(trials, figdir):
    """How often a pen marker was out of view WHEN IT SHOULDN'T BE — i.e. during ACTIVE TRACKING only
    (freeze / button-hold / slow-home already excluded). Makes the tracking-loss burden explicit: a
    large fraction of the frames where the robot was trying to follow the pens had a marker missing,
    far above a workable level. This is the bottleneck behind the settle-time / completion-time cost."""
    if "oof_frac" not in trials.columns or trials["oof_frac"].notna().sum() == 0:
        return None
    # one value PER PARTICIPANT (mean over their C2 trials)
    g = (trials.groupby("participant", as_index=False)
               .agg(oof=("oof_frac", "mean")).dropna())
    v = g["oof"].to_numpy(float)
    if v.size == 0:
        return None
    fig, ax = plt.subplots(figsize=OOF_FIGSIZE)
    ax.boxplot([v], widths=0.5, showmeans=True, medianprops=dict(color="black"))
    jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.15
    for i, (_, r) in enumerate(g.iterrows()):
        ax.scatter(1 + jit[i], r["oof"], s=70, color=PART_CMAP(i % 10),
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks([1]); ax.set_xticklabels(["C2"]); ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of active-tracking frames with a pen out of view")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_oof_burden.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    _write_oof_caption(os.path.join(figdir, "fig_oof_burden.tex"),
                       n=v.size, med=float(np.median(v)), mean=float(v.mean()),
                       lo=float(v.min()), hi=float(v.max()))
    return p


def _write_oof_caption(path, n, med, mean, lo, hi):
    """Write the LaTeX figure float + caption for fig_oof_burden with the ACTUAL computed values."""
    tex = (
        "%% Auto-generated by orient_telemetry_analysis.py — do not hand-edit values.\n"
        "%% Requires \\usepackage{graphicx}. Adjust width / path to your project.\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.42\\linewidth]{figures/fig_oof_burden.png}\n"
        "  \\caption{Fraction of active-tracking frames (freeze excluded) with a pen out of view,\n"
        "    C2 (median $%.2f$, $n=%d$). Circle: one participant; box: median and IQR; triangle:\n"
        "    mean. Markers were lost about half the time the robot was tracking.}\n"
        "  \\label{fig:oof-burden}\n"
        "\\end{figure}\n" % (med, n))
    with open(path, "w") as f:
        f.write(tex)


# participants dropped from the oof/ folder analysis ONLY (all other figures keep everyone)
COMPLETION_OUTLIER_S = 80.0   # drop C2 trials slower than this from the oof/ folder (2 extreme outliers)


def _write_oof_rel_caption(path, slug, phrase, r, p, rs, ps, n, unit="trial", note=""):
    """LaTeX caption for one oof-vs-outcome scatter, with the ACTUAL correlation values. `unit` is
    the plotting unit ('trial' or 'participant')."""
    tex = (
        "%% Auto-generated by orient_telemetry_analysis.py — do not hand-edit values.\n"
        "%% Requires \\usepackage{graphicx}.\n"
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=0.42\\linewidth]{figures/oof/oof_vs_%s.png}\n"
        "  \\caption{%s versus pen out-of-view fraction (active-tracking frames only, freeze\n"
        "    excluded), one dot per %s ($n=%d$). %sPearson $r=%+.2f$ ($p=%.3f$), Spearman\n"
        "    $\\rho=%+.2f$ ($p=%.3f$). Circle: one %s; line: least-squares fit.}\n"
        "  \\label{fig:oof-%s}\n"
        "\\end{figure}\n" % (slug, phrase, unit, n, note, r, p, rs, ps, unit, slug))
    with open(path, "w") as f:
        f.write(tex)


def fig_oof_relationships(trials, figdir, sync_csv=None, singleton_csv=None):
    """Individual presentation figures (one per outcome) of pen out-of-view fraction (active-tracking
    only) vs each outcome, in <figdir>/oof/. Completion time and settle time are TRIAL-LEVEL (one dot
    per C2 trial, coloured by participant); synchronicity (bimanual offset) and singularities
    (singleton rate) only exist per participant, so they stay participant-level. Trials slower than
    COMPLETION_OUTLIER_S are dropped as extreme outliers."""
    if "oof_frac" not in trials.columns or trials["oof_frac"].notna().sum() == 0:
        return []
    oofdir = os.path.join(figdir, "oof"); os.makedirs(oofdir, exist_ok=True)
    # keep ALL trials; drop only the extreme-completion outliers (the two >80 s trials)
    t = trials.copy()
    ndrop = 0
    if "completion_time_s" in t.columns:
        bad = (t["completion_time_s"] > COMPLETION_OUTLIER_S).fillna(False)
        ndrop = int(bad.sum())
        t = t[~bad]
    note = ("%d trial%s with completion $>%.0f$\\,s excluded. "
            % (ndrop, "" if ndrop == 1 else "s", COMPLETION_OUTLIER_S)) if ndrop else ""
    # trials from the same participant share a colour (like the trial-level example figure)
    pids = list(dict.fromkeys(t["participant"].astype(str)))
    pcol = {p: PART_CMAP(i % 10) for i, p in enumerate(pids)}
    # participant-level behavioural values (no per-trial source)
    sync = _load_behavior_c2(sync_csv, "median_sync_ms")
    singl = _load_behavior_c2(singleton_csv, "singleton_rate")
    g = t.groupby("participant")
    pbase = pd.DataFrame({"participant": list(g.groups.keys())})
    pbase["pid"] = pbase["participant"].map(_pid_key)
    pbase["oof"] = pbase["participant"].map(g["oof_frac"].mean())
    if sync:
        pbase["sync_ms"] = pbase["pid"].map(sync)
    if singl:
        pbase["singleton_rate"] = pbase["pid"].map(singl)
    # (column, y-label, slug, caption phrase, level)
    outcomes = [("completion_time_s", "Completion time (s)", "completion", "Completion time", "trial"),
                ("settle_time_s", "Settle time (s)", "settle", "Settle time", "trial"),
                ("sync_ms", "Bimanual offset (ms)", "sync", "Bimanual offset (synchronicity)", "participant"),
                ("singleton_rate", "Singleton rate", "singleton",
                 "Singleton rate (one-handed presses)", "participant")]
    made, stat_rows = [], []
    for col, ylab, slug, phrase, level in outcomes:
        if level == "trial":
            if col not in t.columns:
                continue
            m = t[["participant", "oof_frac", col]].dropna()
            if len(m) < 2:
                continue
            x = m["oof_frac"].to_numpy(float); y = m[col].to_numpy(float)
            cols = [pcol[str(pp)] for pp in m["participant"]]
            size = 34
        else:
            if col not in pbase.columns:
                continue
            m = pbase[["participant", "oof", col]].dropna()
            if len(m) < 2:
                continue
            x = m["oof"].to_numpy(float); y = m[col].to_numpy(float)
            cols = [pcol.get(str(pp), PART_CMAP(0)) for pp in m["participant"]]
            size = 70
        fig, ax = plt.subplots(figsize=OOF_FIGSIZE)
        for i in range(len(m)):
            ax.scatter(x[i], y[i], s=size, color=cols[i], edgecolor="white",
                       linewidth=0.5, zorder=3, alpha=0.85)
        if np.std(x) > 0:
            b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b0 + b1 * xs, "-", color="#333", linewidth=2, zorder=2)
        ax.set_xlabel("pen out-of-view fraction (tracking only)"); ax.set_ylabel(ylab)
        if y.min() >= 0:
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        p = os.path.join(oofdir, f"oof_vs_{slug}.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        rr, pp_, nn = corr_with_p(x, y, "pearson"); rs, ps, _ = corr_with_p(x, y, "spearman")
        _write_oof_rel_caption(os.path.join(oofdir, f"oof_vs_{slug}.tex"),
                               slug, phrase, rr, pp_, rs, ps, nn, unit=level, note=note)
        made.append(p)
        stat_rows.append({"outcome": phrase, "unit": level, "n": nn, "pearson_r": rr,
                          "pearson_p": pp_, "spearman_rho": rs, "spearman_p": ps})
    if stat_rows:
        pd.DataFrame(stat_rows).to_csv(os.path.join(oofdir, "oof_correlations.csv"), index=False)
        _write_oof_table(os.path.join(oofdir, "oof_correlations.tex"), stat_rows, note=note)
    return made


def _write_oof_table(path, rows, note=""):
    """LaTeX table of the pen-out-of-view-fraction vs outcome correlations. Includes the unit
    (trial / participant) and n per row, since completion & settle are trial-level while
    synchronicity & singletons are participant-level."""
    def stars(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    body = ""
    for r in rows:
        body += ("    %s & %s & %d & %+.2f & %.3f%s & %+.2f & %.3f%s \\\\\n"
                 % (r["outcome"], r.get("unit", "trial"), r["n"],
                    r["pearson_r"], r["pearson_p"], stars(r["pearson_p"]),
                    r["spearman_rho"], r["spearman_p"], stars(r["spearman_p"])))
    tex = (
        "%% Auto-generated by orient_telemetry_analysis.py — do not hand-edit values.\n"
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\caption{Pen out-of-view fraction (active-tracking frames only) vs C2 outcomes.\n"
        "    Completion and settle time are trial-level; synchronicity and singleton rate are\n"
        "    participant-level (no per-trial value available). %s$^{*}p<.05$, $^{**}p<.01$,\n"
        "    $^{***}p<.001$.}\n"
        "  \\label{tab:oof-correlations}\n"
        "  \\begin{tabular}{llrrrrr}\n"
        "    \\hline\n"
        "    Outcome & Unit & $n$ & Pearson $r$ & $p$ & Spearman $\\rho$ & $p$ \\\\\n"
        "    \\hline\n"
        "%s"
        "    \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}\n" % (note, body))
    with open(path, "w") as f:
        f.write(tex)


def fig_pen_usage(trials, figdir):
    """Does the operator CONVERGE the pens on a point, or hold them parallel / skew?
    gap = how far the two pen lines pass from each other; angle = pen separation."""
    if "pen_gap_med_mm" not in trials.columns or trials["pen_gap_med_mm"].notna().sum() == 0:
        return None, []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    stats = []
    # A: median line-to-line gap per trial (strip) with the "converged" threshold
    v = trials["pen_gap_med_mm"].dropna().to_numpy(float)
    axes[0].boxplot([v], widths=0.5, showmeans=True, medianprops=dict(color="black"))
    jit = (np.random.default_rng(0).random(v.size) - 0.5) * 0.15
    axes[0].scatter(np.ones(v.size) + jit, v, s=30, color="#4C78A8", edgecolor="white", zorder=3)
    axes[0].axhline(PEN_CONVERGE_MM, color="#54A24B", linestyle="--", linewidth=1.5)
    axes[0].set_xticks([1]); axes[0].set_xticklabels(["C2"])
    axes[0].set_ylabel("median pen line-to-line gap (mm)")
    axes[0].set_title(f"Do the pens converge?\n<{PEN_CONVERGE_MM:.0f} mm (green) = on a point",
                      fontsize=10, loc="left")
    # B: mean pen angle per trial with the near-parallel threshold
    if trials["pen_angle_mean_deg"].notna().sum():
        a = trials["pen_angle_mean_deg"].dropna().to_numpy(float)
        axes[1].boxplot([a], widths=0.5, showmeans=True, medianprops=dict(color="black"))
        jit = (np.random.default_rng(1).random(a.size) - 0.5) * 0.15
        axes[1].scatter(np.ones(a.size) + jit, a, s=30, color="#B279A2", edgecolor="white", zorder=3)
        axes[1].axhline(PEN_PARALLEL_DEG, color="#E45756", linestyle="--", linewidth=1.5)
        axes[1].set_xticks([1]); axes[1].set_xticklabels(["C2"])
        axes[1].set_ylabel("mean pen separation angle (deg)")
        axes[1].set_title(f"Near-parallel?\n<{PEN_PARALLEL_DEG:.0f}° (red) = intersection ill-defined",
                          fontsize=10, loc="left")
    else:
        axes[1].set_visible(False)
    # C: on-point fraction vs completion time
    stats.append(_scatter_reg(axes[2], trials, "frac_converged", "completion_time_s",
                              "Fraction of frames pens on-point", "Completion time (s)"))
    fig.suptitle("Pen technique — is the operator using the intersection?",
                 y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_pen_usage.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, [s for s in stats if s]


# --- across-block LEARNING / practice effect (block 1 -> 2 -> 3) ------------------------------
# For each metric, "improvement" direction is noted; the trend is a slope + correlation of the
# metric against block number, computed on per-participant x block means (so each participant
# contributes one point per block, controlling for who was present).
LEARNING_METRICS = [
    ("completion_time_s",  "Completion time (s)",           "lower"),
    ("int_rms_mm",         "Intersection RMS wander (mm)",  "lower"),
    ("int_mean_vel_mm_s",  "Intersection mean velocity",    "lower"),
    ("frac_converged",     "Pens on-point fraction",        "higher"),
    ("pen_gap_med_mm",     "Pen line-to-line gap (mm)",     "lower"),
    ("settle_time_s",      "Settle time (s)",               "lower"),
    ("n_freeze",           "Freezes per trial",             "lower"),
]


def learning_tables(trials):
    """Return (per_participant_block_means, per_block_means, trend_df) or (None, None, None)."""
    if "block" not in trials.columns or trials["block"].notna().sum() == 0:
        return None, None, None
    tr = trials.dropna(subset=["block"]).copy()
    tr["block"] = tr["block"].astype(int)
    metrics = [m for m, _, _ in LEARNING_METRICS if m in tr.columns and tr[m].notna().any()]
    if not metrics or tr["block"].nunique() < 2:
        return None, None, None
    pb = tr.groupby(["participant", "block"], as_index=False)[metrics].mean()
    blk = pb.groupby("block", as_index=False)[metrics].mean().sort_values("block")
    better = {m: d for m, _, d in LEARNING_METRICS}
    rows = []
    for m in metrics:
        sub = pb[["block", m]].dropna()
        if sub["block"].nunique() < 2 or len(sub) < 3:
            continue
        x = sub["block"].to_numpy(float); y = sub[m].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else np.nan
        r, p, n = corr_with_p(x, y, "pearson")
        first, last = float(blk[m].iloc[0]), float(blk[m].iloc[-1])
        # "improved" if it moved in the beneficial direction from first to last block
        improved = (last < first) if better[m] == "lower" else (last > first)
        rows.append({"metric": m, "improves_when": better[m], "slope_per_block": slope,
                     "pearson_r": r, "pearson_p": p, "n": n,
                     "first_block_mean": first, "last_block_mean": last,
                     "improved_1_to_last": bool(improved)})
    return pb, blk, pd.DataFrame(rows)


def fig_learning(trials, figdir):
    pb, blk, trend = learning_tables(trials)
    if pb is None:
        return None
    metrics = [(m, t) for m, t, _ in LEARNING_METRICS if m in pb.columns and pb[m].notna().any()]
    if not metrics:
        return None
    ncol = min(3, len(metrics)); nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    axes = axes.ravel()
    blocks = sorted(pb["block"].unique())
    for ax, (m, title) in zip(axes, metrics):
        for i, (pid, s) in enumerate(pb.groupby("participant")):
            s = s.sort_values("block")
            ax.plot(s["block"], s[m], "-o", color=PART_CMAP(i % 10), alpha=0.45, markersize=4)
        bm = blk.sort_values("block")
        ax.plot(bm["block"], bm[m], "-o", color="#111", linewidth=2.6, markersize=9,
                zorder=5, label="mean")
        sub = ""
        row = trend[trend["metric"] == m] if trend is not None else None
        if row is not None and len(row):
            rr = row.iloc[0]
            arrow = "improves" if rr["improved_1_to_last"] else "worsens"
            sub = (f"slope={rr['slope_per_block']:+.2f}/block  r={rr['pearson_r']:+.2f} "
                   f"(p={rr['pearson_p']:.3f})  [{arrow}]")
        ax.set_xticks(blocks); ax.set_xlabel("block")
        ax.set_title(f"{title}\n{sub}", fontsize=10, loc="left")
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Learning across blocks (1 → 3): faint = participants, bold = mean",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_learning.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_intersection_traces(trials, telem, figdir, n_examples=3):
    """Qualitative evidence: intersection displacement-from-centroid over time for the
    steadiest vs. jumpiest C2 trials, with pen-dropout spans shaded."""
    have = trials.dropna(subset=["int_rms_mm"]).sort_values("int_rms_mm")
    if have.empty or telem.empty:
        return None
    picks = pd.concat([have.head(n_examples).assign(kind="steadiest"),
                       have.tail(n_examples).assign(kind="jumpiest")])
    fig, axes = plt.subplots(2, n_examples, figsize=(4.6 * n_examples, 7.2), squeeze=False)
    row_of = {"steadiest": 0, "jumpiest": 1}
    col_ctr = {0: 0, 1: 0}
    for _, tr in picks.iterrows():
        seg = telem[(telem["file"] == tr["file"]) &
                    (telem["t"] > tr["t_start"]) & (telem["t"] <= tr["t_stop"])].copy()
        r = row_of[tr["kind"]]; c = col_ctr[r]; col_ctr[r] += 1
        if c >= n_examples:
            continue
        ax = axes[r][c]
        if seg.empty:
            ax.set_visible(False); continue
        t0 = seg["t"].iloc[0]
        tt = seg["t"].to_numpy(float) - t0
        X = seg[["Xint_x", "Xint_y", "Xint_z"]].to_numpy(float)
        fin = np.all(np.isfinite(X), axis=1)
        disp = np.full(len(X), np.nan)
        if fin.any():
            ctr = X[fin].mean(axis=0)
            disp[fin] = np.linalg.norm(X[fin] - ctr, axis=1) * 1000.0
        ax.plot(tt, disp, "-", color="#E45756", linewidth=1.5)
        # shade pen-dropout spans (n_pens < 2)
        oof = seg["n_pens"].to_numpy(float) < 2
        _shade_spans(ax, tt, oof, color="#888", alpha=0.18)
        ax.set_title(f"P{tr['participant']} trial {int(tr['trial'])} · {tr['kind']}\n"
                     f"RMS={tr['int_rms_mm']:.1f} mm · CT={tr['completion_time_s']:.1f} s",
                     fontsize=9, loc="left")
        ax.set_xlabel("time in trial (s)"); ax.set_ylabel("intersection offset (mm)")
    fig.suptitle("Intersection wander over time — steadiest (top) vs jumpiest (bottom) C2 trials\n"
                 "grey = a pen marker was out of frame", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_intersection_traces.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def _shade_spans(ax, t, mask, **kw):
    """Shade contiguous True spans of `mask` (pen-dropout frames) along the time axis t."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return
    spans, s = [], None
    for i, mv in enumerate(mask):
        if mv and s is None:
            s = i
        elif not mv and s is not None:
            spans.append((s, i - 1)); s = None
    if s is not None:
        spans.append((s, len(mask) - 1))
    for a, b in spans:
        ax.axvspan(t[a], t[min(b, len(t) - 1)], **kw)


# =============================================================================================
# main
# =============================================================================================
def gather_paths(items, ext):
    paths = []
    for item in items:
        if os.path.isdir(item):
            paths += sorted(glob.glob(os.path.join(item, "**", f"*.{ext}"), recursive=True))
        else:
            paths += sorted(glob.glob(item))
    # de-dup, keep order
    seen = set(); uniq = []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def _pid_key(s):
    return str(s).strip().lstrip("0") or "0"


def _load_behavior_c2(csv_path, value_col):
    """Load a per-participant C2 value (e.g. bimanual offset, singleton rate) from a run_analysis
    output CSV (per_participant_condition.csv / singleton_by_condition.csv). Returns {pid_key: value}."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    d = pd.read_csv(csv_path)
    if value_col not in d.columns or "participant" not in d.columns:
        return None
    if "condition" in d.columns:
        d = d[d["condition"].astype(str).str.upper() == "C2"]
    d = d.groupby("participant", as_index=False)[value_col].mean()
    return {_pid_key(p): float(v) for p, v in zip(d["participant"], d[value_col]) if np.isfinite(v)}


def fig_freeze_vs_coordination(trials, figdir, sync_csv=None, singleton_csv=None):
    """Does operator TIMING predict C2 coordination? Grid of predictors x outcomes, participant-level:
      predictors = settle time (press→next-freeze: reorient + commit) AND freeze→press (post-freeze
                   working time);  outcomes = C2 bimanual offset AND C2 singleton rate.
    Hypothesis: more time -> lower offset & fewer singletons (negative correlations)."""
    sync = _load_behavior_c2(sync_csv, "median_sync_ms")
    singl = _load_behavior_c2(singleton_csv, "singleton_rate")
    outcomes = ([("Bimanual offset (ms) [lower=better]", sync)] if sync else []) + \
               ([("Singleton rate [lower=better]", singl)] if singl else [])
    if not outcomes:
        print("[timing] no behavioural sync/singleton CSV supplied (--sync-csv / --singleton-csv) — "
              "timing-vs-coordination figure skipped.")
        return None, []
    preds = [(c, lab, how) for c, lab, how in
             [("settle_time_s", "settle time (press → freeze, s)", "mean"),
              ("freeze_to_press_s", "freeze → press time (s)", "mean")]
             if c in trials.columns and trials[c].notna().sum() > 0]
    if not preds:
        return None, []
    fig, axes = plt.subplots(len(preds), len(outcomes),
                             figsize=(5.8 * len(outcomes), 4.6 * len(preds)), squeeze=False)
    stats = []
    for i, (pcol, plabel, how) in enumerate(preds):
        g = trials.groupby("participant")[pcol]
        pv = (g.median() if how == "median" else g.mean()).reset_index()
        pv["pid"] = pv["participant"].map(_pid_key)
        for j, (ylabel, omap) in enumerate(outcomes):
            ax = axes[i][j]
            m = pv.assign(y=pv["pid"].map(omap)).dropna(subset=["y", pcol])
            x = m[pcol].to_numpy(float); y = m["y"].to_numpy(float)
            for k, (_, r) in enumerate(m.iterrows()):
                ax.scatter(r[pcol], r["y"], s=60, color=PART_CMAP(k % 10),
                           edgecolor="white", linewidth=0.6, zorder=3)
                ax.annotate(f"P{r['pid']}", (r[pcol], r["y"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            if np.std(x) > 0 and len(x) >= 2:
                b1, b0 = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, b0 + b1 * xs, "-", color="#333", linewidth=2, zorder=2)
            rr, pp, nn = corr_with_p(x, y, "pearson"); rs, ps, _ = corr_with_p(x, y, "spearman")
            ax.set_xlabel(plabel); ax.set_ylabel(ylabel)
            ax.set_title(f"r={rr:+.2f} (p={pp:.3f}, n={nn})  ρ={rs:+.2f} (p={ps:.3f})",
                         loc="left", fontsize=9)
            stats.append((f"{pcol} vs {ylabel.split('[')[0].strip()} (participant)",
                          "", rr, pp, rs, ps, nn))
    fig.suptitle("Operator timing vs C2 coordination — settle time (top) & freeze→press time (bottom)",
                 y=1.0, fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(figdir, "fig_timing_vs_coordination.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, stats


def _concat_run(trials, telem, outdir, sync_csv=None, singleton_csv=None):
    if not trials:
        raise SystemExit("No C2 trials found in the input(s).")
    trials_all = pd.concat(trials, ignore_index=True)
    telem_all = pd.concat(telem, ignore_index=True) if telem \
        else pd.DataFrame(columns=["t"] + TLM_CH + ["file"])
    return run(trials_all, telem_all, outdir, sync_csv, singleton_csv)


EXCLUDE_PARTICIPANTS = {"try", "test"}   # test / practice runs to drop from analysis


def _report_active_oof(trials, outdir):
    """Report the mean fraction of ACTIVE-TRACKING frames (freeze / button-hold / retract excluded)
    in which a pen was out of view — i.e. how often the markers were lost WHILE the robot was looking
    for them. Prints three complementary numbers and writes them to active_oof_summary.csv:
      pooled          = total out-of-view tracking frames / total tracking frames (frame-weighted)
      mean_of_trials  = mean of each trial's oof_frac (trial-weighted)
      mean_of_parts   = mean of each participant's mean oof_frac (participant-weighted)"""
    if "oof_frac" not in trials.columns or trials["oof_frac"].notna().sum() == 0:
        return None
    rows = {}
    if {"pen_oof_frames", "orient_frames"}.issubset(trials.columns):
        num = float(pd.to_numeric(trials["pen_oof_frames"], errors="coerce").sum())
        den = float(pd.to_numeric(trials["orient_frames"], errors="coerce").sum())
        rows["pooled"] = num / den if den > 0 else np.nan
    tr_v = pd.to_numeric(trials["oof_frac"], errors="coerce").dropna().to_numpy(float)
    rows["mean_of_trials"] = float(tr_v.mean())
    rows["sd_of_trials"] = float(tr_v.std(ddof=1)) if len(tr_v) > 1 else np.nan
    ppm = trials.groupby("participant")["oof_frac"].mean()
    pp_v = ppm.dropna().to_numpy(float)
    rows["mean_of_participants"] = float(pp_v.mean())
    rows["sd_of_participants"] = float(pp_v.std(ddof=1)) if len(pp_v) > 1 else np.nan
    rows["median_of_participants"] = float(ppm.median())
    rows["n_trials"] = int(len(tr_v))
    rows["n_participants"] = int(len(pp_v))
    # ready-to-quote percentages (participant-level mean ± SD is the headline)
    m_pct = 100 * rows["mean_of_participants"]; s_pct = 100 * rows["sd_of_participants"]
    quote = "%.0f\\pm%.0f\\%%" % (m_pct, s_pct)
    rows["quote_percent"] = "%.0f +/- %.0f %%" % (m_pct, s_pct)
    print("\n" + "=" * 64)
    print("PEN OUT-OF-VIEW while ACTIVELY TRACKING (freeze / hold / retract excluded)")
    if "pooled" in rows:
        print(f"  pooled frame fraction        : {rows['pooled']:.3f}   "
              f"(total lost tracking frames / total tracking frames)")
    print(f"  per-trial        : {rows['mean_of_trials']:.3f} +/- {rows['sd_of_trials']:.3f}"
          f"  (n={rows['n_trials']} trials)")
    print(f"  per-participant  : {rows['mean_of_participants']:.3f} +/- {rows['sd_of_participants']:.3f}"
          f"  (n={rows['n_participants']} participants)")
    print(f"  >>> QUOTE: on average a pen was out of view {m_pct:.0f} +/- {s_pct:.0f}% of the "
          f"frames while the robot was tracking (participant-level mean +/- SD, n={rows['n_participants']})")
    print("=" * 64)
    pd.DataFrame([rows]).to_csv(os.path.join(outdir, "active_oof_summary.csv"), index=False)
    with open(os.path.join(outdir, "active_oof_quote.tex"), "w") as f:
        f.write("%% Auto-generated: participant-level mean +/- SD of the active-tracking "
                "pen-out-of-view fraction.\n"
                "\\newcommand{\\oofmean}{$%s$}  %% e.g. use as: a pen was lost \\oofmean{} of "
                "tracking frames\n" % quote)
    return rows


def run(trials_all, telem_all, outdir, sync_csv=None, singleton_csv=None):
    os.makedirs(outdir, exist_ok=True)
    figdir = os.path.join(outdir, "figures"); os.makedirs(figdir, exist_ok=True)

    # drop test participants (e.g. a run entered as "try")
    if "participant" in trials_all.columns:
        pnorm = trials_all["participant"].astype(str).str.strip().str.lower()
        keep = ~pnorm.isin(EXCLUDE_PARTICIPANTS)
        if (~keep).any():
            print(f"[analysis] excluding {int((~keep).sum())} trial(s) from test participant(s) "
                  f"{sorted(EXCLUDE_PARTICIPANTS)}")
            trials_all = trials_all[keep].copy()
        if "file" in telem_all.columns and not trials_all.empty:
            telem_all = telem_all[telem_all["file"].isin(set(trials_all["file"]))].copy()
    if trials_all.empty:
        raise SystemExit("No trials left after excluding test participants.")

    trials_all = _clean_trial_outliers(trials_all)     # drop extreme TIME values (10x second max)

    _report_active_oof(trials_all, outdir)             # mean OOF fraction while ACTIVELY tracking

    trials_all.to_csv(os.path.join(outdir, "per_c2_trial.csv"), index=False)

    # aggregate every numeric metric to participant means (whichever are present)
    agg_cols = [c for c in
                ["completion_time_s", "settle_time_s", "freeze_to_press_s",
                 "int_rms_mm", "int_pathlen_mm", "int_pathlen_norm",
                 "int_rms_norm", "interbutton_travel_mm", "interbutton_mean_mm", "int_mean_vel_mm_s",
                 "int_ellipse_area_mm2", "int_sparc", "int_vel_p95_mm_s", "int_vel_max_mm_s",
                 "int_fastjump_events", "int_fastjump_rate_hz", "int_jitter_rms_mm",
                 "int_jerk_rms_mm_s3", "oof_frac", "pen_oof_events",
                 "board_rms_mm", "atten_ratio", "track_lag_ms", "track_err_mean_mm",
                 "track_err_rot_deg", "robot_active_frac", "pen_gap_med_mm", "pen_gap_mean_mm",
                 "pen_angle_mean_deg",
                 "frac_converged", "frac_parallel", "n_freeze"] if c in trials_all.columns]
    ppc = trials_all.groupby("participant", as_index=False).agg(
        {"trial": "count", **{c: "mean" for c in agg_cols}}).rename(columns={"trial": "n_c2_trials"})
    ppc.to_csv(os.path.join(outdir, "per_participant.csv"), index=False)

    figs, cor_rows = [], []
    f1, s1 = fig_wander_vs_performance(trials_all, figdir); figs.append(f1); cor_rows += s1
    f2, s2 = fig_dropout_vs_performance(trials_all, figdir); figs.append(f2); cor_rows += s2
    figs.append(fig_participant_profile(trials_all, figdir))
    fa, sa = fig_assistance(trials_all, figdir); figs.append(fa); cor_rows += sa
    fwr, swr = fig_wander_ratio(trials_all, figdir, sync_csv, singleton_csv); figs.append(fwr); cor_rows += swr
    fdc, sdc = fig_dropout_vs_coordination(trials_all, figdir, sync_csv, singleton_csv); figs.append(fdc); cor_rows += sdc
    fid, sid = fig_intersection_distance_vs_outcomes(trials_all, figdir, sync_csv, singleton_csv); figs.append(fid); cor_rows += sid
    figs.append(fig_oof_burden(trials_all, figdir))
    figs += fig_oof_relationships(trials_all, figdir, sync_csv, singleton_csv)
    fj, sj = fig_jitter(trials_all, figdir); figs.append(fj); cor_rows += sj
    fp, sp = fig_pen_usage(trials_all, figdir); figs.append(fp); cor_rows += sp
    figs.append(fig_learning(trials_all, figdir))
    ff, sf = fig_freeze_vs_coordination(trials_all, figdir, sync_csv, singleton_csv)
    figs.append(ff); cor_rows += sf
    figs.append(fig_intersection_traces(trials_all, telem_all, figdir))

    # across-block learning tables
    pb, blk, trend = learning_tables(trials_all)
    if pb is not None:
        pb.to_csv(os.path.join(outdir, "per_participant_block.csv"), index=False)
        if trend is not None and not trend.empty:
            trend.to_csv(os.path.join(outdir, "learning_trends.csv"), index=False)

    # trial-level correlations for the added dynamics metrics vs completion time
    for xcol in ["int_mean_vel_mm_s", "int_ellipse_area_mm2", "int_sparc", "int_pathlen_norm",
                 "int_rms_norm", "int_vel_p95_mm_s", "int_fastjump_rate_hz", "int_jitter_rms_mm",
                 "int_jerk_rms_mm_s3", "atten_ratio",
                 "track_lag_ms", "track_err_mean_mm", "robot_active_frac",
                 "pen_gap_med_mm", "pen_gap_mean_mm", "frac_converged", "frac_parallel"]:
        if xcol in trials_all.columns and trials_all[xcol].notna().sum() >= 3:
            r, p, n = corr_with_p(trials_all[xcol], trials_all["completion_time_s"], "pearson")
            rs, ps, _ = corr_with_p(trials_all[xcol], trials_all["completion_time_s"], "spearman")
            cor_rows.append((xcol, "completion_time_s", r, p, rs, ps, n))

    # participant-level correlations too (unit = participant) — the "why some did worse" test
    for xcol in ["int_rms_mm", "int_pathlen_mm", "int_pathlen_norm", "int_rms_norm",
                 "int_mean_vel_mm_s", "int_ellipse_area_mm2", "int_sparc", "int_vel_p95_mm_s",
                 "int_fastjump_rate_hz", "int_jitter_rms_mm", "int_jerk_rms_mm_s3",
                 "oof_frac", "pen_oof_events", "atten_ratio"]:
        if xcol not in ppc.columns:
            continue
        r, p, n = corr_with_p(ppc[xcol], ppc["completion_time_s"], "pearson")
        rs, ps, _ = corr_with_p(ppc[xcol], ppc["completion_time_s"], "spearman")
        cor_rows.append((xcol + " (participant-level)", "completion_time_s", r, p, rs, ps, n))

    cordf = pd.DataFrame(cor_rows, columns=["x", "y", "pearson_r", "pearson_p",
                                            "spearman_r", "spearman_p", "n"])
    cordf.to_csv(os.path.join(outdir, "correlations.csv"), index=False)

    n_part = trials_all["participant"].nunique()
    print(f"[analysis] {len(trials_all)} C2 trials from {n_part} participant(s).")
    print(f"[analysis] tables + figures -> {outdir}/")
    for f in filter(None, figs):
        print(f"           {f}")
    print("\nWander/dropout vs completion time (trial-level unless noted):")
    for _, r in cordf.iterrows():
        print(f"  {r['x']:38s} vs {r['y']:18s}  "
              f"r={r['pearson_r']:+.2f} (p={r['pearson_p']:.3f})  "
              f"ρ={r['spearman_r']:+.2f} (p={r['spearman_p']:.3f})  n={int(r['n'])}")
    if trend is not None and not trend.empty:
        print("\nLearning across blocks (block-1 mean -> last-block mean; slope per block):")
        for _, r in trend.iterrows():
            tag = "IMPROVED" if r["improved_1_to_last"] else "no gain"
            print(f"  {r['metric']:20s} {r['first_block_mean']:8.2f} -> {r['last_block_mean']:8.2f}"
                  f"  slope={r['slope_per_block']:+.2f}/blk  r={r['pearson_r']:+.2f}"
                  f" (p={r['pearson_p']:.3f})  [{tag}]")
    return ppc, cordf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xdf", nargs="+", help="one or more .xdf files, globs, or a directory")
    ap.add_argument("--csv", nargs="+",
                    help="one or more flat orient_log_*.csv files, globs, or a directory "
                         "(written directly by orient_experiment.py — no XDF/pyxdf needed)")
    ap.add_argument("--out", default="c2_evidence")
    ap.add_argument("--sync-csv", default=None,
                    help="run_analysis per_participant_condition.csv — joins C2 bimanual offset for "
                         "the time-before-freeze figure")
    ap.add_argument("--singleton-csv", default=None,
                    help="run_analysis singleton_by_condition.csv — joins C2 singleton rate for "
                         "the time-before-freeze figure")
    ap.add_argument("--selftest", action="store_true",
                    help="synthesize mock XDF streams and run the full pipeline (no real data)")
    args = ap.parse_args()

    if args.selftest:
        trials, telem = [], []
        for path, streams in _make_selftest_streams():
            tdf, td = extract_file(streams, path)
            trials.append(tdf); telem.append(td)
        _concat_run(trials, telem, args.out, args.sync_csv, args.singleton_csv)
        return

    if args.csv:                                            # flat-CSV mode (no XDF)
        paths = gather_paths(args.csv, "csv")
        if not paths:
            raise SystemExit(f"No .csv files matched {args.csv!r}.")
        trials, telem = [], []
        for path in paths:
            tdf, td = extract_csv_file(path)
            if not tdf.empty:
                trials.append(tdf)
            if not td.empty:
                telem.append(td)
            print(f"[read] {os.path.basename(path)}: {len(tdf)} C2 trial(s), {len(td)} frames")
        _concat_run(trials, telem, args.out, args.sync_csv, args.singleton_csv)
        return

    if not args.xdf:
        raise SystemExit("Provide --csv <files|dir> (flat CSV) or --xdf <files|dir> (XDF), "
                         "or --selftest.")
    paths = gather_paths(args.xdf, "xdf")
    if not paths:
        raise SystemExit(f"No .xdf files matched {args.xdf!r}.")
    trials, telem = [], []
    for path in paths:
        streams = load_xdf_streams(path)
        tdf, td = extract_file(streams, path)
        if not tdf.empty:
            trials.append(tdf)
        if not td.empty:
            telem.append(td)
        print(f"[read] {os.path.basename(path)}: {len(tdf)} C2 trial(s), {len(td)} telemetry frames")
    _concat_run(trials, telem, args.out, args.sync_csv, args.singleton_csv)


# =============================================================================================
# self-test: build pyxdf-shaped mock streams with an injected wander->slowness relationship
# =============================================================================================
def _mk_stream(name, stype, ts, series, ch_labels=None):
    info = {"name": [name], "type": [stype], "channel_count": [str(len(ch_labels) if ch_labels else 1)]}
    if ch_labels:
        info["desc"] = [{"channels": [{"channel": [{"label": [c]} for c in ch_labels]}]}]
    return {"info": info, "time_stamps": np.asarray(ts, float), "time_series": series}


def _make_selftest_streams(n_participants=6, trials_per=4, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for pi in range(n_participants):
        pid = pi + 1
        base_noise = 0.003 + 0.010 * (pi / (n_participants - 1))   # participants differ in pen steadiness
        logger_t, logger_s = [], []
        grid_t, grid_s = [], []
        robot_t, robot_s = [], []
        tlm_t, tlm_rows = [], []
        clock = 1000.0 + pi * 500.0
        for k in range(trials_per):
            trial = pi * 9 + k + 1
            t_start = clock
            logger_t.append(t_start); logger_s.append(
                [f"START:{(k % 8) + 1};P={pid};TRIAL={trial};COND=C2;BLOCK={k // 3 + 1}"])
            # telemetry frames for this trial
            fps, dur = 6.0, 12.0
            nfr = int(fps * dur)
            noise = base_noise * (1.0 + 0.5 * rng.standard_normal())
            noise = max(0.001, noise)
            ctr = np.array([0.40, 0.0, 0.30])
            pathlen = 0.0; prev = None; Xs = []
            for fi in range(nfr):
                tt = t_start + 3.2 + fi / fps          # after the 3.2 s countdown
                X = ctr + noise * rng.standard_normal(3)
                # inject occasional pen dropout, more for noisier participants
                npens = 2 if rng.random() > (0.03 + 2.0 * noise) else rng.integers(0, 2)
                seenA = 1.0 if npens == 2 or rng.random() > 0.5 else 0.0
                seenB = float(npens) - seenA if npens < 2 else 1.0
                seenB = max(0.0, min(1.0, seenB))
                if npens < 2:
                    row = [float(npens), seenA, seenB, np.nan, np.nan, np.nan, np.nan,
                           ctr[0], ctr[1], ctr[2], np.nan, np.nan, float(trial)]
                else:
                    if prev is not None:
                        pathlen += float(np.linalg.norm(X - prev))
                    prev = X.copy(); Xs.append(X)
                    row = [2.0, 1.0, 1.0, X[0], X[1], X[2], noise * 1000.0 * 2,
                           ctr[0], ctr[1], ctr[2], abs(rng.standard_normal()) * 5, abs(rng.standard_normal()) * 2,
                           float(trial)]
                tlm_t.append(tt); tlm_rows.append(row)
            Xs = np.array(Xs)
            rms_mm = float(np.sqrt(((Xs - Xs.mean(0)) ** 2).sum(1).mean())) * 1000.0 if len(Xs) else 0.0
            # performance: jumpier intersection -> slower presses (the relationship we want to detect)
            ct = 8.0 + 0.25 * rms_mm + rng.normal(0, 0.6)
            # 8 presses spanning ct seconds, starting shortly after orient
            p0 = t_start + 4.0
            for j in range(8):
                pt = p0 + ct * j / 7.0
                logger_t.append(pt); logger_s.append([f"PAIR:{(k % 8) + 1}"])
                robot_t.append(pt - 0.3); robot_s.append(
                    [f"MOVE_START:{(k % 8) + 1} trial={trial} cond=C2 func_delay={0.2 + 0.02*rms_mm:.3f}"])
            t_stop = p0 + ct + 1.0
            grid_t.append(t_stop); grid_s.append(["STOP"])
            # emulate the controller TRIAL_SUMMARY
            oof_frames = int(sum(1 for r in tlm_rows[-nfr:] if r[0] < 2))
            robot_t.append(t_stop - 0.05); robot_s.append([
                f"TRIAL_SUMMARY trial={trial} cond=C2 block={k//3+1} P={pid} orient_frames={nfr} "
                f"int_n={len(Xs)} int_rms_mm={rms_mm:.2f} "
                f"int_std_mm=[{Xs[:,0].std()*1000:.2f},{Xs[:,1].std()*1000:.2f},{Xs[:,2].std()*1000:.2f}] "
                f"int_pathlen_mm={pathlen*1000:.1f} pen_oof_frames={oof_frames} "
                f"pen_oof_events={max(1,oof_frames//3)} penA_miss={oof_frames//2} penB_miss={oof_frames//2} "
                f"oof_frac={oof_frames/max(1,nfr):.3f} nfaces=8 "
                f"func_delay_mean={0.2+0.02*rms_mm:.3f} present_time_mean={0.8:.3f}"])
            clock = t_stop + 8.0
        out.append((f"P{pid}_selftest.xdf", [
            _mk_stream("LoggerMarkers", "Markers", logger_t, logger_s),
            _mk_stream("GridControl", "Markers", grid_t, grid_s),
            _mk_stream("RobotMarkers", "Markers", robot_t, robot_s),
            _mk_stream("OrientTelemetry", "Telemetry", tlm_t, tlm_rows, ch_labels=TLM_CH),
        ]))
    return out


if __name__ == "__main__":
    main()
