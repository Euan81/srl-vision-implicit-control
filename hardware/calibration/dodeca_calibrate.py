#!/usr/bin/env python3
"""
dodeca_calibrate.py — DodecaPen "DC" (dodecahedron calibration) + pen-tip pivot.

Follows DodecaPen (Wu et al., UIST 2017), Sections "Dodecahedron Calibration" and
"Pen-tip Calibration":

  * Dodecahedron calibration (DC): a one-time offline bundle adjustment that
    recovers the precise pose of every marker on the body, correcting the model
    error left by hand-gluing. Marker poses are initialised to their ideal
    positions and one marker is fixed to remove gauge freedom (paper does the
    same); per-image dodecahedron poses are initialised from APE.

  * Pen-tip calibration: press the tip on a surface so it stays fixed while the
    body moves; from the tracked poses (R_k, t_k) the fixed tip satisfies
    R_k1 c + t_k1 = R_k2 c + t_k2, a linear least-squares solve for the tip c in
    the body frame (paper's formulation).

DEVIATION from the paper (stated, with reason):
  * The paper's DC minimises a PHOTOMETRIC cost (their eq. 8 — pixel intensities
    of a rendered model vs the image), which is their dense-alignment objective.
    We minimise marker-corner REPROJECTION error instead (the standard "marker
    map" bundle adjustment, as in ArUco's marker_mapper). Reason: it needs no
    textured renderer/mipmaps, is robust, and is sufficient to remove the
    gluing/model error — which is exactly what DC is for. The photometric
    refinement is the separate DPR stage (not implemented yet).

Outputs dodeca_layout.npz (ids, corners Nx4x3 in the body frame, tip) which
dodecapen_pose.py loads automatically.

Usage:
  python3 dodeca_calibrate.py --pen A --live                # live 2K capture, pen A -> dodeca_layout.npz
  python3 dodeca_calibrate.py --pen B --live                # live 2K capture, pen B -> dodeca_layout_B.npz
  python3 dodeca_calibrate.py --pen B --images calib/*.jpg --pivot pivot/*.jpg   # from photos
  python3 dodeca_calibrate.py --selftest                    # synthetic verification, no camera

--pen selects the marker ID range (A = 4-14, B = 15-25) and the default output file.
Running with no --live and no --images now ABORTS instead of silently writing the ideal
layout (which would wipe a real calibration).
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import cv2
from scipy.optimize import least_squares

from dodecapen_pose import build_layout, detector, MARKER_M

# Marker-local corner model (TL, TR, BR, BL), matching cv2 detection order.
_h = MARKER_M / 2.0
L_LOCAL = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]], float)
ANCHOR_ID = 4   # fixed marker (gauge), = body frame anchor (top face)


# ---------------------------------------------------------------------------
def corners_to_pose(P):
    """4x3 body-frame corners (TL,TR,BR,BL) -> (rvec, t) mapping local->body."""
    c = P.mean(0)
    x = P[1] - P[0]; x /= np.linalg.norm(x)
    y = P[0] - P[3]; y /= np.linalg.norm(y)
    z = np.cross(x, y); z /= np.linalg.norm(z)
    R = np.column_stack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel(), c


def pose_to_corners(rv, t):
    R, _ = cv2.Rodrigues(rv)
    return (R @ L_LOCAL.T).T + t


def ape_pose(det, BODY):
    """Single PnP over all known markers in one image -> (rvec, tvec) body->cam."""
    obj, img = [], []
    for mid, c in det.items():
        if mid in BODY:
            obj.append(BODY[mid]); img.append(c)
    if len(obj) < 1:
        return None
    obj = np.vstack(obj).astype(np.float64); img = np.vstack(img).astype(np.float64)
    ok, rv, tv = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    return (rv.ravel(), tv.ravel()) if ok else None


# ---------------------------------------------------------------------------
# Bundle adjustment (DC)
# ---------------------------------------------------------------------------
def bundle_adjust(dets, id_base=4):
    """dets: list of {id: corners(4,2)}. Returns calibrated BODY_CORNERS dict.
    id_base selects the pen (A=4, B=15); its top face is the gauge-fixed anchor."""
    anchor = id_base
    BODY0, _, _, _ = build_layout(id_base=id_base)
    marker_ids = sorted(m for m in BODY0 if m != anchor)
    # init marker poses (rvec,t) from ideal layout
    mpose = {m: corners_to_pose(BODY0[m]) for m in BODY0}
    # init view poses from APE on the ideal layout; drop frames with <1 marker
    views = []
    for det in dets:
        p = ape_pose(det, BODY0)
        if p is not None:
            views.append((det, p))
    nM, nV = len(marker_ids), len(views)

    def pack():
        x = []
        for m in marker_ids:
            x += [*mpose[m][0], *mpose[m][1]]
        for _, (rv, tv) in views:
            x += [*rv, *tv]
        return np.array(x, float)

    def unpack(x):
        mp = {anchor: mpose[anchor]}
        for i, m in enumerate(marker_ids):
            mp[m] = (x[6*i:6*i+3], x[6*i+3:6*i+6])
        vp = []
        off = 6*nM
        for k in range(nV):
            vp.append((x[off+6*k:off+6*k+3], x[off+6*k+3:off+6*k+6]))
        return mp, vp

    # ADDITION (not in the paper): a soft prior pinning each marker to its ideal
    # CAD pose. Reprojection-only BA on a compact object is weakly constrained and
    # can deform while keeping reprojection low; the prior uses the known geometry
    # to remove that degeneracy and keeps the solve to small gluing corrections.
    PRIOR_R, PRIOR_T = np.deg2rad(5.0), 0.003   # allowed std: 5 deg, 3 mm
    ideal = {m: corners_to_pose(BODY0[m]) for m in BODY0}

    def residuals(x):
        mp, vp = unpack(x)
        res = []
        for (det, _), (rv_v, t_v) in zip(views, vp):
            for mid, obs in det.items():
                if mid not in mp:
                    continue
                rv_m, t_m = mp[mid]
                rv_cm, t_cm, *_ = cv2.composeRT(rv_m, t_m.reshape(3, 1),
                                                rv_v, t_v.reshape(3, 1))
                proj, _ = cv2.projectPoints(L_LOCAL, rv_cm, t_cm, K, dist)
                res.append((proj.reshape(4, 2) - obs).ravel())
        for m in marker_ids:                        # prior toward ideal pose
            res.append((mp[m][0] - ideal[m][0]) / PRIOR_R)
            res.append((mp[m][1] - ideal[m][1]) / PRIOR_T)
        return np.concatenate(res)

    x0 = pack()
    sol = least_squares(residuals, x0, method='lm', xtol=1e-12, ftol=1e-12)
    mp, _ = unpack(sol.x)
    rms = np.sqrt(np.mean(sol.fun**2))
    print(f"[DC] views={nV} markers={nM+1}  reproj RMS={rms:.3f}px")
    return {m: pose_to_corners(*mp[m]) for m in mp}


# ---------------------------------------------------------------------------
# Pen-tip pivot calibration
# ---------------------------------------------------------------------------
def pivot_tip(poses):
    """poses: list of (R(3x3), t(3)) body->cam with the tip held fixed.
    Solve R_k c + t_k = p  (c = tip in body frame, p = fixed point)."""
    A, b = [], []
    for R, t in poses:
        A.append(np.hstack([R, -np.eye(3)])); b.append(-t)
    A = np.vstack(A); b = np.concatenate(b)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3]   # tip c in body frame


# ---------------------------------------------------------------------------
# Pen -> (top-face id_base, default output file). IDs run id_base..id_base+10.
PEN_IDBASE = {"A": 4, "B": 15}
PEN_OUT    = {"A": "dodeca_layout.npz", "B": "dodeca_layout_B.npz"}


def _open_camera(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {idx}.")
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened at {aw}x{ah}")
    if (aw, ah) != (2560, 1440):
        print(f"[WARN] got {aw}x{ah}, not 2K — the intrinsics must match the capture resolution.")
    return cap


def _put(img, t, org, c=(0, 220, 80), s=0.6):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, c, 1)


def live_capture(idx, pen_ids):
    """Live 2K capture of this pen's DC + pivot frames.
    Returns (dc_frames, pivot_frames) as lists of {id: corners(4,2)}, or
    (None, None) if cancelled. Only markers belonging to this pen are kept."""
    cap = _open_camera(idx)
    dc, piv, seen = [], [], set()
    print("LIVE capture — SPACE: grab DC view (turn the pen between grabs, cover all faces)")
    print("               p: grab PIVOT view (press the tip on a fixed point, rotate body)")
    print("               u: undo last DC | ENTER: finish & calibrate | Esc: cancel")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cs, ids, _ = detector.detectMarkers(frame)
        det = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, cs, ids)
            det = {int(i): c.reshape(4, 2) for c, i in zip(cs, ids.flatten()) if int(i) in pen_ids}
        _put(frame, f"DC views: {len(dc)}   pivot views: {len(piv)}   faces now: {sorted(det)}", (20, 44))
        _put(frame, f"faces ever seen: {sorted(seen)}  of  {sorted(pen_ids)}", (20, 80), (180, 180, 180), 0.55)
        _put(frame, "SPACE DC | p pivot | u undo | ENTER done | Esc cancel", (20, 116), (170, 170, 170), 0.55)
        disp = frame
        h, w = disp.shape[:2]
        if w > 1280:
            disp = cv2.resize(disp, (1280, int(h * 1280 / w)))
        cv2.imshow("dodeca DC capture", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cap.release(); cv2.destroyAllWindows(); return None, None
        if k == 13:
            break
        if k == ord(' '):
            if len(det) >= 2:
                dc.append(det); seen |= set(det)
                print(f"[+] DC view {len(dc)}  faces={sorted(det)}")
            else:
                print(f"[skip] need >= 2 of this pen's faces in view (saw {sorted(det)})")
        elif k == ord('p'):
            if len(det) >= 1:
                piv.append(det); print(f"[+] pivot view {len(piv)}  faces={sorted(det)}")
            else:
                print("[skip] no faces of this pen visible")
        elif k == ord('u') and dc:
            dc.pop(); print(f"[-] undo DC -> {len(dc)} views")
    cap.release(); cv2.destroyAllWindows()
    return dc, piv


def main():
    global K, dist
    ap = argparse.ArgumentParser()
    ap.add_argument("--pen", choices=["A", "B"], default="A",
                    help="which pen: A = IDs 4-14, B = IDs 15-25")
    ap.add_argument("--live", action="store_true",
                    help="capture DC/pivot frames live from the 2K camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--images", nargs="+", help="DC photos (instead of --live)")
    ap.add_argument("--pivot", nargs="+", help="photos with the tip held fixed")
    ap.add_argument("--K", default="camera_matrix.npy")
    ap.add_argument("--dist", default="dist_coeffs.npy")
    ap.add_argument("--out", default=None, help="output npz (default depends on --pen)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    K = np.load(args.K); dist = np.load(args.dist)
    id_base = PEN_IDBASE[args.pen]
    out_path = args.out or PEN_OUT[args.pen]
    pen_ids = set(range(id_base, id_base + 11))
    BODY, tip, _, _ = build_layout(id_base=id_base)

    def detect(paths):
        res = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                continue
            cs, ids, _ = detector.detectMarkers(im)
            if ids is None:
                continue
            res.append({int(i): c.reshape(4, 2) for c, i in zip(cs, ids.flatten())})
        return res

    # ---- gather input (live OR photos); refuse to run on nothing -------------
    dc_dets, piv_dets = None, None
    if args.live:
        dc_dets, piv_dets = live_capture(args.camera, pen_ids)
        if dc_dets is None:
            print("[cancel] no file written."); return
    elif args.images:
        dc_dets = detect(sorted(sum([glob.glob(g) for g in args.images], [])))
        if args.pivot:
            piv_dets = detect(sorted(sum([glob.glob(g) for g in args.pivot], [])))
    else:
        raise SystemExit(
            f"[abort] no input given. Use --live, or --images (and --pivot).\n"
            f"         Refusing to overwrite {out_path} with the ideal layout.")

    if not dc_dets:
        raise SystemExit("[abort] no usable DC views — nothing saved.")

    # ---- bundle adjustment (DC) + optional pen-tip pivot --------------------
    BODY = bundle_adjust(dc_dets, id_base=id_base)
    if piv_dets:
        poses = []
        for det in piv_dets:
            p = ape_pose(det, BODY)
            if p:
                poses.append((cv2.Rodrigues(p[0])[0], p[1]))
        if len(poses) >= 3:
            tip = pivot_tip(poses)
            print(f"[pivot] tip in body frame (mm): {np.round(tip*1000, 2)}")
        else:
            print(f"[pivot] only {len(poses)} usable views (<3) — keeping the ideal tip.")

    ids = np.array(sorted(BODY)); corners = np.array([BODY[i] for i in ids])
    np.savez(out_path, ids=ids, corners=corners, tip=tip)
    print(f"[saved] {out_path}  ({len(ids)} markers, pen {args.pen})")


# ---------------------------------------------------------------------------
def selftest():
    global K, dist
    K = np.array([[1100, 0, 640], [0, 1100, 360], [0, 0, 1]], float); dist = np.zeros(5)
    rng = np.random.default_rng(0)
    BODY0, _, _, INFO = build_layout()
    # ground-truth marker poses = ideal + small gluing perturbation (anchor fixed)
    GT = {}
    for m in BODY0:
        rv, t = corners_to_pose(BODY0[m])
        if m != ANCHOR_ID:
            rv = rv + np.deg2rad(2.0) * rng.standard_normal(3)
            t = t + 0.001 * rng.standard_normal(3)
        GT[m] = pose_to_corners(rv, t)
    # synth views
    dets = []
    for _ in range(45):
        rvv = 0.9 * rng.standard_normal(3); R, _ = cv2.Rodrigues(rvv)
        tvv = np.array([rng.uniform(-.04, .04), rng.uniform(-.04, .04), 0.45])
        det = {}
        for m, (c, n, up) in INFO.items():
            if (R @ n)[2] < -0.15:                     # facing camera
                Pc = (R @ GT[m].T).T + tvv
                px, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, dist)
                det[m] = px.reshape(4, 2) + 0.3 * rng.standard_normal((4, 2))
        if len(det) >= 2:
            dets.append(det)
    # error before vs after calibration (corner distance to GT, in mm)
    def err(BODY):
        e = [np.linalg.norm(BODY[m] - GT[m], axis=1).max() for m in GT]
        return max(e) * 1000
    print(f"[selftest] ideal-vs-GT max corner error  = {err(BODY0):.2f} mm")
    CAL = bundle_adjust(dets)
    print(f"[selftest] calibrated-vs-GT max corner err = {err(CAL):.2f} mm")
    # pivot test
    c_true = np.array([0.0, 0.0, -0.2017])
    poses = []
    for _ in range(20):
        R, _ = cv2.Rodrigues(0.7 * rng.standard_normal(3))
        p_fixed = np.array([0.1, 0.0, 0.6])
        t = p_fixed - R @ c_true
        poses.append((R, t))
    c_est = pivot_tip(poses)
    print(f"[selftest] pivot tip error = {np.linalg.norm(c_est-c_true)*1000:.4f} mm")


if __name__ == "__main__":
    main()
