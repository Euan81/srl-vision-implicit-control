"""
calibrate_camera.py — Camera intrinsics calibration for the Logitech C270 (or any webcam).

Saves camera_matrix.npy and dist_coeffs.npy in the current directory,
ready for use by handeye_calib.py and approach.py.

Board: 9 × 6 interior corners, 27 mm squares  (matches gen_chessboard_a4.py)
Resolution: 2560 × 1440 (2K)  (must match handeye_calib.py AND approach6dof.py)

FOCUS: turn Auto Focus OFF on the camera and leave the manual focus slider at
the SAME position you will run approach6dof.py with. Hold the board within the
fixed-focus depth of field (~0.10–0.40 m for this rig) so every capture is
SHARP — the sharpness gate below rejects blurry frames. Do not refocus between
this calibration, the hand-eye calibration, and the live run.

Workflow
--------
1.  Hold the printed chessboard in front of the camera.
2.  Move it to DIVERSE angles — tilt left/right/up/down, rotate, vary distance.
    The coverage map in the top-right shows which image regions you've hit.
3.  Hold STILL — the script auto-captures when the board is detected and stable.
    Press SPACE to force-capture immediately.
4.  Collect at least MIN_IMAGES (default 25).  Press ENTER to calibrate.
5.  Aim for reprojection error < 0.5 px.  > 1.0 px means redo with better images.

Keys
----
  SPACE   force capture (board must be detected)
  ENTER   run calibration (requires >= MIN_IMAGES)
  u       undo last capture
  Esc     quit

Requirements: opencv-contrib-python, numpy, Pillow
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATTERN_SIZE  = (9, 6)      # interior corners (cols, rows)
SQUARE_MM     = 27.0        # physical square side in mm
CAMERA_INDEX  = 0
WIDTH, HEIGHT = 2560, 1440  # 2K — MUST match handeye_calib.py and approach6dof.py
FOURCC        = "MJPG"      # force MJPEG so 2K fits over USB at full frame rate
MIN_IMAGES    = 25          # minimum captures before calibration allowed

# Auto-capture tuning
STABLE_FRAMES   = 12        # consecutive frames board must be still
STABLE_MAX_PX   = 2.5       # max corner movement (px) to count as "still"
DIVERSE_MIN_PX  = 120       # min mean corner shift from nearest prior capture (px; raised for 2K)
CAPTURE_COOLDOWN = 1.5      # seconds between auto-captures

# Focus / sharpness gate. A fixed-focus lens is only sharp within its depth of
# field; a soft board wrecks corner localisation and biases the intrinsics.
# Reject frames whose board region is blurry (variance of Laplacian too low).
# The threshold is lighting/exposure dependent — watch the live "Sharpness"
# readout in the HUD and set this to ~50–70 % of the value you see when the
# board is crisp and well lit.
SHARPNESS_MIN = 100.0       # min Laplacian variance over the board ROI to accept

# ---------------------------------------------------------------------------
# 3-D object points for one board view  (Z = 0 plane)
# ---------------------------------------------------------------------------
_objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
_objp[:, :2] = np.mgrid[
    0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]
].T.reshape(-1, 2) * (SQUARE_MM / 1000.0)   # metres

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
_find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
               cv2.CALIB_CB_NORMALIZE_IMAGE |
               cv2.CALIB_CB_FAST_CHECK)

def detect(gray):
    """Return (corners, ok) — corners are subpixel-refined if found."""
    ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, _find_flags)
    if ok:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    return corners, ok

def sharpness_of(gray, corners):
    """Focus metric: variance of the Laplacian over the board's bounding box.
    Higher = sharper. Restricted to the board region so a blurry/busy
    background doesn't skew the score."""
    pts = corners.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    h, w = gray.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())

# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------
def is_stable(history):
    """True if corners barely moved over the last STABLE_FRAMES frames."""
    if len(history) < STABLE_FRAMES:
        return False
    ref = history[0].reshape(-1, 2)
    for c in list(history)[1:]:
        if np.mean(np.linalg.norm(c.reshape(-1, 2) - ref, axis=1)) > STABLE_MAX_PX:
            return False
    return True

def is_diverse(corners, captured):
    """True if corners differ enough from every previous capture."""
    if not captured:
        return True
    flat = corners.reshape(-1, 2)
    for prev in captured:
        d = np.mean(np.linalg.norm(flat - prev.reshape(-1, 2), axis=1))
        if d < DIVERSE_MIN_PX:
            return False
    return True

# ---------------------------------------------------------------------------
# Coverage map  (top-right inset showing which image regions are covered)
# ---------------------------------------------------------------------------
CMAP_W, CMAP_H = 200, 113   # proportional to 1280×720

def build_coverage(captured_corners, img_w, img_h):
    cmap = np.zeros((CMAP_H, CMAP_W, 3), dtype=np.uint8)
    sx = CMAP_W / img_w
    sy = CMAP_H / img_h
    for corners in captured_corners:
        for pt in corners.reshape(-1, 2):
            px = int(pt[0] * sx)
            py = int(pt[1] * sy)
            if 0 <= px < CMAP_W and 0 <= py < CMAP_H:
                cv2.circle(cmap, (px, py), 2, (0, 200, 80), -1)
    cv2.rectangle(cmap, (0, 0), (CMAP_W-1, CMAP_H-1), (160, 160, 160), 1)
    return cmap

# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def put(img, text, pt, color=(0, 220, 80), scale=0.50, thickness=1):
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def draw_hud(frame, n, board_ok, stable, diverse, sharp_val, sharp_ok, last_cap_t):
    h, w = frame.shape[:2]

    # Border colour
    if board_ok and stable and diverse and sharp_ok:
        bcol = (0, 255, 120)    # green — about to capture
    elif board_ok:
        bcol = (0, 200, 220)    # yellow — detected but not ready
    else:
        bcol = (40, 40, 200)    # red — not detected
    cv2.rectangle(frame, (0, 0), (w-1, h-1), bcol, 4)

    # Status
    if not board_ok:
        msg, col = "Board NOT detected", (40, 80, 255)
    elif not sharp_ok:
        msg, col = f"Too blurry ({sharp_val:.0f}) — fix focus / move to focus range", (40, 120, 255)
    elif not stable:
        msg, col = "Hold still...", (40, 200, 230)
    elif not diverse:
        msg, col = "Too similar to a previous pose — move more", (40, 180, 255)
    else:
        msg, col = "CAPTURING...", (0, 255, 120)
    put(frame, msg, (10, 32), color=col)

    put(frame, f"Captures: {n} / {MIN_IMAGES} min", (10, 60))

    scol = (0, 220, 80) if sharp_ok else (40, 120, 255)
    put(frame, f"Sharpness: {sharp_val:.0f} / {SHARPNESS_MIN:.0f} min", (10, 86), color=scol)

    hints = [
        "SPACE  force capture",
        f"ENTER  calibrate {'(ready)' if n >= MIN_IMAGES else f'(need {MIN_IMAGES-n} more)'}",
        "u      undo last",
        "Esc    quit",
    ]
    for i, t in enumerate(hints):
        put(frame, t, (10, h - 20 - (len(hints)-1-i)*22),
            color=(200, 200, 200), scale=0.42)

    # Tip rotation
    tips = [
        "Tilt toward camera",
        "Tilt away from camera",
        "Rotate 45°",
        "Move closer",
        "Move further away",
        "Top-left corner",
        "Bottom-right corner",
        "Fill the frame",
    ]
    tip = tips[n % len(tips)]
    put(frame, f"Next: {tip}", (10, h - 20 - len(hints)*22 - 10),
        color=(180, 180, 80), scale=0.44)

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
MIN_IMAGES_AFTER_PRUNING = 10   # never prune below this count
PRUNE_THRESHOLD_PX       = 1.0  # remove images with per-image error above this

def _run_once(obj_pts, img_pts, img_size):
    """Single calibration pass. Returns (rms, K, dist, per_image_errors)."""
    # Standard 5-parameter model (k1, k2, p1, p2, k3).
    # CALIB_FIX_K3 pins k3=0 for extra stability with webcams.
    # Do NOT use CALIB_RATIONAL_MODEL here — it has 8+ parameters and
    # overfits badly unless you have hundreds of very high-quality images.
    flags = cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None, flags=flags
    )
    errors = []
    for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
        errors.append(float(np.mean(
            np.linalg.norm(ip.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        )))
    return rms, K, dist, errors


def calibrate_with_pruning(obj_pts_in, img_pts_in, img_size):
    """
    Iteratively calibrate and remove the single worst image each round
    until all per-image errors are below PRUNE_THRESHOLD_PX, or until
    MIN_IMAGES_AFTER_PRUNING images remain.

    Returns (rms, K, dist, errors, kept_indices, removed_indices).
    """
    indices = list(range(len(obj_pts_in)))
    removed = []

    round_n = 0
    while True:
        obj = [obj_pts_in[i] for i in indices]
        img = [img_pts_in[i] for i in indices]
        rms, K, dist, errors = _run_once(obj, img, img_size)

        worst_local = int(np.argmax(errors))
        worst_err   = errors[worst_local]

        if worst_err <= PRUNE_THRESHOLD_PX:
            break   # all images within threshold — done

        if len(indices) <= MIN_IMAGES_AFTER_PRUNING:
            print(f"  [WARN] Stopped pruning at {len(indices)} images "
                  f"(floor={MIN_IMAGES_AFTER_PRUNING}). "
                  f"Worst remaining error: {worst_err:.3f} px.")
            break

        worst_global = indices[worst_local]
        removed.append((worst_global, worst_err))
        indices.pop(worst_local)
        round_n += 1
        print(f"  Round {round_n}: removed image {worst_global+1:>2} "
              f"(error={worst_err:.3f} px)  —  {len(indices)} remain")

    return rms, K, dist, errors, indices, removed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # MJPG BEFORE the size: 2K (and even 1080p) uncompressed exceeds USB-2
    # bandwidth, so the driver silently drops the frame rate or the resolution.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}.")

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fcc_int    = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_str    = "".join(chr((fcc_int >> 8 * i) & 0xFF) for i in range(4))
    print(f"Camera opened at {actual_w}×{actual_h} @ {actual_fps:.0f}fps  fourcc={fcc_str}")

    # HARD GUARD: refuse to calibrate at the wrong resolution. Intrinsics are
    # resolution-specific; a K built at anything other than what approach6dof.py
    # runs at makes every pose wrong. This is the #1 silent calibration bug.
    if (actual_w, actual_h) != (WIDTH, HEIGHT):
        print(f"\n[ABORT] Requested {WIDTH}×{HEIGHT} but the camera delivered "
              f"{actual_w}×{actual_h}.")
        print(f"        The 2K mode is not reaching OpenCV. Fixes to try:")
        print(f"          • select 2K in the camera's own app AND replug it,")
        print(f"          • confirm the cable/port is USB-3 (blue) for 2K bandwidth,")
        print(f"          • or set WIDTH,HEIGHT here to the resolution shown above")
        print(f"            and match handeye_calib.py + approach6dof.py to it.")
        cap.release()
        raise SystemExit(1)
    if actual_fps and actual_fps < 20:
        print(f"[WARN] Only {actual_fps:.0f} fps — expect motion blur; move slowly "
              f"and pause before each capture.")

    obj_pts_list  = []       # 3-D points per capture
    img_pts_list  = []       # 2-D corners per capture
    cap_corners   = []       # for diversity check and coverage map

    corner_hist   = deque(maxlen=STABLE_FRAMES)
    last_cap_t    = 0.0
    stable        = False
    diverse       = False

    print(f"\nWave the chessboard in front of the camera.")
    print(f"Auto-capture fires when board is steady and new enough.")
    print(f"Collect {MIN_IMAGES}+ images then press ENTER.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, board_ok = detect(gray)

        sharp_val = 0.0
        sharp_ok  = False
        if board_ok:
            sharp_val = sharpness_of(gray, corners)
            sharp_ok  = sharp_val >= SHARPNESS_MIN
            corner_hist.append(corners.copy())
            stable  = is_stable(corner_hist)
            diverse = is_diverse(corners, cap_corners)
            cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, board_ok)
        else:
            corner_hist.clear()
            stable  = False
            diverse = False

        n = len(obj_pts_list)
        draw_hud(frame, n, board_ok, stable, diverse, sharp_val, sharp_ok, last_cap_t)

        # Coverage map inset (top-right)
        cmap = build_coverage(cap_corners, actual_w, actual_h)
        frame[8 : 8+CMAP_H, actual_w-CMAP_W-8 : actual_w-8] = cmap
        put(frame, "Coverage", (actual_w-CMAP_W-8, 8+CMAP_H+14),
            color=(160,160,160), scale=0.38)

        cv2.imshow("Camera Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Auto-capture ──────────────────────────────────────────────────
        now = time.time()
        auto = (board_ok and stable and diverse and sharp_ok and
                now - last_cap_t > CAPTURE_COOLDOWN)

        # SPACE force-captures even if soft, but warn so a blurry frame isn't
        # added to the calibration set unknowingly.
        force = (key == ord(' ') and board_ok)
        if force and not sharp_ok:
            print(f"[WARN] Forced capture is blurry (sharpness {sharp_val:.0f} < "
                  f"{SHARPNESS_MIN:.0f}) — it may degrade the calibration.")
        do_capture = auto or force

        if do_capture:
            obj_pts_list.append(_objp.copy())
            img_pts_list.append(corners.copy())
            cap_corners.append(corners.copy())
            last_cap_t = now
            corner_hist.clear()   # reset stability window
            print(f"[+] Capture {len(obj_pts_list):>2}")

        # ── Undo ─────────────────────────────────────────────────────────
        elif key == ord('u'):
            if obj_pts_list:
                obj_pts_list.pop()
                img_pts_list.pop()
                cap_corners.pop()
                print(f"[-] Removed last.  {len(obj_pts_list)} remaining.")
            else:
                print("[INFO] Nothing to undo.")

        # ── Calibrate ────────────────────────────────────────────────────
        elif key == 13:
            n = len(obj_pts_list)
            if n < MIN_IMAGES:
                print(f"[SKIP] Need {MIN_IMAGES}, have {n}.")
                continue

            print(f"\nCalibrating with {n} images…")
            print(f"Pruning images with error > {PRUNE_THRESHOLD_PX} px:\n")

            rms, K, dist, errors, kept, removed = calibrate_with_pruning(
                obj_pts_list, img_pts_list, (actual_w, actual_h)
            )

            print(f"\n{'─'*52}")
            if removed:
                print(f"  Removed {len(removed)} image(s):  "
                      + ", ".join(f"#{i+1} ({e:.2f}px)" for i, e in removed))
            else:
                print(f"  No images removed — all within {PRUNE_THRESHOLD_PX} px.")
            print(f"  Images used for final calibration: {len(kept)} / {n}")
            print(f"{'─'*52}")
            print(f"  Overall RMS reprojection error : {rms:.4f} px")
            if rms < 0.5:
                grade, col = "EXCELLENT", "\033[92m"
            elif rms < 1.0:
                grade, col = "GOOD",      "\033[93m"
            else:
                grade, col = "POOR — collect more diverse images", "\033[91m"
            print(f"  Grade  : {col}{grade}\033[0m")
            print(f"{'─'*52}")
            print(f"\n  Per-image errors after pruning (px):")
            for local_i, (global_i, e) in enumerate(zip(kept, errors)):
                bar = "█" * int(e / 0.05)
                print(f"    img {global_i+1:>2}:  {e:.3f}  {bar}")

            print(f"\n  Camera matrix K:\n{K}")
            print(f"\n  Distortion coeffs:\n{dist.flatten()}")

            np.save("camera_matrix.npy", K)
            np.save("dist_coeffs.npy",   dist)
            print(f"\nSaved: camera_matrix.npy  dist_coeffs.npy")

            if rms >= 1.0:
                print("\n[WARN] RMS > 1 px even after pruning.")
                print("       Ensure the board is printed flat, well-lit, and not blurry.")
            else:
                print("\nCamera calibration complete.  Re-run handeye_calib.py now.")
            break

        # ── Quit ─────────────────────────────────────────────────────────
        elif key == 27:
            print("Quit — no files saved.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
