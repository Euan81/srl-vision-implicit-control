## Calibration

Three one-time procedures, in this order. Each writes artefacts that
`orient_experiment.py` loads from its working directory.

| Script | Produces | Target |
|---|---|---|
| `calibrate_camera.py` | `camera_matrix.npy`, `dist_coeffs.npy` | `chessboard_9x6_27mm_A4.pdf` |
| `handeye_calib.py` | `T_EE_cam.npz` (key `T_EE_cam`) | `charuco_5x7_35_26_A4.pdf` |
| `dodeca_calibrate.py` | Per-marker pen layouts and tip offset | The DodecaPens themselves |

`pose_utils.py` holds the shared SE(3)/SO(3) helpers and the `PoseFilter` used
throughout: geodesic blending on SO(3), linear on translation, with the
estimate held for up to 15 frames through dropout (alpha_R = alpha_t = 0.50,
`max_stale` = 15 — the values in Appendix B).

### 1. Intrinsics

Turn autofocus **off** and leave the manual focus where the live run will use
it. Do not refocus between intrinsics, hand-eye and the session — the
calibration is only valid at that focus. Hold the board across diverse angles
and distances within roughly 0.10-0.40 m; the coverage map shows which image
regions remain uncovered. At least 25 captures.

### 2. Hand-eye

Fix the ChArUco board rigidly, put the arm in gravity-compensation mode, and
capture at least 15 poses with heavy variation in wrist rotation — small wrist
rotations at different arm positions give the best conditioning.

### 3. DodecaPen

Offline bundle adjustment following Wu et al. (UIST 2017), correcting the
hand-gluing error in each marker's pose, with one marker fixed to remove gauge
freedom. The tip is then recovered by holding it fixed against a surface while
the body moves, giving a linear least-squares solve for the tip in the body
frame. The implementation minimises marker-corner reprojection error rather
than the paper's photometric cost; the deviation and its rationale are stated
in the script header.

### Values used in the study

- Intrinsics: fx 1510.03, fy 1506.79, cx 1310.97, cy 709.20 at 2560x1440
  (88.4 deg diagonal FoV). Distortion k1 0.1080, k2 -0.1131, p1 0.00204,
  p2 0.0000695, k3 0.
- Hand-eye: camera 302 mm from the end-effector frame; position error 4.23 mm,
  rotation error 0.507 deg, from 46 image-pose pairs. The subsequent
  Levenberg-Marquardt refinement gave reprojection RMS 6.109 px, position
  residual 3.42 mm and rotation residual 0.582 deg. This RMS is the hand-eye
  refinement figure and is not comparable to the sub-pixel target used during
  intrinsics calibration.
- Pens: 11 markers each (15 mm), marker centres 22.27-23.00 mm (pen A) and
  22.27-23.27 mm (pen B) from the body origin; tip offset 121.7 mm along -z.

Intrinsics are resolution-specific: recalibrate if capture resolution changes.
Print both targets at 100% and check the 50 mm scale bar — a scaled print
gives a plausible-looking calibration with a wrong metric scale.

### Targets

| File | Geometry |
|---|---|
| `chessboard_9x6_27mm_A4.pdf` | 9x6 interior corners, 27 mm squares |
| `charuco_5x7_35_26_A4.pdf` | 5x7, 35 mm squares, 26 mm markers, DICT_4X4_50 |

**Note.** The ChArUco target uses DICT_4X4_50, the same dictionary as the task
board (8 markers, 20 mm) and the two DodecaPens (11 markers each, 15 mm),
which together occupy IDs 0-29. The ChArUco markers reuse IDs in that range,
so the calibration target and the task objects must never be in frame at the
same time.