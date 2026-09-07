import os
import csv
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
import numpy as np
import pylsl
import time
import threading
import atexit
from cv2 import aruco
from transforms import TransformManager

# Full-pipeline two-pen tracker (APE + ICT + DPR). Provides K, dist, PENS, detect.
# Imported defensively so the grasp workflow still runs if the pens module or its
# calibration is absent — in that case pen-driven ORIENT is simply disabled.
try:
    import dodecapen_tracker as dt
    PENS_AVAILABLE = True
except Exception as _pen_e:                     # noqa: BLE001 (want any import failure)
    dt = None
    PENS_AVAILABLE = False
    print(f"[WARN] dodecapen_tracker unavailable ({_pen_e}); "
          f"pen-driven ORIENT disabled.")

# Google MediaPipe Hands (GMH) — used for the finger-touch handover release.
# Implemented with the Tasks HandLandmarker API EXACTLY like the attached test2.py,
# NOT the legacy 'mediapipe.solutions' module. The legacy solutions API is not
# shipped on newer Python builds (e.g. Python 3.14 / mediapipe 0.10.x), where
# 'import mediapipe' succeeds but 'mediapipe.solutions' is absent — the Tasks API
# is the one that runs there, so this file uses it exclusively.
# Imported defensively: everything else still runs if mediapipe is missing
# entirely (in that case the finger-touch release is disabled and the manual 'h'
# key still forces a release).
mp = mp_python = vision = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
    print(f"[MediaPipe] {getattr(mp, '__version__', '?')} — Tasks HandLandmarker backend")
except Exception as _mp_e:                      # noqa: BLE001
    mp = None; mp_python = None; vision = None; MP_AVAILABLE = False
    print(f"[WARN] mediapipe unavailable ({_mp_e}); finger-touch release disabled.")

# ===========================================================================
#  approach6dof_merged.py — full 6-DOF (SE(3)) grasp servo for the Unitree Z1
# ===========================================================================
#
#  WHAT THIS FILE IS
#  -----------------
#  Best-of merge: ArUco detection pipeline from approach6dofopus.py (better
#  per-marker visualisation, in-plane/side approach geometry) combined with
#  accuracy improvements from approach6dof1706.py (single-board PnP, on-
#  manifold pose filter, TCP-correct servo, debounced lock, latency fix).
#
#  FROM approach6dof1706.py  (accuracy improvements)
#  -------------------------------------------------
#  1. SINGLE RIGID-BODY BOARD PnP. All detected marker corners are stacked
#     into one solvePnPGeneric(SOLVEPNP_IPPE) call over the full 140 mm
#     board. Per-marker PnP uses only a 20 mm baseline -> large rotation
#     noise. The joint solve uses the widest available baseline and yields
#     lower rotation variance. Axis-angle averaging across per-marker results
#     is not a valid mean on SO(3) (Hartley et al., "Rotation Averaging",
#     IJCV 2013) and is replaced entirely by this single solve.
#     Planar two-fold ambiguity disambiguated by temporal continuity (once
#     primed) then lowest reprojection error among forward-facing candidates
#     (Collins & Bartoli, IJCV 2014).
#
#  2. ON-MANIFOLD POSE FILTER in the BASE frame. Exponential low-pass:
#     geodesic blend on SO(3), linear on R^3 (Sola, Deray & Atchuthan,
#     "A micro Lie theory", arXiv:1812.01537, 2018; Moakher 2002). Because
#     the board is static in the base frame the held estimate stays valid
#     through marker dropout / occlusion (up to MAX_STALE frames), and the
#     servo target stops jittering.
#
#  3. TCP / TOOL-OFFSET CORRECTION in se3_control. The SDK integrates the
#     FLANGE (EE) frame. Mapping the tip target to an EE target through the
#     constant T_EE_tip before computing the se3_log error removes the
#     omega x r lever-arm error that arises when the tip-origin velocity is
#     sent on the world-linear channel while omega acts at the flange.
#     (Modern Robotics adjoint, Lynch & Park 2017; Murray, Li & Sastry 1994.)
#
#  4. FK PULLED ONCE PER FRAME. T_base_cam and T_base_tip are both derived
#     from the same get_T_base_EE() sample, eliminating the timing mismatch
#     that arose when calling the SDK twice per frame.
#
#  5. LOCK-CRITERION DEBOUNCING. LOCK_FRAMES consecutive in-threshold frames
#     are required before the orientation is frozen for the straight-in grasp,
#     preventing a single noisy measurement from triggering the lock.
#
#  6. CAP_PROP_BUFFERSIZE = 1. Prevents the capture queue from backing up,
#     keeping the perception -> command latency at one frame.
#
#  FROM approach6dofopus.py  (kept)
#  ---------------------------------
#  * IN-PLANE APPROACH DIRECTION. The gripper approaches along the arm->object
#    vector projected onto the marker plane (not along the pure board normal).
#    Correct for side-grasping the dumbbell's flat face from the edge.
#  * GRASP_Z_OFFSET: centring offset along the surface normal to place the
#    jaws at the midplane of the flat section (half the slab thickness).
#  * PER-MARKER VISUALISATION: frame axes and ID label drawn at each detected
#    marker's location (derived from the board pose, not from a per-marker
#    PnP), giving clear visual feedback during setup and servo.
#  * MARKER_LAYOUT / BOARD GEOMETRY (8 markers on a 160 mm board, ±68 mm offsets).
#
# ===========================================================================

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_INDEX    = 0
MARKER_SIZE     = 0.020     # metres per marker side

PRE_GRASP_DIST  = 0.160     # standoff back along approach axis (m)
GRASP_DEPTH     = 0.1025    # grasp depth back along approach axis (m)
GRASP_Z_OFFSET  = -0.030   # offset along surface normal (m); for the 160 mm board object
# Systematic VERTICAL correction for the grasp (base +Z = up). 0 = no offset (removed). Set a small
# +/- value here only if the grasp is consistently high/low again.
GRASP_VERTICAL_OFFSET = 0.0


POS_THRESH      = 0.010     # 10 mm: commit to the insertion sooner. GRASP re-servos
                            # position anyway (only orientation is frozen at lock), so a
                            # looser standoff tolerance costs no final accuracy — it just
                            # stops the arm hovering in front waiting to nail the standoff.
ROT_THRESH      = np.deg2rad(2.0)   # 2.5 deg: above the ~1.8 deg servo residual so the lock
                                    # actually triggers instead of the arm hesitating while
                                    # the PID integral slowly closes the last degree.
STOP_DIST       = 0.015     # 15 mm: gripper-close trigger (was 5 mm) — larger = less sensitive,
                            # closes with more margin instead of requiring the tip to reach exactly
STABLE_FRAMES   = 3         # consecutive detections before DETECT -> SERVO
LOCK_FRAMES     = 3         # was 5 — fewer in-threshold frames -> commits to insertion faster
DEBUG_EVERY     = 30

# Servo stall watchdog. If a SERVO/GRASP move makes no real progress for
# STALL_TIMEOUT seconds, abort to STATE_FAULT (arm halts, gripper holds) instead
# of pushing into an unreachable target / joint limit indefinitely. The Python
# control layer has no reachability model, and the C++ bridge keeps re-issuing the
# last twist, so a stall must be detected here. STALL_MIN_IMPROVE is the smallest
# pos_err drop (m) that counts as progress. STALL_NEAR_TARGET: once pos_err is
# within this of zero the arm has essentially arrived, so a plateau there is NOT a
# stall (it's just waiting on the orientation lock / debounce) — only a plateau
# while still FAR from target counts as stuck-against-a-limit.
STALL_TIMEOUT     = 4.0
STALL_MIN_IMPROVE = 0.001
STALL_NEAR_TARGET = 0.020

# Gains. Twist (omega_b [rad/s-ish], v_b [m/s-ish]) before plant scaling.
# Kp_rot lowered 3.0 -> 1.5: the pure-P servo limit-cycled ~1.8 deg at Kp_rot=3
# (proportional gain + loop latency = hunting). Lower gain shrinks the hunt so the
# orientation settles below ROT_THRESH instead of loosening the tolerance. Raise
# back toward 3 if convergence is too slow; if rot_err still floors ~1.8 deg with
# the arm nearly still, it's pose jitter -> lower FILTER_ALPHA_R instead.
Kp_pos          = 3.0
Kp_rot          = 1.5
# SE(3) servo integral/derivative gains — make se3_control a full PID. The integral
# removes the steady-state position/orientation residual a pure-P velocity servo
# leaves (arm stiction, load droop, loop latency). Kd defaults low: a derivative on
# the noisy monocular pose error amplifies noise, so raise Kd_rot only to damp a
# visible limit-cycle. Set Ki_* to 0 to recover pure-P.
Ki_pos          = 1.0
Kd_pos          = 0.01
Ki_rot          = 0.5
Kd_rot          = 0.01
SE3_I_CLAMP_LIN = 0.05   # anti-windup clamp on the linear error integral (m·s)
SE3_I_CLAMP_ANG = 0.5    # anti-windup clamp on the angular error integral (rad·s)
MAX_LIN         = 0.75  # was 0.75 — raise the linear-speed ceiling for faster approach/lift
MAX_ANG         = 1.00

# GRASP insertion tuning. The target is FROZEN at lock and FK is clean, so the default
# servo gains (integral 0.6x with no position damping) hunt and make the insertion
# hesitate. For GRASP only: cut most of the integral, add position damping, and cap the
# linear speed so it pushes in smoothly. Raise GRASP_KI_POS if it stops short of the
# object; raise GRASP_KD_POS if it still oscillates.
GRASP_KI_POS = 0.6      # linear integral (was Ki_pos*0.6 = 0.6)
GRASP_KD_POS = 0.08     # linear damping   (was Kd_pos*0.6 = 0.0)
GRASP_KI_ROT = 0.15     # angular integral (orientation is locked, so small)
GRASP_KD_ROT = 0.02
GRASP_MAX_LIN = 0.20    # gentle insertion speed cap (vs MAX_LIN 0.75)

# MERGED APPROACH+GRASP commit band. Approach and grasp are one continuous motion:
# the tip is servoed straight to the FINAL grasp pose (not a standoff), so the EE
# always heads toward the object. The consistent computed in-plane grasp orientation
# (compute_grasp / build_grasp_orientation) is tracked live the whole way, so the
# gripper is always oriented to grasp the same way. Near contact the CONSISTENT
# computed grasp pose is frozen and the servo switches to the gentle insertion gains
# above for a clean, repeatable straight-in grasp (immune to the eye-in-hand markers
# leaving frame at close range). COMMIT_DIST: tip-to-grasp distance under which — once
# orientation has also converged (< ROT_THRESH, debounced LOCK_FRAMES) — the pose is
# frozen. HARD_COMMIT_DIST: freeze unconditionally this close, so the last cm never
# rides a jittery live target even if alignment lags. Raise COMMIT_DIST for a longer
# gentle final run-in; lower it to track the object live for longer before committing.
COMMIT_DIST      = 0.060    # m: freeze the computed grasp pose once this close + aligned
HARD_COMMIT_DIST = 0.030    # m: freeze unconditionally this close (alignment fallback)

# PRESENT / HOLD orientation control — PID, runs at ORIENT_HZ in a separate thread.
# Tune Kp_present up until it converges fast; Kd_present up to kill overshoot;
# Ki_present up to drive out any steady-state offset (arm stiction / load droop).
# Set Ki_* to 0 to recover the old PD behaviour.
Kp_present      = 1.2      # was 0.3 (tau~3.3s -> ~0.8s); PRESENT levelling now ~4x faster
Ki_present      = 0.10
Kd_present      = 0.02      # was 0.005; more damping to keep the faster loop from overshooting
ORIENT_HZ       = 100
# Integral anti-windup: each axis of the error integral (units rad·s) is clamped to
# this, so the I term can't wind up while the arm slews or the output saturates at
# MAX_ANG. Keep it small — just enough to overcome the steady-state offset.
ORIENT_I_CLAMP  = 0.5

# ORIENT — PEN-DRIVEN (replaces the old fixed demo sweep). The same 100 Hz PD
# thread is reused, but instead of stepping a scripted setpoint on a timer the
# ORIENT state watches the two DodecaPens (pens_mean_tilt logic, ported below),
# classifies which discrete tilt the pens' mean direction points to, and — once
# that classification holds steady for ORIENT_STABLE_T — commands that single
# tilt (±30° pitch or roll) for ORIENT_HOLD_T seconds, then returns to level and
# watches again (repeatable). Kp_orient is raised above Kp_present so a 30 deg
# move settles quickly: closed-loop error decays as exp(-Kp*t), tau = 1/Kp; at
# Kp_orient=1.5, tau≈0.67 s reaches ~29.7 deg by 3 s (peak rate under MAX_ANG).
ORIENT_SWEEP_DEG = 30.0     # commanded tilt magnitude (deg) — ±pitch or ±roll
Kp_orient        = 0.5
Ki_orient        = 0.10
Kd_orient        = 0.02
# Slew-rate limit on the orientation command (rad/s per second). When a new tilt is
# commanded the setpoint steps by 30 deg, which would step the velocity command to full
# speed on the very first tick -> a jerky start. Capping how fast the command can change
# ramps it up over ~MAX_ANG/ORIENT_MAX_ANG_ACC s (~0.25 s), smoothing the first step.
ORIENT_MAX_ANG_ACC = 4.0

ORIENT_STABLE_T     = 0.3   # s the pen classification must hold before committing
ORIENT_HOLD_T       = 10.0  # s to hold the commanded tilt before returning (fallback
                            # if no pair press arrives to end it early)
# After a pair press (logger 'PAIR' marker) ends the tilt hold, wait this long before
# actually returning to level, so the arm doesn't start moving while the operator is
# still pressing the buttons. Gives time to release the pair first.
ORIENT_INTERRUPT_DELAY = 0.5   # s post-press settle before the return-to-level

# Open-palm RELEASE GUARD: never release while the object is tilted (mid-ORIENT) or
# while a pen is being tracked (operator still gesturing). LEVEL_TOL is the max
# roll/pitch that still counts as "level" (the pen tilts are ±30°, so 10° is safe).
LEVEL_TOL = np.deg2rad(10.0)
ORIENT_RETURN_MAX_T = 4.0   # s cap on the return-to-level settle before re-watching
ORIENT_MISS_TOL     = 6      # tolerated consecutive frames with <2 pens before the
                             # steadiness window + stability timer are reset. Brief
                             # recognition dropouts within this many frames HOLD
                             # progress instead of restarting the wait.

# Map a pen-classified label to an absolute base-frame ZYX-Euler setpoint (deg).
LABEL_TO_RPY = {
    "PITCH +30": (0.0,  -ORIENT_SWEEP_DEG, 0.0),
    "PITCH -30": (0.0, ORIENT_SWEEP_DEG, 0.0),
    "ROLL +30":  ( -ORIENT_SWEEP_DEG, 0.0, 0.0),
    "ROLL -30":  (ORIENT_SWEEP_DEG, 0.0, 0.0),
}

# ORIENT — CONTINUOUS PEN-TRACKING SERVO (replaces the discrete tilt classifier above;
# LABEL_TO_RPY / PROTOTYPES / classify_tilt are now unused, kept only for reference).
# Every frame the two pens define a plane (normal = mean/bisector of their long axes)
# and an "intersection" point (closest point of the two pen lines). The grasped board
# is driven continuously so its CENTRE reaches that point and its plane (board Z-axis
# normal) aligns to that normal — full 6-DOF, run from the MAIN loop via se3_control.
# The 100 Hz orientation-only worker is stopped on ORIENT entry so it can't fight the
# linear channel. No workspace clamp (follows the pens fully); freezes on pen dropout.
ORIENT_KP_POS       = 2.0    # linear P (tau=1/KP=0.5s). Sized so a <=20cm move settles to ~5mm in
                             #   <2s (t = ln(0.20/0.005)/KP ~ 1.85s). Was 1.0 (~3s).
ORIENT_KP_ROT       = 3.6    # angular P (tau=0.28s). FASTER: ~40deg move -> ~2deg in ~1.0s. Was 2.5.
ORIENT_TRACK_MAXLIN = 0.35   # linear-speed cap (m/s-ish). 20cm barely reaches it, so mostly gain-led.
ORIENT_TRACK_MAXANG = 1.20   # angular-speed cap (rad/s). FASTER: big face-to-face turns aren't cap-bound. Was 0.70.
# Angular ACCELERATION cap -> controllable, non-abrupt pitch/roll: limits how fast the angular
# command can change (rad/s^2). Lower = gentler, more deliberate board rotation. 0 = off.
ORIENT_ANG_ACCEL_MAX = 9.0    # rad/s^2: FASTER ramp — reach the 1.20 angular cap in ~0.13s. Was 5.0.
# Standoff cap: the board centre tracks the pen-line intersection, but is never placed
# more than ORIENT_STANDOFF_MAX from the pen tips measured ALONG the mean vector. Within
# that range it sits at the intersection; beyond it (lines cross far out, or behind the
# tips) it is pulled back onto the mean vector at this distance. Keeps the board a fixed,
# safe standoff from the operator's pen tips.
ORIENT_STANDOFF_MAX = 0.04   # (two-pen legacy) max board-centre distance from tips along mean vector (m)
# SINGLE-PEN DIRECT HANDLE. ORIENT now tracks ONE pen (penA, the first/right DodecaPen) and
# moves the board so its plane normal follows the pen's pointing axis and its centre sits at the
# pen tip plus this forward offset along the axis. "The object goes where the pen points."
# 5-DOF (plane + point); the spin about the normal stays free/auto. 0.0 -> board centre AT the tip.
ORIENT_HANDLE_PEN    = 0      # which pen index is the handle (0 = penA / first / right)
ORIENT_HANDLE_OFFSET = 0.0    # forward offset of the board centre from the pen tip along the axis (m)
# Low-pass on the pen-defined target board pose (base frame): geodesic on SO(3), linear
# on R^3. Small alpha = smoother arm, more lag. Retained across brief pen dropouts, so a
# dropout FREEZES the arm on the last target instead of stepping it.
ORIENT_TGT_ALPHA_R  = 0.5    # (legacy fixed alpha; superseded by the adaptive scheme below)
ORIENT_TGT_ALPHA_T  = 0.5    # (legacy fixed alpha; superseded by the adaptive scheme below)
# SPEED-ADAPTIVE target smoothing (1-euro-filter style). The blend factor scales from
# ORIENT_ALPHA_MIN (pens still -> heavily smoothed, no jitter) up to ORIENT_ALPHA_MAX (pens
# moving fast -> snappy, minimal lag), so the board FOLLOWS FLUIDLY without ever waiting for
# the pens to settle. The proxy for "fast" is the per-frame gap between the incoming target
# and the current smoothed one (REF_* = the gap at which the blend reaches ALPHA_MAX).
ORIENT_ALPHA_MIN     = 0.45   # blend at rest (0..1; lower = smoother/steadier when still)
ORIENT_ALPHA_MAX     = 0.90   # blend when the target moves fast (higher = snappier)
ORIENT_ALPHA_REF_POS = 0.020  # m:   position gap that reaches ALPHA_MAX
ORIENT_ALPHA_REF_ANG = 0.087  # rad (~5 deg): rotation gap that reaches ALPHA_MAX
# RETURN-TO-INITIAL after a button press: on every PAIR the arm drives back to the start
# ('s'/HOME) pose, then resumes ORIENT tracking for the (advanced) next face.
ORIENT_RETURN_POS_TOL = 0.02   # m:  counts as "reached the initial pose"
ORIENT_RETURN_ANG_TOL = 0.10   # rad (~6 deg)
ORIENT_RETURN_TIMEOUT = 5.0    # s:  resume tracking even if not fully settled by then
# ── ORIENT tracking upgrades (see the STATE_ORIENT block) ────────────────────────────────
# (1) TARGET ROBUSTNESS.
#   * Near-parallel pens make the closest-point intersection ill-conditioned (it jumps far
#     along the pens for small angular noise). If the undirected angle between the two pen
#     axes is below PEN_PARALLEL_MIN_DEG, the target is NOT updated and the arm holds.
#   * The plane normal is a reprojection-error-WEIGHTED mean of the two pen axes so a
#     poorly-seen pen (high PnP reproj) is down-weighted: w_i = 1/(reproj_i + PEN_NORMAL_W_EPS).
PEN_PARALLEL_MIN_DEG = 12.0    # deg; below this the pens are too parallel -> hold the target
PEN_NORMAL_W_EPS     = 0.5     # px; regulariser in the reproj weighting of the normal average
# (2) TARGET-VELOCITY FEED-FORWARD. The (already SE(3)-low-passed) target centre is finite-
#   differenced to estimate its velocity, clamped + lightly smoothed, and added to the servo's
#   linear command so the arm LEADS a moving pen target instead of lagging it.
ORIENT_FF_GAIN  = 1.0     # fraction of the estimated target velocity fed forward (0 = off)
ORIENT_FF_MAX   = 0.15    # clamp on the feed-forward speed (m/s-ish)  (1.5x with the track speed)
# STUCK -> TRANSLATION-ONLY fallback. If the tip stops making progress while there is still error to
# reduce, the 6-DOF target is likely momentarily unreachable (wrist limit / singularity). Drop the
# rotation and keep following the point with translation; retry rotation after a short window (when
# the pose allows it). Raibert & Craig (1981) selection-matrix idea, applied transiently.
ORIENT_STUCK_EPS       = 0.003   # m: tip must move at least this to count as progress
ORIENT_STUCK_ROT_EPS   = 0.0087  # rad (~0.5deg): rot_err shrinking by this also counts as progress.
                                 #   Without this, a mostly-ROTATIONAL correction (tip barely moves)
                                 #   is falsely flagged stuck -> rotation dropped -> board won't turn.
ORIENT_STUCK_T         = 0.8     # s: no progress for this long (with work to do) -> declare stuck
ORIENT_STUCK_RECOVER_T = 1.5     # s: stay translation-only after a stall, then retry rotation
ORIENT_BUTTON_HOLD_T   = 0.5     # s: after a button (PAIR) press, hold the arm still this long so
                                 #   the user can reposition their hands before it moves on
# Fluency-timing thresholds (Study 2, human-robot fluency). Used to timestamp, per presented face,
# when the robot STARTS moving and when it is READY (settled) at the presentation pose.
FD_MOVE_EPS  = 0.02              # m/s-ish: commanded linear speed above this = "robot has started moving"
FD_READY_POS = 0.010             # m:   tip position error below this = at the presentation pose
FD_READY_ROT = np.deg2rad(5.0)   # rad: orientation error below this = settled
ORIENT_FF_ALPHA = 0.4     # EMA on the finite-difference velocity estimate (noise reduction)
# Hand-detection throttle. MediaPipe hand detection is the biggest per-frame cost in ORIENT and
# throttles the whole tracking loop. Run it every N frames and reuse the cached hands in between,
# so the pen-following control rate is higher (snappier). 1 = every frame (no throttle).
ORIENT_HANDS_EVERY = 3
# Board detection is DISABLED during ORIENT: once orienting starts the board is assumed
# fixed (rigid grip, FK-driven control, no slip) until the last button press, so the board
# ArUco detect — the biggest per-frame cost — is skipped entirely and the markers cached at
# ORIENT entry are reused. Set False to restore per-frame board detection during ORIENT.
ORIENT_DETECT_BOARD = False
# (3) removed: the board uses plain minimal rotation to align its plane and never spins about
#   its own normal (no wrist-singularity yaw search).
# BACK-OFF-TO-REACQUIRE. When both pens aren't visible, instead of freezing, retreat the
# (eye-in-hand) camera straight back along its optical axis so more of the scene fits in
# frame and both pens come back into view. Self-terminating: the instant both pens are
# seen again the tracking servo takes over. Bounded by ORIENT_RETREAT_MAX from where the
# pens were lost so it can't back into a joint limit forever; beyond that it just holds.
ORIENT_RETREAT_SPEED = 0.075   # speed of the horizontal move back toward base (m/s-ish)
ORIENT_RETREAT_MAX   = 0.15   # max distance to back up from the loss point (m)

# PEN-LOSS -> SLOW HOME RETRACT. If BOTH pens stay unseen for longer than
# ORIENT_PEN_LOST_HOME_S, stop freezing mid-air and instead ease the tip gently back to the
# fixed HOME pose (FK-only, no perception needed) at a deliberately slow speed. Stays in ORIENT:
# the instant the pens reappear, normal pen-tracking resumes (the PID is reset so it doesn't lurch).
ORIENT_PEN_LOST_HOME_S   = 1.0     # pens unseen this long -> begin the slow retract to home
ORIENT_PEN_LOST_HOME_LIN = 0.06    # slow linear-speed cap for the retract (m/s-ish)
ORIENT_PEN_LOST_HOME_ANG = 0.35    # gentle angular-speed cap for the retract (rad/s)

# ── ORIENT: parallel-plane + LSL-driven face presentation (2807 redesign) ────────────────────
# The board is driven PARALLEL to the pens' plane (normal = cross of the two pen axes) plus a
# fixed presentation tilt, instead of PERPENDICULAR to the pens. The TARGET FACE is taken from the
# runner's LoggerMarkers LSL (START:<seq> sets the Williams face order; each correct PAIR:<face>
# advances the pointer) — the pens no longer SELECT the face, they only POSE it. Each face is a
# single point FACE_POINT_OFFSET_M above the board plane at its 3x3 grid cell; the board is placed
# so THAT point reaches the pen-line intersection.  >>> VERIFY the three physical constants below
# on the actual board (pitch, offset side, and that face 1 lands top-left).  <<<
ORIENT_TILT_DEG          = 0.0     # board tilt off the pen plane. 0 = present the face PARALLEL to
                                   #   the pen plane (no lean). Raise to lean the board for side
                                   #   clearance; sign = lean direction.
ORIENT_MIN_PEN_ANGLE_DEG = 20.0    # below this pen-pair angle the cross-product normal is unreliable
ORIENT_MIN_PEN_SIN       = float(np.sin(np.deg2rad(ORIENT_MIN_PEN_ANGLE_DEG)))
# ORIENT rotation toggle. True = full 6-DOF: the board follows the pen intersection AND rotates
# near-parallel to the pen plane (+ORIENT_TILT_DEG). False = translation-only (keep orientation).
ORIENT_USE_ROTATION      = True
# Cap the board tilt so it can't swing too far: the desired orientation is clamped to +/-this many
# degrees per axis (roll, pitch, yaw) RELATIVE to the level (ORIENT-entry) orientation.
ORIENT_MAX_TILT_DEG      = 45.0
# DOF LOCK (task-space selection / virtual fixture, Raibert & Craig 1981; Abbott & Okamura 2007).
# When a lock is True the board's target for that axis is HELD FIXED at ..._LOCK_*_DEG (relative to
# the level ORIENT-entry orientation) and the servo regulates it there — the board does not rotate
# on that axis while it still tracks position + the free axes. Here PITCH is locked (level).
ORIENT_LOCK_PITCH        = True    # lock the board pitch (forward/back lean)
ORIENT_LOCK_PITCH_DEG    = 0.0     # pitch angle to hold (0 = level; set a constant lean if wanted)
ORIENT_LOCK_ROLL         = False   # lock the board roll too (side-to-side)
ORIENT_LOCK_ROLL_DEG     = 0.0
# (yaw always tracks the pens, clamped to +/-ORIENT_MAX_TILT_DEG.)
# SIDE-APPROACH WEDGE. Keep each button reachable only from its EXTERIOR: spin the board about its
# normal so the TARGET face's outward radial (straight for an edge face, 45 deg for a corner) stays
# within +/-ORIENT_APPROACH_HALF_DEG of the direction the tools come from (pen tips -> intersection).
# It's a TOLERANCE: only the excess beyond the wedge is corrected (no spin while already inside).
ORIENT_APPROACH_WEDGE      = True
ORIENT_APPROACH_HALF_DEG   = 45.0   # half-width of the 90-degree approach wedge
ORIENT_APPROACH_MARGIN_DEG = 5.0    # correct to just inside the wedge edge (avoids edge chatter)
FACE_GRID_PITCH_M        = 0.030   # centre-to-centre spacing of the 3x3 faces (VERIFY on the board)
FACE_POINT_OFFSET_M      = 0.022   # 22 mm above the board plane, along +Z_board (the working face)
# RIGID-GRIP DECOUPLING. Once gripped the object cannot move relative to the EE, so the object->EE
# transform is captured ONCE at ORIENT entry and the servo then runs from FK + that fixed grip — the
# per-frame ArUco/PnP jitter no longer enters the control loop. A watchdog still compares the FK-
# predicted board pose against the live vision pose and warns on a sustained divergence (a bump / real
# slip); the watchdog NEVER perturbs the command.
ORIENT_SLIP_POS       = 0.015   # m:   FK-vs-vision board-centre divergence flagged as possible slip
ORIENT_SLIP_ANG       = 10.0    # deg: FK-vs-vision board-rotation divergence
ORIENT_SLIP_FRAMES    = 10      # sustained frames over threshold before warning
ORIENT_SLIP_RECAPTURE = False   # True -> re-capture the grip on sustained divergence (else warn only)
# Face number (1..8) -> (row, col) IN THE BOARD/CAMERA FRAME (row 0 = +Y top, col 0 = -X left).
# DERIVED from the physical ArUco marker positions (see face_mapping_check.py): corner marker 0 is
# on face 3 (board top-left), marker 1 on face 5 (top-right), marker 2 on face 7 (bottom-right).
# With the faces numbered consecutively (clockwise) around the ring, that uniquely gives:
#     3 4 5
#     2 . 6      (face 1 = bottom-left, then 2=LC,3=TL,4=TC,5=TR,6=RC,7=BR,8=BC)
#     1 8 7
_FACE_CELLS = [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]  # BL,LC,TL,TC,TR,RC,BR,BC
NUM_TO_CELL = {i + 1: rc for i, rc in enumerate(_FACE_CELLS)}

def williams_sequences(n=8):
    """The runner's balanced Latin square (identical to session_runner.williams_sequences), so the
    approach side can reconstruct the face order from just the sequence number in START:<seq>."""
    starter, lo, hi, take_lo = [0], 1, n - 1, True
    while len(starter) < n:
        if take_lo:
            starter.append(lo); lo += 1
        else:
            starter.append(hi); hi -= 1
        take_lo = not take_lo
    return [[(starter[j] + i) % n + 1 for j in range(n)] for i in range(n)]

def face_point_local(face):
    """Selected face's point in the BOARD frame (origin = board centre; X right, Y up, Z out of the
    board — see MARKER_LAYOUT). Returns the board centre (zeros) when the face is unknown, so the
    servo degrades gracefully to the old 'centre -> intersection' behaviour if no START arrived."""
    if face is None:
        return np.zeros(3)
    rc = NUM_TO_CELL.get(int(face))
    if rc is None:
        return np.zeros(3)
    row, col = rc
    x = (col - 1) * FACE_GRID_PITCH_M          # col 0 = left (-X), col 2 = right (+X)
    y = (1 - row) * FACE_GRID_PITCH_M          # row 0 = top  (+Y), row 2 = bottom (-Y)
    return np.array([x, y, FACE_POINT_OFFSET_M], float)

# Pose filter (base frame). alpha in (0, 1]: 1 = no smoothing.
FILTER_ALPHA_R  = 0.5
FILTER_ALPHA_T  = 0.5
MAX_STALE       = 15        # frames held estimate stays valid without detection

# Gripper mounting. EE-X is the approach/pointing axis (Z1 tip offset confirms).
# JAW_AXIS_IS_EEZ: True -> jaws straddle along EE-Z (surface normal side).
# Flip if the gripper arrives rotated 90 deg about the approach axis.
JAW_AXIS_IS_EEZ = True
SCREW_COUPLED   = True      # False -> straight-line world translation (decoupled PBVS)

GRIPPER_OPEN    = -0.5
GRIPPER_CLOSE   = +1.0
GRIPPER_NEUTRAL =  0.0

# Marker layout in object/board frame (metres). Origin at board centre.
# 160 mm (16 cm) square board, 8 markers on the perimeter in a 3x3 grid with the
# centre cell empty: 4 corners + 4 edge-midpoints. Each marker is 20 mm and sits
# 2 mm from the board edge(s) it borders, leaving a 48 mm clear gap between adjacent
# markers.
#   ±68 mm corner offset: half_board(80) - edge_margin(2) - half_marker(10) = 68 mm.
#   Edge-midpoint markers sit halfway between the two corners on their edge.
# Frame: X right, Y up, Z out of the board toward the camera (top-left = (-x, +y)).
# IDs 0-3 keep their previous corner positions; 26-29 are the new edge-midpoints,
# numbered clockwise from top.
MARKER_LAYOUT = {
    0:  np.array([-0.068,  0.068, 0.0]),   # top-left      (corner)
    1:  np.array([ 0.068,  0.068, 0.0]),   # top-right     (corner)
    2:  np.array([ 0.068, -0.068, 0.0]),   # bottom-right  (corner)
    3:  np.array([-0.068, -0.068, 0.0]),   # bottom-left   (corner)
    26: np.array([ 0.000,  0.068, 0.0]),   # top-centre    (edge midpoint)
    27: np.array([ 0.068,  0.000, 0.0]),   # right-centre  (edge midpoint)
    28: np.array([ 0.000, -0.068, 0.0]),   # bottom-centre (edge midpoint)
    29: np.array([-0.068,  0.000, 0.0]),   # left-centre   (edge midpoint)
}

# Per-marker corners in board frame, in TL-TR-BR-BL order matching aruco output.
# board_corner = marker_centre_in_board + marker_local_corner
_h = MARKER_SIZE / 2.0
_MARKER_LOCAL_CORNERS = np.array(
    [[-_h, _h, 0.0], [_h, _h, 0.0], [_h, -_h, 0.0], [-_h, -_h, 0.0]],
    dtype=np.float64)
BOARD_CORNERS = {m: (MARKER_LAYOUT[m] + _MARKER_LOCAL_CORNERS) for m in MARKER_LAYOUT}

# Object footprint estimated from the KNOWN marker positions: the markers sit on the
# object's corners/edges, so the outer extent of all marker corners (board frame) is
# the object boundary. The ORIENT box + touch test use this to reconstruct the FULL
# object box from whatever markers happen to be visible.
_MARK_CORNERS = np.vstack(list(BOARD_CORNERS.values()))
MARK_HALF_X = float(np.max(np.abs(_MARK_CORNERS[:, 0])))
MARK_HALF_Y = float(np.max(np.abs(_MARK_CORNERS[:, 1])))

# ---------------------------------------------------------------------------
# Camera / detector
# ---------------------------------------------------------------------------
try:
    K    = np.load("camera_matrix.npy")
    dist = np.load("dist_coeffs.npy")
except FileNotFoundError as e:
    print(f"[ERROR] missing camera calibration ({e}). Run calibrate_camera.py first.")
    raise SystemExit(1)

DICT     = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params   = aruco.DetectorParameters()
# Corner refinement: sub-pixel corners are the single biggest lever on pose accuracy
# for small markers (Wang & Olson, "AprilTag 2", IROS 2016) — but CORNER_REFINE_APRILTAG
# is also the single biggest per-frame COST, especially at 2560x1440. If the loop is
# slow (see the [fps] print), CORNER_REFINE_SUBPIX is much cheaper and nearly as
# accurate; CORNER_REFINE_NONE is fastest of all (coarser pose). Swap the line below.
# AprilTag refinement = most accurate corners = best APPROACH precision. The orient phase
# no longer detects the board (ORIENT_DETECT_BOARD=False), so this cost is only paid during
# approach/present, where precision matters — keep it on AprilTag.
params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG   # accuracy; slowest
# params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX   # ~as accurate, much faster
# params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE     # fastest, coarsest
detector = aruco.ArucoDetector(DICT, params)

# ---------------------------------------------------------------------------
# SE(3) / SO(3) helpers (Lynch & Park, Modern Robotics, 2017)
# ---------------------------------------------------------------------------
def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).flatten(); return T

def R_from_rvec(rv):
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1)); return R

def inv_T(T):
    """Analytical inverse of a homogeneous transform."""
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t; return Ti

def skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])

def orthonormalize(R):
    """Project a near-rotation onto SO(3) via SVD (chordal L2 projection,
    Moakher 2002), forcing det = +1."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1; Rn = U @ Vt
    return Rn

def so3_log(R):
    """Matrix log of SO(3) -> rotation vector (axis * angle)."""
    rv, _ = cv2.Rodrigues(orthonormalize(R)); return rv.flatten()

def so3_exp(w):
    """Matrix exp of so(3): rotation vector -> rotation matrix."""
    R, _ = cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1)); return R

def rpy_to_R(roll, pitch, yaw):
    """ZYX-Euler (roll about X, pitch about Y, yaw about Z) -> rotation matrix.
    Same convention as TransformManager.pose_to_matrix (R = Rz @ Ry @ Rx),
    so it reconstructs the EE rotation that produced the endPosture triple."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def R_to_rpy(R):
    """Inverse of rpy_to_R: rotation matrix -> (roll, pitch, yaw) in the same ZYX convention
    (R = Rz @ Ry @ Rx). Gimbal-lock safe near pitch = +/-90 deg."""
    R = orthonormalize(R)
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock -> fold into yaw, roll = 0
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])
    return float(roll), float(pitch), float(yaw)

def se3_log(T):
    """Matrix log of SE(3): body twist (omega_b, v_b) with expm([V]) = T.
    Closed-form left-Jacobian inverse (Modern Robotics eq. 3.92, Lynch & Park 2017)."""
    R = T[:3, :3]; p = T[:3, 3]
    omega = so3_log(R); theta = np.linalg.norm(omega)
    if theta < 1e-9:
        return np.zeros(3), p.copy()
    W = skew(omega)
    Vinv = (np.eye(3)
            - 0.5 * W
            + (1.0 / theta**2) * (1.0 - (theta / 2.0) / np.tan(theta / 2.0)) * (W @ W))
    return omega, Vinv @ p

# ---------------------------------------------------------------------------
# Pen-tilt classification  (pens_mean_tilt logic, ported in)
# ---------------------------------------------------------------------------
# Discrete tilt prototypes: the plane-normal each ORIENT setpoint produces,
# built with the SAME rpy_to_R (ZYX) used everywhere else in this file so the
# classifier and the commanded setpoint stay consistent. The mean pen direction
# v is compared as an UNDIRECTED line (abs dot) against these normals; the
# closest — if unambiguous and outside the level cone — is the commanded tilt.
_TILT_Z = np.array([0.0, 0.0, 1.0])
def _tilt_normal(roll_deg, pitch_deg):
    return rpy_to_R(np.deg2rad(roll_deg), np.deg2rad(pitch_deg), 0.0) @ _TILT_Z

PROTOTYPES = {
    "PITCH +30": _tilt_normal(0.0,  ORIENT_SWEEP_DEG),
    "PITCH -30": _tilt_normal(0.0, -ORIENT_SWEEP_DEG),
    "ROLL +30":  _tilt_normal( ORIENT_SWEEP_DEG, 0.0),
    "ROLL -30":  _tilt_normal(-ORIENT_SWEEP_DEG, 0.0),
}
LEVEL_NORMAL = _TILT_Z.copy()

NEUTRAL_CONE_DEG = 12.0     # v within this of the level normal -> NEUTRAL (no tilt)
AMBIG_DEG        = 5.0      # top-two prototypes closer than this -> ambiguous
# Loosened from the pen script's 8 / 2.5° / 5 mm so a commit happens quickly even
# with imperfect recognition. Fewer frames fill the window faster; the wider
# wobble/spread tolerances stop small hand tremor from reading as "moving".
STAB_WIN         = 4        # frames in the pen-steadiness window (was 8)
STAB_ANG_DEG     = 4.0      # max angular wobble across the window, deg (was 2.5)
STAB_POS_M       = 0.010    # max midpoint spread across the window, m (was 0.005)

def _ang(u, w):
    """Undirected angle between two vectors, degrees (treats ±u as the same line)."""
    c = abs(float(np.dot(u, w))) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def classify_tilt(v):
    """EXACT pens_mean_tilt_v2 classifier. Nearest discrete tilt for the mean pen
    direction v (unit, EE frame). Returns (label, detail); label is a PROTOTYPES
    key, or 'NEUTRAL'."""
    if _ang(v, LEVEL_NORMAL) < NEUTRAL_CONE_DEG:
        return "NEUTRAL", "v ~ base normal (level plane already perpendicular)"
    scored = sorted(((lbl, _ang(v, n)) for lbl, n in PROTOTYPES.items()),
                    key=lambda kv: kv[1])
    (best, a0), (second, a1) = scored[0], scored[1]
    if a1 - a0 < AMBIG_DEG:
        return "NEUTRAL", f"ambiguous ({best} {a0:.1f} vs {second} {a1:.1f} deg)"
    return best, f"plane-normal off v by {a0:.1f} deg"

# ---------------------------------------------------------------------------
# Pen-line geometry for the continuous ORIENT servo
# ---------------------------------------------------------------------------
def closest_point_two_lines(p1, d1, p2, d2, eps=1e-9):
    """Closest point of two 3-D lines { p_i + s*d_i }. Two lines almost never meet
    exactly, so this returns the midpoint of the common-perpendicular feet as the
    "intersection" estimate, plus the residual gap between the lines.
    Returns (midpoint, gap, foot1, foot2, parallel)."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d1 = np.asarray(d1, float); d2 = np.asarray(d2, float)
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    r = p1 - p2
    b = float(d1 @ d2); d = float(d1 @ r); e = float(d2 @ r)
    denom = 1.0 - b * b                      # a=c=1 for unit directions
    if abs(denom) < eps:                     # (near-)parallel -> anchor on line 1's tip
        s = 0.0; t = e; parallel = True
    else:
        s = (b * e - d) / denom
        t = (e - b * d) / denom
        parallel = False
    foot1 = p1 + s * d1
    foot2 = p2 + t * d2
    mid = 0.5 * (foot1 + foot2)
    gap = float(np.linalg.norm(foot1 - foot2))
    return mid, gap, foot1, foot2, parallel

def mean_direction(a1, a2):
    """Bisector direction of two axes, sign-canonicalised so ±a map to one line.
    This is the normal of the 'perpendicular to the pens' plane."""
    a1 = np.asarray(a1, float); a2 = np.asarray(a2, float)
    if np.dot(a1, a2) < 0.0:
        a2 = -a2
    v = a1 + a2
    n = np.linalg.norm(v)
    if n < 1e-9:
        return a1 / (np.linalg.norm(a1) + 1e-12)
    return v / n

def weighted_mean_direction(a1, a2, w1, w2):
    """Reprojection-error-weighted bisector of two axes (down-weights the worse-seen pen).
    Sign-canonicalised so ±a map to one line; falls back to a1 if degenerate."""
    a1 = np.asarray(a1, float); a2 = np.asarray(a2, float)
    if np.dot(a1, a2) < 0.0:
        a2 = -a2
    v = float(w1) * a1 + float(w2) * a2
    n = np.linalg.norm(v)
    if n < 1e-9:
        return a1 / (np.linalg.norm(a1) + 1e-12)
    return v / n

def rot_between(a, b):
    """Shortest-arc rotation matrix R with R @ a_hat == b_hat (both 3-vectors).
    Used to rotate the board minimally so its normal lands on the plane normal,
    carrying the in-plane axes along (no gratuitous spin about the normal)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b)); s = float(np.linalg.norm(v))
    if s < 1e-9:                             # parallel or antiparallel
        if c > 0.0:
            return np.eye(3)
        perp = (np.array([0.0, 1.0, 0.0]) if abs(a[0]) > 0.9
                else np.array([1.0, 0.0, 0.0]))
        axis = np.cross(a, perp); axis /= (np.linalg.norm(axis) + 1e-12)
        return so3_exp(np.pi * axis)         # 180° about a perpendicular axis
    return so3_exp(np.arctan2(s, c) * (v / s))

# --- pen drawing helpers, using the pen tracker's OWN calibration (dt.K/dt.dist),
#     bodies copied verbatim from pens_mean_tilt_v2.py -------------------------
def _pen_project(points_cam):
    pts = np.asarray(points_cam, float).reshape(-1, 1, 3)
    px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), dt.K, dt.dist)
    return px.reshape(-1, 2)

def _ipt(a):
    """Native-int (x, y) pixel tuple for OpenCV drawing. OpenCV 4.13 rejects numpy
    integer scalars from tuple(np_array) with 'wrong type', so cast to Python int."""
    return (int(a[0]), int(a[1]))

def _pen_draw_vector(img, p0, d, length, color, label):
    pts = np.asarray([p0, p0 + length * d], float)
    if np.any(pts[:, 2] <= 1e-3):
        return
    seg = _pen_project(pts)
    if not np.all(np.isfinite(seg)):          # skip degenerate projections
        return
    h, w = img.shape[:2]
    if np.any(np.abs(seg[:, 0]) > 5 * w) or np.any(np.abs(seg[:, 1]) > 5 * h):
        return
    p1, p2 = _ipt(seg[0]), _ipt(seg[1])
    draw_color = tuple(int(c) for c in color)
    cv2.arrowedLine(img, p1, p2, draw_color, 3, tipLength=0.18)
    cv2.putText(img, label, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

def _pen_draw_perp_plane(img, c, v, half, color=(80, 200, 255)):
    if c[2] <= 0:
        return
    u1 = np.cross(v, _TILT_Z)
    if np.linalg.norm(u1) < 1e-6:
        u1 = np.cross(v, np.array([1.0, 0.0, 0.0]))
    u1 /= (np.linalg.norm(u1) + 1e-12)
    u2 = np.cross(v, u1); u2 /= (np.linalg.norm(u2) + 1e-12)
    corners = np.array([c - half * u1 - half * u2, c + half * u1 - half * u2,
                        c + half * u1 + half * u2, c - half * u1 + half * u2])
    cpx = _pen_project(corners).astype(np.int32)
    ov = img.copy(); cv2.fillConvexPoly(ov, cpx, color)
    cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
    cv2.polylines(img, [cpx], True, color, 2)

# ---------------------------------------------------------------------------
# Finger-touch handover release  (Google MediaPipe Hands)
# ---------------------------------------------------------------------------
# End-state condition: while ORIENTing, detect the human hand with MediaPipe and
# ALWAYS draw it (so you can see whether/where it is seen). Recover each hand's true
# 3D pose in the camera frame (see the 3D FINGER-TOUCH block below) and, if >= 2 of a
# single hand's fingertips are actually TOUCHING the board for HAND_CONFIRM frames,
# release the grasp (open the gripper). Two independent pieces:
#   _detect_hands  — runs MediaPipe every call, draws all 21 landmarks + skeleton,
#                    returns each hand's landmark pixel coords AND metric world
#                    landmarks. Independent of the board, so the overlay shows hands
#                    even with no markers in view.
#   _board_square  — convex hull of ALL detected board-marker corners = the object
#                    square; held for a few frames through occlusion (drawn for
#                    reference only; the release test is now the 3D touch check).
FINGER_TIP_IDS      = (4, 8, 12, 16, 20)   # thumb, index, middle, ring, pinky tips
HAND_CONFIRM        = 3      # consecutive frames touching before releasing
HAND_HULL_MAX_STALE = 8      # frames the last square is reused when markers occluded

# 3D FINGER-TOUCH RELEASE (replaces the old open-palm gesture). Ported verbatim from
# finger_touch_check.py. A monocular camera cannot recover a hand's depth from 2D
# landmarks alone, so we build a metric hand model from MediaPipe's world landmarks
# but set EACH finger's true size from the operator's measured base-knuckle->fingertip
# length (asked once at startup), then solvePnP that model against the 2D landmarks to
# recover the hand's full 6-DOF pose in the CAMERA frame — hence every fingertip's true
# 3D position. A fingertip counts as TOUCHING the board when it is within
# TOUCH_PLANE_TOL of the marker plane AND inside the board footprint. Release fires
# when any SINGLE hand has >= TOUCH_FINGERS_MIN of its 5 fingertips touching for
# HAND_CONFIRM consecutive frames (still gated by the level + no-pen release guards).
TOUCH_PLANE_TOL        = 0.015   # 15 mm: max fingertip distance to the board plane
TOUCH_FOOTPRINT_MARGIN = 0.020   # 20 mm slack past the board edge for a valid touch
TOUCH_FINGERS_MIN      = 2       # fingertips (of 5) in ONE hand required to release

# PER-FINGER hand model. Each finger's metric size is set from an operator measurement
# of that finger's BASE KNUCKLE (MCP) -> FINGERTIP length (MediaPipe landmarks are
# skeletal joints; the knuckle apex sits over the MCP joint). These are the MediaPipe
# bone chains, base knuckle first, fingertip last.
FINGER_BONES = {
    "thumb":  (2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
# Operator per-finger lengths, base knuckle -> fingertip (m). Overwritten at startup.
HAND_FINGER_M = {"thumb": 0.065, "index": 0.080, "middle": 0.087,
                 "ring": 0.080, "pinky": 0.067}
TIP_NAMES = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

# Per-finger distance smoothing.
DIST_EMA_ALPHA         = 0.35    # per-finger distance smoothing (0=frozen, 1=raw)
EMA_MAX_STALE          = 8       # dropped-pose frames before a hand's smoothing resets

# Hand-pose stabilisation / gating (per hand). Monocular PnP over MediaPipe landmarks
# is poorly conditioned in DEPTH and occasionally flips to a wrong-depth branch. We
# seed each solve from that hand's previous accepted pose and REJECT solves with gross
# reprojection error, implausible depth, or a large depth jump (a flip still fits the
# 2D, so it is caught by the depth gates, not by reprojection).
HAND_REPROJ_MAX_PX = 60.0   # loose sanity gate (RMS px at 2K); flips caught by depth
HAND_Z_MIN         = 0.05   # plausible hand distance from camera (m)
HAND_Z_MAX         = 1.50
HAND_MAX_DZ        = 0.12   # max between-frame depth jump while locked (m)
HAND_RELOCK_AFTER  = 6      # consecutive rejects before dropping the lock to re-acquire

# Object-pose tracking. Once grasped, the object is rigidly held by the gripper, and
# the camera is on the same arm (eye-in-hand) — so the object's full pose in the
# CAMERA frame is ~CONSTANT afterwards. We seed it at grasp, then REFRESH it every
# frame from whatever board markers remain visible (the 8-marker board keeps >=2 in
# view even while held), and hold the last pose only under full occlusion.
# BOARD_HALF: half-side of the drawn box in the board plane (m); BOARD_BOX_THICK: its
# extrusion along the board normal (m). The 160 mm board has half-side 0.080, so
# 0.080 outlines the physical board edge (markers sit at ±0.068).
BOARD_HALF     = 0.080
BOARD_BOX_THICK = 0.030
# Live refresh of the tracked object pose: needs >= OBJ_LIVE_MIN_MARKERS visible
# board markers; OBJ_LIVE_ALPHA is the per-frame geodesic blend toward the live
# pose (0 = frozen, 1 = raw live, no smoothing). This corrects grasp-estimate error
# and any slip in the gripper, and de-jitters the box.
OBJ_LIVE_MIN_MARKERS = 2
OBJ_LIVE_ALPHA       = 0.3
OBJ_LIVE_MAX_JUMP    = np.deg2rad(30.0)   # reject a live pose that jumps more than this
                                          # vs the tracked pose (planar two-fold flip) —
                                          # a flip is what scatters the drawn box
# ORIENT box pose smoothing. Eye-in-hand keeps the object ~static in the camera frame,
# so the ORIENT box pose is heavily low-passed toward the live PnP (small alpha = very
# stable, and lag is invisible because the object isn't moving). This is what stops the
# box from shifting when a marker drops out; flips are still rejected via OBJ_LIVE_MAX_JUMP.
ORIENT_BOX_ALPHA     = 0.15

# Standard 21-landmark hand topology, so we can draw the skeleton ourselves and
# stay backend-agnostic (the Tasks API has no drawing_utils helper).
HAND_CONNECTIONS_STD = (
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # index
    (5, 9), (9, 10), (10, 11), (11, 12),         # middle
    (9, 13), (13, 14), (14, 15), (15, 16),       # ring
    (13, 17), (17, 18), (18, 19), (19, 20),      # pinky
    (0, 17),                                     # palm base
)
# Tasks-API model asset (the same hand_landmarker.task used by test2.py). Loaded
# from next to the script; auto-downloaded once if absent. Official Google-hosted.
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/1/hand_landmarker.task")

# MediaPipe Hands instance (created once at startup, if available).
_hands           = None
_hand_last_hull  = None      # last board-corner convex hull, (N,1,2) int32
_hand_hull_stale = 0
_hand_ts         = 0         # monotonic timestamp (ms) for Tasks VIDEO mode

def _board_square(ids, corners_list):
    """Convex hull (int32, shape (N,1,2)) of ALL detected board-marker corners —
    the square made by the 4 ArUco markers. Uses whatever markers are visible and
    holds the last hull for a few frames through occlusion. Returns hull or None."""
    global _hand_last_hull, _hand_hull_stale
    pts = []
    if ids is not None:
        for mid, c in zip(ids.flatten(), corners_list):
            if int(mid) in MARKER_LAYOUT:
                pts.append(np.asarray(c).reshape(-1, 2))
    if pts:
        hull = cv2.convexHull(np.vstack(pts).astype(np.int32))
        _hand_last_hull = hull; _hand_hull_stale = 0
        return hull
    _hand_hull_stale += 1
    if _hand_last_hull is not None and _hand_hull_stale <= HAND_HULL_MAX_STALE:
        return _hand_last_hull
    return None

def _download_file(url, path):
    """Download url -> path, robust to macOS' missing CA certs. Tries proper
    verification via certifi, then falls back to an UNVERIFIED SSL context (this
    is a one-time fetch of a public Google-hosted model asset)."""
    import urllib.request, ssl, shutil
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    # 1) verified via certifi if available
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
        return
    except Exception as e:                           # noqa: BLE001
        print(f"[MediaPipe] verified download failed ({type(e).__name__}); "
              f"retrying without SSL verification ...")
    # 2) fallback: unverified context (common macOS python-without-certs case)
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)

def _draw_hand_px(frame, pts):
    """Draw the hand skeleton (connections + landmark dots) from pixel points."""
    for a, b in HAND_CONNECTIONS_STD:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)

def _detect_hands(frame, draw_frame=None):
    """Detect hands (VIDEO mode, full frame). Draws each hand on draw_frame and
    returns a list of per-hand dicts: {"px": [(x,y)*21], "world": (21,3) or None,
    "label": str}. The label is MediaPipe's handedness ("Left"/"Right"), used as the
    per-hand key for pose continuity, smoothing and calibration; falls back to
    hand{index}. `frame` should be the pristine camera image so MediaPipe never sees
    the ArUco overlays; skeletons are drawn on draw_frame."""
    global _hand_ts
    if not MP_AVAILABLE or _hands is None:
        return []
    if draw_frame is None:
        draw_frame = frame
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _hand_ts += 1
    res = _hands.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), _hand_ts)
    world_all = res.hand_world_landmarks if res.hand_world_landmarks else []
    handed    = res.handedness if res.handedness else []
    hands = []
    seen_labels = {}
    for i, hlm in enumerate(res.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hlm]
        world = None
        if i < len(world_all):
            world = np.array([[p.x, p.y, p.z] for p in world_all[i]],
                             dtype=np.float64)
        label = None
        if i < len(handed) and handed[i]:
            label = handed[i][0].category_name          # "Left" / "Right"
        if not label:
            label = f"hand{i}"
        if label in seen_labels:                        # guard duplicate labels
            label = f"{label}{i}"
        seen_labels[label] = True
        hands.append({"px": pts, "world": world, "label": label})
    for hnd in hands:
        _draw_hand_px(draw_frame, hnd["px"])
    return hands

# --- 3D hand pose (per-finger operator model, per-hand PnP continuity) -----------
# Per-hand pose-continuity state, keyed by hand label, so two hands are tracked and
# gated independently.
_hand_prev_rvec = {}        # key -> rvec
_hand_prev_tvec = {}        # key -> tvec
_hand_reject    = {}        # key -> consecutive-reject count
_hand_fail      = {}        # key -> reason the last solve was rejected

def _hand_relock(key=None):
    """Drop the pose lock for one hand key (or all keys if key is None)."""
    if key is None:
        _hand_prev_rvec.clear(); _hand_prev_tvec.clear(); _hand_reject.clear()
    else:
        _hand_prev_rvec.pop(key, None); _hand_prev_tvec.pop(key, None)
        _hand_reject.pop(key, None)

def _hand_tip_positions_cam(world_lm, px_lm, key):
    """Recover the 3D CAMERA-frame positions of a hand's 21 landmarks, with per-hand
    temporal continuity + outlier gating. `key` is the hand's handedness label.

    Builds a metric hand model from MediaPipe's world landmarks, sizing each finger
    from the operator's measured base-knuckle->fingertip length (HAND_FINGER_M). The
    pose is solved by solvePnP, SEEDED from this hand's previous accepted frame so it
    stays on the same depth branch, then gated on reprojection error / plausible depth
    / depth continuity. Returns (pts_cam (21,3) m, reproj_px) on an accepted frame, or
    (None, None) if the frame was rejected as unstable."""
    _hand_fail[key] = ""
    prev_rvec = _hand_prev_rvec.get(key)
    prev_tvec = _hand_prev_tvec.get(key)
    if world_lm is None:
        _hand_fail[key] = "no world landmarks"
        return None, None
    W = np.asarray(world_lm, dtype=np.float64).reshape(-1, 3)
    P = np.asarray(px_lm, dtype=np.float64).reshape(-1, 2)
    if W.shape[0] < 21 or P.shape[0] != W.shape[0]:
        _hand_fail[key] = "bad landmark count"
        return None, None
    # Per-finger scale: measured length / model bone-length (sum of segment norms,
    # pose-invariant since bone lengths don't change as the finger bends).
    r = {}
    for f, chain in FINGER_BONES.items():
        L_model = sum(np.linalg.norm(W[chain[k + 1]] - W[chain[k]])
                      for k in range(len(chain) - 1))
        if L_model < 1e-6:
            return None, None
        r[f] = HAND_FINGER_M[f] / L_model
    s0 = float(np.mean(list(r.values())))      # shared scale for palm / knuckles
    obj = (W * s0).copy()                       # globally scaled model
    # Re-stretch each finger about its base knuckle so its bones match the operator.
    for f, chain in FINGER_BONES.items():
        base = chain[0]
        extra = r[f] / s0
        for k in chain[1:]:
            obj[k] = obj[base] + extra * (obj[k] - obj[base])
    obj = obj.astype(np.float64)
    img = P.astype(np.float64)

    def _reject(reason):
        _hand_fail[key] = reason
        _hand_reject[key] = _hand_reject.get(key, 0) + 1
        if _hand_reject[key] > HAND_RELOCK_AFTER:
            _hand_relock(key)
        return None, None

    # Solve. Seed from this hand's previous accepted pose (keeps the same depth
    # branch); fall back to a from-scratch SQPNP if none or the seeded solve fails.
    ok = False; rvec = tvec = None
    if prev_rvec is not None:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, dist, rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok = False
    if not ok:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            ok = False
    if not ok:
        return _reject("solvePnP failed")

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    reproj = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - img) ** 2, axis=1))))
    z = float(tvec.reshape(3)[2])
    dz = abs(z - float(prev_tvec.reshape(3)[2])) if prev_tvec is not None else 0.0
    if reproj > HAND_REPROJ_MAX_PX:
        return _reject(f"reproj {reproj:.0f}px>{HAND_REPROJ_MAX_PX:.0f}")
    if not (HAND_Z_MIN <= z <= HAND_Z_MAX):
        return _reject(f"depth {z*1000:.0f}mm out of range")
    if dz > HAND_MAX_DZ:
        return _reject(f"depth jump {dz*1000:.0f}mm>{HAND_MAX_DZ*1000:.0f} (flip)")

    _hand_prev_rvec[key] = np.asarray(rvec, dtype=np.float64).copy()
    _hand_prev_tvec[key] = np.asarray(tvec, dtype=np.float64).copy()
    _hand_reject[key] = 0
    R = R_from_rvec(rvec)
    return (R @ obj.T).T + tvec.reshape(3), reproj

def _board_pose_cam(R_obj_cam, t_obj_cam):
    """Best available board pose in the CAMERA frame as (R_co, t_co). Prefer the live
    board PnP; fall back to the grasp-frozen/tracked obj_pose_cam. Returns None if
    neither is available. (The board-frame origin lies in the marker plane, z=0.)"""
    if R_obj_cam is not None and t_obj_cam is not None:
        return R_obj_cam, np.asarray(t_obj_cam, dtype=np.float64).reshape(3)
    if obj_pose_cam is not None:
        return obj_pose_cam[:3, :3], obj_pose_cam[:3, 3].copy()
    return None

# Stable ORIENT board pose (camera frame). Seeded at ORIENT entry, then heavily smoothed
# toward the live PnP each frame, holding on dropout and rejecting flips.
_orient_box_R = None
_orient_box_t = None

def _orient_update_box(R_live, t_live):
    """Update the smoothed ORIENT board pose from the live PnP (markers seen this frame).
    board_pose recovers a FULL pose from any visible subset via the known layout, so the
    pose is available down to a single marker; the heavy low-pass + flip-reject + hold
    mean that losing a marker doesn't move the box. Eye-in-hand keeps the object static
    in the camera frame, so the smoothing adds no visible lag."""
    global _orient_box_R, _orient_box_t
    if R_live is None or t_live is None:
        return                                             # no markers -> hold last pose
    t_live = np.asarray(t_live, dtype=np.float64).reshape(3)
    R_live = orthonormalize(R_live)
    if _orient_box_R is None:
        _orient_box_R, _orient_box_t = R_live, t_live.copy()
        return
    dR = float(np.linalg.norm(so3_log(_orient_box_R.T @ R_live)))
    if dR > OBJ_LIVE_MAX_JUMP:                             # flip / gross outlier -> hold
        return
    a = ORIENT_BOX_ALPHA
    _orient_box_R = orthonormalize(_orient_box_R @ so3_exp(a * so3_log(_orient_box_R.T @ R_live)))
    _orient_box_t = (1.0 - a) * _orient_box_t + a * t_live

def _fingertips_touching(pts_cam, R_co, t_co):
    """Assess each of a hand's 5 fingertips against the board. Markers sit at z=0 in
    the board frame, so distance to the plane is the fingertip's board-frame
    z-coordinate (SIGNED: +ve in front of the marked face, -ve behind). A fingertip is
    TOUCHING when |dist| <= TOUCH_PLANE_TOL AND it is inside the board footprint.
    Returns (count, info) where info[tid] = {"touch": bool, "dist_mm": signed float,
    "in_foot": bool}."""
    info = {}; n = 0
    for tid in FINGER_TIP_IDS:
        p_obj = R_co.T @ (pts_cam[tid] - t_co)          # fingertip in board frame
        # Boundary from the marker-estimated object extent (±MARK_HALF_X/Y), plus slack.
        in_foot = (abs(p_obj[0]) <= MARK_HALF_X + TOUCH_FOOTPRINT_MARGIN and
                   abs(p_obj[1]) <= MARK_HALF_Y + TOUCH_FOOTPRINT_MARGIN)
        dist_m = float(p_obj[2])
        touching = bool(abs(dist_m) <= TOUCH_PLANE_TOL and in_foot)
        info[tid] = {"touch": touching, "dist_mm": dist_m * 1000.0, "in_foot": in_foot}
        n += int(touching)
    return n, info

def _project_cam_point(p_cam):
    """Project a point already in the CAMERA frame to a pixel. Returns
    (px, depth) or (None, None) if the point is behind the camera."""
    p_cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
    if p_cam[2] <= 1e-6:
        return None, None
    uv, _ = cv2.projectPoints(p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)
    return (int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))), float(p_cam[2])

def _board_box_pts():
    """8 corners of a box around the markers, in the OBJECT frame (metres): a square
    of half-side BOARD_HALF in the board plane, extruded ±BOARD_BOX_THICK/2 along the
    board normal (z). Order: front face (z>0) 0-3, back face (z<0) 4-7."""
    h = BOARD_HALF; z = BOARD_BOX_THICK / 2.0
    return np.array([(-h, -h,  z), ( h, -h,  z), ( h,  h,  z), (-h,  h,  z),
                     (-h, -h, -z), ( h, -h, -z), ( h,  h, -z), (-h,  h, -z)],
                    dtype=np.float64)

def _draw_board_box(img, T_cam_obj, color=(0, 200, 255)):
    """Project and draw a 3D wireframe box around the markers from the object's
    camera-frame pose T_cam_obj. Returns (front_face_quad int32 (4,1,2), centre_px)
    for the release region test, or (None, None) if the box is behind the camera."""
    R = T_cam_obj[:3, :3]; t = T_cam_obj[:3, 3]
    if t[2] <= 1e-6:
        return None, None
    box_obj = _board_box_pts()
    # Any corner at/behind the camera projects to a wild pixel and draws lines across
    # the frame — skip the whole box in that case (all-or-nothing keeps it clean).
    pts_cam = (R @ box_obj.T).T + t
    if np.any(pts_cam[:, 2] <= 1e-3):
        return None, None
    rvec = cv2.Rodrigues(R)[0]; tvec = t.reshape(3, 1)
    uv, _ = cv2.projectPoints(box_obj, rvec, tvec, K, dist)
    uv = uv.reshape(-1, 2)
    if not np.all(np.isfinite(uv)):                     # degenerate projection -> skip
        return None, None
    h_img, w_img = img.shape[:2]
    if np.max(np.abs(uv)) > 10 * max(h_img, w_img):     # absurd (grazing) projection -> skip
        return None, None
    uv = uv.astype(int)
    front, back = uv[:4], uv[4:]
    dim = (color[0] // 2, color[1] // 2, color[2] // 2)
    loop = ((0, 1), (1, 2), (2, 3), (3, 0))
    for a, b in loop:                                   # front face (bright)
        cv2.line(img, _ipt(front[a]), _ipt(front[b]), color, 2)
    for a, b in loop:                                   # back face (dim)
        cv2.line(img, _ipt(back[a]), _ipt(back[b]), dim, 1)
    for i in range(4):                                  # connecting edges
        cv2.line(img, _ipt(front[i]), _ipt(back[i]), color, 1)
    cuv, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, dist)
    centre = (int(cuv[0, 0, 0]), int(cuv[0, 0, 1]))
    put(img, "OBJ", (centre[0] + 8, centre[1] - 8), color, 0.5)
    return front.reshape(-1, 1, 2).astype(np.int32), centre

def _draw_box_from_pose(img, R_co, t_co, hx, hy, hz, color=(0, 200, 255), label="OBJ"):
    """Draw a 3D box of half-extents (hx, hy) in the board plane and ±hz along the board
    normal, placed at a given CAMERA-frame board pose (R_co, t_co). Same projection
    guards as _draw_board_box. Lets ORIENT draw the object box directly from a live
    board pose + a marker-derived extent, without touching the frozen-pose path used by
    the other states. Returns (front_face_quad int32 (4,1,2), centre_px) or (None, None)."""
    t = np.asarray(t_co, dtype=np.float64).reshape(3)
    if t[2] <= 1e-6:
        return None, None
    corners = np.array([(-hx, -hy,  hz), ( hx, -hy,  hz), ( hx,  hy,  hz), (-hx,  hy,  hz),
                        (-hx, -hy, -hz), ( hx, -hy, -hz), ( hx,  hy, -hz), (-hx,  hy, -hz)],
                       dtype=np.float64)
    pts_cam = (R_co @ corners.T).T + t
    if np.any(pts_cam[:, 2] <= 1e-3):                   # any corner behind camera -> skip
        return None, None
    rvec = cv2.Rodrigues(R_co)[0]; tvec = t.reshape(3, 1)
    uv, _ = cv2.projectPoints(corners, rvec, tvec, K, dist); uv = uv.reshape(-1, 2)
    if not np.all(np.isfinite(uv)):
        return None, None
    h_img, w_img = img.shape[:2]
    if np.max(np.abs(uv)) > 10 * max(h_img, w_img):     # grazing blow-up -> skip
        return None, None
    uv = uv.astype(int); front, back = uv[:4], uv[4:]
    dim = (color[0] // 2, color[1] // 2, color[2] // 2)
    loop = ((0, 1), (1, 2), (2, 3), (3, 0))
    for a, b in loop:
        cv2.line(img, _ipt(front[a]), _ipt(front[b]), color, 2)
    for a, b in loop:
        cv2.line(img, _ipt(back[a]), _ipt(back[b]), dim, 1)
    for i in range(4):
        cv2.line(img, _ipt(front[i]), _ipt(back[i]), color, 1)
    cuv, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, dist)
    centre = (int(cuv[0, 0, 0]), int(cuv[0, 0, 1]))
    put(img, label, (centre[0] + 8, centre[1] - 8), color, 0.5)
    return front.reshape(-1, 1, 2).astype(np.int32), centre

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def project_pt(rv, tv):
    """Project a pose's origin into image coordinates.
    Since the input point is [0,0,0], only tv (camera-frame position) matters."""
    pt, _ = cv2.projectPoints(np.zeros((1, 3), np.float32), rv, tv, K, dist)
    return tuple(int(x) for x in pt[0, 0])

def put(img, text, pt, color=(0, 220, 80), scale=0.48):
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

# ---------------------------------------------------------------------------
# Single rigid-body board PnP  (from 1706, replacing per-marker PnP + fusion)
# ---------------------------------------------------------------------------
# Gates on the board PnP, using the reprojection error the solver already returns.
# REPROJ_MAX_PX drops grossly bad fits; AMBIG_REPROJ_RATIO refuses to PRIME on a
# near-tie between the two planar IPPE candidates (the fronto-parallel two-fold
# flip), which would otherwise lock in a wrong orientation. Both thresholds are
# resolution-dependent — tune against observed reprojection RMS.
REPROJ_MAX_PX      = 8.0
AMBIG_REPROJ_RATIO = 1.30

_prev_board_R = None   # last accepted board rotation in camera frame

def board_pose(ids, corners_list):
    """Solve ONE board pose from all visible known markers (1706 method).

    Stacks every visible marker's 4 corners into a single solvePnPGeneric
    call with SOLVEPNP_IPPE. The 140 mm board baseline drives rotation
    variance far below what per-marker PnP achieves.

    Disambiguation: once a prior exists, pick the candidate closest in SO(3)
    to the previous frame (temporal continuity); otherwise pick the lowest-
    reprojection-error forward-facing solution (Collins & Bartoli 2014).

    Returns (R_obj_cam, t_obj_cam, reproj_rms) or (None, None, None).
    """
    global _prev_board_R
    if ids is None:
        return None, None, None

    obj_pts, img_pts = [], []
    for mid, c in zip(ids.flatten(), corners_list):
        mid = int(mid)
        if mid in BOARD_CORNERS:
            obj_pts.append(BOARD_CORNERS[mid])
            img_pts.append(c.reshape(4, 2))
    if not obj_pts:
        return None, None, None

    obj = np.vstack(obj_pts).astype(np.float32)
    img = np.vstack(img_pts).astype(np.float32)

    n_sol, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
        obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if n_sol < 1:
        return None, None, None

    cands = []
    for rv, tv, err in zip(rvecs, tvecs, reproj):
        R = R_from_rvec(rv); t = tv.flatten()
        faces = np.dot(R[:, 2], t) < 0.0   # board normal points toward camera
        cands.append((R, t, float(np.ravel(err)[0]), faces))

    if _prev_board_R is not None:
        R, t, err, _ = min(
            cands, key=lambda c: np.linalg.norm(so3_log(_prev_board_R.T @ c[0])))
    else:
        facing = [c for c in cands if c[3]]
        pool = sorted(facing if facing else cands, key=lambda c: c[2])
        # Refuse to PRIME on an ambiguous planar solve: if the two lowest-error
        # candidates are a near-tie the two-fold flip is unresolved, so wait for a
        # clearer (more oblique) view rather than lock onto a possibly-flipped pose.
        if len(pool) >= 2 and pool[1][2] <= AMBIG_REPROJ_RATIO * max(pool[0][2], 1e-6):
            return None, None, None
        R, t, err, _ = pool[0]

    # Drop grossly bad fits regardless of how the candidate was chosen.
    if err > REPROJ_MAX_PX:
        return None, None, None

    _prev_board_R = R
    return R, t, err

# ---------------------------------------------------------------------------
# On-manifold pose filter in the BASE frame  (from 1706)
# ---------------------------------------------------------------------------
class PoseFilter:
    """Exponential low-pass for a static rigid body expressed in the BASE frame.

    Rotation blended along the SO(3) geodesic (Sola et al. 2018; Moakher 2002);
    translation blended linearly. Holds the estimate for up to max_stale frames
    when detection drops out, keeping the servo target smooth through occlusion.
    """
    def __init__(self, alpha_R, alpha_t, max_stale):
        self.alpha_R = alpha_R; self.alpha_t = alpha_t
        self.max_stale = max_stale
        self.R = None; self.t = None; self.stale = 0

    def update(self, R, t):
        if self.R is None:
            self.R = orthonormalize(R)
            self.t = np.asarray(t, dtype=np.float64).copy()
        else:
            dr = so3_log(self.R.T @ R)                      # geodesic increment
            self.R = orthonormalize(self.R @ so3_exp(self.alpha_R * dr))
            self.t = (1.0 - self.alpha_t) * self.t + self.alpha_t * np.asarray(t)
        self.stale = 0

    def hold(self):
        if self.R is not None: self.stale += 1

    def valid(self):
        return self.R is not None and self.stale <= self.max_stale

    def reset(self):
        self.R = None; self.t = None; self.stale = 0

pose_filter = PoseFilter(FILTER_ALPHA_R, FILTER_ALPHA_T, MAX_STALE)

# ---------------------------------------------------------------------------
# 6-DOF grasp geometry — IN-PLANE approach (from opus, ported to base frame)
# ---------------------------------------------------------------------------
def build_grasp_orientation(approach, z_obj):
    """Target EE rotation for a side/in-plane approach (opus geometry).
        EE-X := approach  (in-plane approach vector pointing toward object)
        EE-Z := z_obj     (surface normal; jaw straddle axis if JAW_AXIS_IS_EEZ)
    SVD-orthonormalized to guarantee a valid right-handed rotation."""
    x_axis = approach / (np.linalg.norm(approach) + 1e-12)
    n = z_obj / (np.linalg.norm(z_obj) + 1e-12)
    if JAW_AXIS_IS_EEZ:
        z_axis = n; y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
    else:
        y_axis = n; z_axis = np.cross(x_axis, y_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
    return orthonormalize(R)

def compute_grasp(R_obj_base, t_obj_base, T_base_tip):
    """In-plane approach grasp geometry, all in the BASE frame.

    Approach direction is whichever of the board's X- or Y-axis is closer to
    the arm->object direction (both are in-plane axes fixed to the board),
    flipped so the arm arrives from its current side. Picking the nearer axis
    lets the gripper come in along the closest edge instead of always along X;
    the two choices differ by a 90 deg rotation of the gripper about the
    surface normal.

    Returns (pre_grasp_T, grasp_T) as full SE(3) TIP target poses in base.
    """
    p_obj = t_obj_base
    z_obj = R_obj_base[:, 2]
    z_obj = z_obj / (np.linalg.norm(z_obj) + 1e-12)

    # Board's two in-plane axes, projected into the board plane and re-orthonormalised
    # against z_obj (guards against filter drift).
    def _in_plane_axis(col):
        a = R_obj_base[:, col].copy().astype(float)
        a -= np.dot(a, z_obj) * z_obj
        return a / (np.linalg.norm(a) + 1e-12)
    board_x = _in_plane_axis(0)
    board_y = _in_plane_axis(1)

    # Direction from the arm to the object, projected into the board plane.
    arm_to_obj = p_obj - T_base_tip[:3, 3]
    d = arm_to_obj - np.dot(arm_to_obj, z_obj) * z_obj
    d /= (np.linalg.norm(d) + 1e-12)

    # Pick whichever board axis is more parallel to that direction (undirected line,
    # so compare |dot|): that's the "closer" axis to approach along.
    approach = board_x if abs(np.dot(d, board_x)) >= abs(np.dot(d, board_y)) else board_y
    # Orient it so it points from the arm's side INTO the board face.
    if np.dot(approach, arm_to_obj) < 0:
        approach = -approach

    R_target = build_grasp_orientation(approach, z_obj)
    _vert = np.array([0.0, 0.0, GRASP_VERTICAL_OFFSET])       # base-frame vertical lift (fixes 4 cm low)
    pre_grasp_pos = p_obj - approach * PRE_GRASP_DIST + GRASP_Z_OFFSET * z_obj + _vert
    grasp_pos     = p_obj - approach * GRASP_DEPTH    + GRASP_Z_OFFSET * z_obj + _vert
    return make_T(R_target, pre_grasp_pos), make_T(R_target, grasp_pos)

# ---------------------------------------------------------------------------
# TransformManager + LSL
# ---------------------------------------------------------------------------
try:
    tm = TransformManager("T_EE_cam.npz")
except (FileNotFoundError, RuntimeError) as e:
    print(f"[ERROR] {e}"); raise SystemExit(1)

T_EE_tip     = tm.T_EE_tip       # constant flange -> tip (pure +X translation)
T_tip_EE_inv = inv_T(T_EE_tip)   # tip target -> EE (flange) target

# EE<-cam rotation for expressing the pen mean vector in the EE frame before tilt
# classification. Loaded EXACTLY like pens_mean_tilt_v2.py — straight from the
# T_EE_cam.npz hand-eye file — so pen classification is byte-for-byte identical to
# the standalone pen script (falls back to the CAMERA frame if the npz is absent).
PEN_HANDEYE = "T_EE_cam.npz"
R_ee_cam = np.eye(3); PEN_FRAME = "CAMERA"
if os.path.exists(PEN_HANDEYE):
    _he = np.load(PEN_HANDEYE)
    if "T_EE_cam" in _he:
        R_ee_cam = _he["T_EE_cam"][:3, :3].astype(float); PEN_FRAME = "EE"
if PEN_FRAME == "CAMERA":
    print("[warn] no T_EE_cam — pen tilt will be classified in the CAMERA frame.")

# The two pens used for the ORIENT gesture (indices 0,1 in the tracker's table).
if PENS_AVAILABLE:
    try:
        penA, penB = dt.PENS[0], dt.PENS[1]
    except Exception as _pe:                    # noqa: BLE001
        penA = penB = None
        PENS_AVAILABLE = False
        print(f"[WARN] could not get pens from tracker ({_pe}); "
              f"pen-driven ORIENT disabled.")
else:
    penA = penB = None

outlet = pylsl.StreamOutlet(pylsl.StreamInfo(
    "z1_cmd", "Z1 cartesian cmd", 7, 0.0, pylsl.cf_float32, "approach_controller"))

# FLUENCY MARKERS. A string marker stream the robot publishes so LabRecorder captures the
# robot's own timing in the SAME XDF as the runner's LoggerMarkers (FACE/PAIR/FREEZE). LSL
# time-aligns the two streams, so functional delay and human-idle can be computed offline.
# Per presented face the robot emits, on one clock:
#   MOVE_START:<face>   -> robot began moving to present this face
#   READY:<face>        -> robot reached/settled at the presentation pose
#   FUNC_DELAY:<face>=<s> ... -> convenience: intervals measured at FREEZE (see below)
try:
    robot_markers = pylsl.StreamOutlet(pylsl.StreamInfo(
        "RobotMarkers", "Markers", 1, 0.0, "string", "approach_controller_markers"))
    print("[LSL] outlet 'RobotMarkers' ready (fluency timing -> XDF).")
except Exception as _e:                                   # noqa: BLE001
    robot_markers = None
    print(f"[WARN] RobotMarkers outlet init failed ({_e}); fluency markers off.")

# ORIENT TELEMETRY (numeric stream). One sample PER ORIENT FRAME, so the pen-line INTERSECTION
# the board chases (base frame) is recorded over time in the SAME XDF as the runner's markers.
# This is the evidence trail for "why did participant X do worse" — e.g. an intersection that
# keeps jumping (large gap / high per-frame motion) or frequent pen dropouts (n_pens < 2). ORIENT
# only runs on C2 trials, so every sample here belongs to a C2 trial (the 'trial' channel + the
# runner's START:...;TRIAL=;COND= marker let you slice it per trial offline).
ORIENT_TLM_CH = ["n_pens", "penA_seen", "penB_seen", "Xint_x", "Xint_y", "Xint_z",
                 "gap_mm", "tgt_x", "tgt_y", "tgt_z", "pos_err_mm", "rot_err_deg", "trial"]
try:
    _tlm_info = pylsl.StreamInfo("OrientTelemetry", "Telemetry", len(ORIENT_TLM_CH),
                                 0.0, pylsl.cf_float32, "approach_controller_orient_tlm")
    _tlm_chns = _tlm_info.desc().append_child("channels")           # self-describing channel labels
    for _c in ORIENT_TLM_CH:
        _tlm_chns.append_child("channel").append_child_value("label", _c)
    orient_tlm = pylsl.StreamOutlet(_tlm_info)
    print(f"[LSL] outlet 'OrientTelemetry' ready ({len(ORIENT_TLM_CH)} ch, per-frame -> XDF).")
except Exception as _e:                                   # noqa: BLE001
    orient_tlm = None
    print(f"[WARN] OrientTelemetry outlet init failed ({_e}); intersection logging off.")

# ------------------------------------------------------------------------------------------------
# DIRECT CSV LOG (no LabRecorder / XDF needed). One "long" row per ORIENT frame written straight to
# disk: the 6-DOF board + tip pose, the pen-line intersection, pen visibility, servo error, and the
# trial id — plus event rows (START / PAIR / FREEZE / STOP) inline, so completion time and freezes
# are recoverable from this one file. Angles are radians (roll,pitch,yaw), positions metres, in the
# base frame (same convention as z1_pos). This runs ALONGSIDE the LSL stream above; if you record an
# XDF too you get both, and if you don't, this CSV is fully self-contained. Set ORIENT_CSV_LOG=False
# to disable. Opened lazily on the first ORIENT frame, in ORIENT_CSV_DIR, one file per run.
ORIENT_CSV_LOG = True
ORIENT_CSV_DIR = os.environ.get("ORIENT_CSV_DIR", "recordings")
ORIENT_CSV_COLS = ["wall_iso", "t_lsl", "event", "participant", "trial", "cond", "block",
                   "n_pens", "penA_seen", "penB_seen",
                   "Xint_x", "Xint_y", "Xint_z", "gap_mm", "pen_angle_deg",
                   "board_x", "board_y", "board_z", "board_roll", "board_pitch", "board_yaw",
                   "tip_x", "tip_y", "tip_z", "tip_roll", "tip_pitch", "tip_yaw",
                   "tgt_x", "tgt_y", "tgt_z", "pos_err_mm", "rot_err_deg",
                   "func_delay_s", "present_time_s", "cmd_lin_mm_s", "cmd_ang_deg_s", "phase"]
_orient_csv_f = None
_orient_csv_w = None
_orient_csv_path = None
_orient_csv_pending = 0
_csv_event = ""                      # event tag to attach to the NEXT logged frame (START/PAIR/...)


def _pose6(T):
    """(x, y, z, roll, pitch, yaw) from a 4x4 base-frame transform; NaNs if T is None."""
    if T is None:
        return [float("nan")] * 6
    x, y, z = (float(v) for v in T[:3, 3])
    r, p, yw = (float(v) for v in R_to_rpy(T[:3, :3]))
    return [x, y, z, r, p, yw]


def _orient_csv_write(row):
    """Append one dict-row to the ORIENT CSV, opening the file (with header) on first use."""
    global _orient_csv_f, _orient_csv_w, _orient_csv_path, _orient_csv_pending
    if not ORIENT_CSV_LOG:
        return
    if _orient_csv_w is None:
        try:
            os.makedirs(ORIENT_CSV_DIR, exist_ok=True)
            _orient_csv_path = os.path.join(
                ORIENT_CSV_DIR, f"orient_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            _orient_csv_f = open(_orient_csv_path, "w", newline="")
            _orient_csv_w = csv.DictWriter(_orient_csv_f, fieldnames=ORIENT_CSV_COLS)
            _orient_csv_w.writeheader()
            print(f"[csv] logging ORIENT frames -> {_orient_csv_path}")
        except Exception as _e:                               # noqa: BLE001
            print(f"[WARN] could not open ORIENT CSV ({_e}); CSV logging off.")
            globals()["ORIENT_CSV_LOG"] = False
            return
    try:
        _orient_csv_w.writerow({k: row.get(k, "") for k in ORIENT_CSV_COLS})
        _orient_csv_pending += 1
        if _orient_csv_pending >= 30:                         # flush a few times per second
            _orient_csv_f.flush()
            _orient_csv_pending = 0
    except Exception:                                         # noqa: BLE001
        pass


@atexit.register
def _orient_csv_close():
    try:
        if _orient_csv_f is not None:
            _orient_csv_f.flush(); _orient_csv_f.close()
    except Exception:                                         # noqa: BLE001
        pass


def _emit_marker(s):
    """Push a fluency marker to LSL (if available) and echo to the console."""
    if robot_markers is not None:
        try:
            robot_markers.push_sample([s])
        except Exception:                                 # noqa: BLE001
            pass
    print(f"[fluency] {s}")


def _orient_trial_summary():
    """Consolidate this C2 trial's telemetry into ONE marker string (also captured in the XDF).
    Bundles: intersection-motion stats (how much the pen intersection wandered = jitter evidence),
    pen out-of-frame counts, and the functional-delay / present-time aggregates. All values are for
    the ORIENT episode that just finished (== one C2 trial)."""
    # Intersection motion: per-axis variance from running sum/sumsq -> RMS wander about the mean.
    if _xint_n > 0:
        _mean = _xint_sum / _xint_n
        _var  = np.maximum(_xint_sumsq / _xint_n - _mean * _mean, 0.0)   # clamp fp negatives
        _rms_mm  = float(np.sqrt(float(np.sum(_var)))) * 1000.0          # 3D RMS distance from centroid
        _std_mm  = [float(np.sqrt(v)) * 1000.0 for v in _var]           # per-axis std (mm)
    else:
        _rms_mm = float("nan"); _std_mm = [float("nan")] * 3
    _path_mm = _xint_path * 1000.0
    _oof_frac = (_pen_oof_frames / _orient_frame_count) if _orient_frame_count else float("nan")
    _fd_mean = float(np.mean(_fd_list)) if _fd_list else float("nan")
    _pt_mean = float(np.mean(_pt_list)) if _pt_list else float("nan")
    return (f"TRIAL_SUMMARY trial={_trial_id} cond={_trial_cond} block={_trial_block} "
            f"P={_trial_part} orient_frames={_orient_frame_count} "
            f"int_n={_xint_n} int_rms_mm={_rms_mm:.2f} "
            f"int_std_mm=[{_std_mm[0]:.2f},{_std_mm[1]:.2f},{_std_mm[2]:.2f}] "
            f"int_pathlen_mm={_path_mm:.1f} "
            f"pen_oof_frames={_pen_oof_frames} pen_oof_events={_pen_oof_events} "
            f"penA_miss={_penA_miss_frames} penB_miss={_penB_miss_frames} "
            f"oof_frac={_oof_frac:.3f} "
            f"nfaces={len(_fd_list)} func_delay_mean={_fd_mean:.3f} present_time_mean={_pt_mean:.3f}")

# The outlet is written from BOTH the main loop and the 100 Hz orientation thread.
# Funnel every push through one lock so commands can't interleave, and sanitise the
# twist: any non-finite value becomes 0 and every channel is clamped to the [-1, 1]
# the C++ bridge expects. A NaN from a degenerate se3_log would otherwise be sent
# verbatim and integrated by the arm.
_cmd_lock = threading.Lock()

def _sanitize6(v6):
    out = []
    for x in v6:
        x = float(x)
        if not np.isfinite(x):
            x = 0.0
        out.append(max(-1.0, min(1.0, x)))
    return out

def send_cmd(v6, gripper):
    g = float(gripper)
    if not np.isfinite(g):
        g = 0.0
    s = _sanitize6(v6)
    # Full 6-DOF: roll, pitch AND yaw are all commanded in every state (yaw = index 2 of the
    # [roll, pitch, yaw, x, y, z] command). No axis is zeroed.
    sample = s + [g]
    with _cmd_lock:
        outlet.push_sample(sample)

def send_zeros(gripper=GRIPPER_OPEN):
    g = float(gripper)
    if not np.isfinite(g):
        g = 0.0
    with _cmd_lock:
        outlet.push_sample([0.0] * 6 + [g])

# ---------------------------------------------------------------------------
# z1_pos inlet — reads endPosture [roll, pitch, yaw, x, y, z] from the arm
# ---------------------------------------------------------------------------
_z1_pos_inlet  = None

# Orientation-control thread state (PRESENT / HOLD)
_orient_stop   = threading.Event()
_orient_thread = None
_latest_rpy    = None
_orient_lock   = threading.Lock()
_erot_prev     = None       # previous SO(3) error vector (for the D term)
_erot_prev_t   = None
_erot_int      = np.zeros(3) # accumulated SO(3) error integral (for the I term)
# Shared setpoint the worker tracks (base-frame target rotation) + active gains.
# PRESENT sets identity/Kp_present; ORIENT sets the pen-classified tilt (then
# level again) via set_orient_target with Kp_orient/Ki_orient/Kd_orient.
_orient_target_R = np.eye(3)
_orient_Kp       = Kp_present
_orient_Ki       = Ki_present
_orient_Kd       = Kd_present
_orient_reset_d  = False     # drop one D/I sample after a setpoint jump (no spike)

def set_orient_target(roll_deg, pitch_deg, yaw_deg, kp, ki, kd):
    """Thread-safe update of the worker's target rotation and PID gains."""
    global _orient_target_R, _orient_Kp, _orient_Ki, _orient_Kd, _orient_reset_d
    with _orient_lock:
        _orient_target_R = rpy_to_R(np.deg2rad(roll_deg),
                                    np.deg2rad(pitch_deg),
                                    np.deg2rad(yaw_deg))
        _orient_Kp = kp; _orient_Ki = ki; _orient_Kd = kd
        _orient_reset_d = True   # next worker tick recomputes D and drops the I accumulator

def get_z1_pos():
    global _z1_pos_inlet
    if _z1_pos_inlet is None:
        streams = pylsl.resolve_streams()
        for s in streams:
            if s.name() == "z1_pos":
                _z1_pos_inlet = pylsl.StreamInlet(s)
                print("[z1_pos] inlet connected")
                break
    if _z1_pos_inlet is None:
        return None
    # pull_chunk + take last: avoids returning stale FIFO samples when C++ publishes
    # faster than Python consumes (pull_sample returns the OLDEST buffered entry).
    samples, _ = _z1_pos_inlet.pull_chunk(timeout=0.0)
    if not samples:
        return None
    return np.array(samples[-1], dtype=np.float64)

# ---------------------------------------------------------------------------
# ORIENT interrupt inlet — a "PAIR" marker from logger3.py cuts the tilt hold
# ---------------------------------------------------------------------------
# logger3.py pushes a "PAIR" marker on the first press of any button pair. While
# ORIENT is in its TILT phase, receiving that marker ends the hold immediately
# (arm returns to level) instead of waiting out ORIENT_HOLD_T. A daemon thread
# resolves the marker stream (retrying so start order doesn't matter), pulls
# markers, and sets an Event; the main loop consumes it ONLY during ORIENT/TILT.
# Independent of the arm's own z1_pos inlet; both use plain LSL discovery.
ORIENT_MARKER_STREAM = "LoggerMarkers"
_orient_interrupt    = threading.Event()   # set on PAIR  -> resume from a 'b' freeze
_start_requested     = threading.Event()   # set on START -> gate for ORIENT to begin (logger 'Enter')
# --- LSL-driven target face (runner's Williams order; pointer advanced by each correct PAIR) ---
_orient_seq_lock  = threading.Lock()
_orient_seq_order = None                    # face order for the running sequence (from START:<seq>)
_orient_face_idx  = 0                       # pointer into _orient_seq_order; the CURRENT target face
_orient_face_explicit = None                # exact required face from the runner's FACE:<n> marker
# TRIAL IDENTITY carried in the runner's START:...;P=;TRIAL=;COND=;BLOCK= marker (C2 trials only).
# Used to TAG telemetry samples + fluency/summary markers so each is attributable to a C2 trial.
_trial_id   = None                          # global trial index (1..27) from START TRIAL=
_trial_cond = None                          # condition string from START COND= (always C2 here)
_trial_block = None                         # block index from START BLOCK=
_trial_part = None                          # participant id from START P=
                                            # (AUTHORITATIVE — always matches the displayed grid)
_orient_freeze_req   = threading.Event()    # runner FREEZE:1/0 -> requested orient_frozen state (set=frozen)
_orient_freeze_dirty = threading.Event()    # a fresh FREEZE command arrived from LSL (apply once)

def _first_int(s):
    """First run of digits in s -> int, else None (parses '3' out of ':3;P=...')."""
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif num:
            break
    return int(num) if num else None

def _orient_marker_listener():
    """Resolve the logger's 'LoggerMarkers' stream and react to its markers:
      * 'START:<seq>' -> set _start_requested (PRESENT waits for this to enter ORIENT).
      * 'PAIR:<face>' -> set _orient_interrupt (resume tracking from a 'b' freeze).
      * 'STOP'        -> set _task_finished (return home + hand-track), in case the logger
                         itself emits STOP (the grid also emits it on GridControl).
    Re-resolves if the logger restarts / disappears."""
    global _orient_seq_order, _orient_face_idx, _orient_face_explicit
    global _trial_id, _trial_cond, _trial_block, _trial_part
    inlet = None
    while True:
        if inlet is None:
            try:
                for s in pylsl.resolve_streams(wait_time=1.0):
                    if s.name() == ORIENT_MARKER_STREAM:
                        inlet = pylsl.StreamInlet(s)
                        print(f"[interrupt] '{ORIENT_MARKER_STREAM}' inlet connected")
                        break
            except Exception:                       # noqa: BLE001
                inlet = None
            if inlet is None:
                time.sleep(0.5)                     # logger not up yet — retry
                continue
        try:
            sample, _ts = inlet.pull_sample(timeout=0.5)
        except Exception:                           # noqa: BLE001 (logger vanished)
            inlet = None
            continue
        if not sample:
            continue
        m = str(sample[0]).strip().upper()
        if m.startswith("START"):
            # START:<seq>;P=...  -> reconstruct the Williams face order and reset the pointer, so the
            # robot presents the correct face without the pens having to select it.
            _seq = _first_int(m[len("START"):])
            with _orient_seq_lock:
                _orient_face_explicit = None        # wait for the runner's first FACE:<n>
                if _seq is not None and 1 <= _seq <= 8:
                    _orient_seq_order = williams_sequences(8)[_seq - 1]
                    _orient_face_idx  = 0
                    print(f"[interrupt] START seq {_seq} -> face order {_orient_seq_order}")
            # Trial identity (m is upper-cased): START:<seq>;P=..;TRIAL=..;COND=..;BLOCK=..
            def _kv(_key):
                for _tok in m.split(";"):
                    if _tok.strip().startswith(_key + "="):
                        return _tok.split("=", 1)[1].strip()
                return None
            _tv = _first_int(_kv("TRIAL") or ""); _trial_id = _tv if _tv is not None else _trial_id
            _bv = _first_int(_kv("BLOCK") or ""); _trial_block = _bv if _bv is not None else _trial_block
            _trial_cond = _kv("COND") or _trial_cond
            _trial_part = _kv("P") or _trial_part
            print(f"[interrupt] START trial={_trial_id} cond={_trial_cond} "
                  f"block={_trial_block} P={_trial_part}")
            _start_requested.set()                  # logger pressed Enter -> allow ORIENT to begin
        elif m.startswith("FACE"):
            # FACE:<n> from the runner = the face currently REQUIRED on the display. Authoritative,
            # so the robot always shows/aims exactly the face the participant is told to press.
            _f = _first_int(m[len("FACE"):])
            if _f is not None:
                with _orient_seq_lock:
                    _orient_face_explicit = _f
        elif m.startswith("PAIR"):
            # PAIR:<face> fires on EVERY press; advance the fallback pointer only when it matches the
            # current target face (a correct press), mirroring the runner's own grid advance.
            _face = _first_int(m[len("PAIR"):])
            with _orient_seq_lock:
                if (_orient_seq_order and _orient_face_idx < len(_orient_seq_order)
                        and _face == _orient_seq_order[_orient_face_idx]):
                    _orient_face_idx += 1
            _orient_interrupt.set()                 # resume tracking from a 'b' freeze
        elif m.startswith("FREEZE"):
            # FREEZE:1 / FREEZE:0 from the runner ('b' pressed on the runner screen) -> freeze / resume.
            _v = _first_int(m[len("FREEZE"):])
            if _v is None or _v != 0:
                _orient_freeze_req.set()            # anything but 0 -> freeze
            else:
                _orient_freeze_req.clear()          # 0 -> resume
            _orient_freeze_dirty.set()
        elif m.startswith("STOP"):
            _task_finished.set()                    # logger-side stop -> return home + hand-track

threading.Thread(target=_orient_marker_listener, daemon=True).start()

# ---------------------------------------------------------------------------
# GridControl inlet — grid_face_sequence.py pushes "STOP" on the 'GridControl'
# stream when the LAST face of the study sequence has been pressed (the task is
# finished). We use that as the "task done" signal: the arm returns to the home
# ('s') pose and switches to hand-tracking for the handover. (The logger doesn't
# relay this; the grid emits it directly, so we subscribe to GridControl.)
GRID_CONTROL_STREAM = "GridControl"
_task_finished      = threading.Event()

def _grid_control_listener():
    """Resolve 'GridControl' and set _task_finished on every 'STOP' sample.
    Re-resolves if the grid restarts / disappears. Same pattern as the PAIR listener."""
    inlet = None
    while True:
        if inlet is None:
            try:
                for s in pylsl.resolve_streams(wait_time=1.0):
                    if s.name() == GRID_CONTROL_STREAM:
                        inlet = pylsl.StreamInlet(s)
                        print(f"[grid] '{GRID_CONTROL_STREAM}' inlet connected")
                        break
            except Exception:                       # noqa: BLE001
                inlet = None
            if inlet is None:
                time.sleep(0.5)                     # grid not up yet — retry
                continue
        try:
            sample, _ts = inlet.pull_sample(timeout=0.5)
        except Exception:                           # noqa: BLE001 (grid vanished)
            inlet = None
            continue
        if sample and str(sample[0]).strip().upper().startswith("STOP"):
            _task_finished.set()

threading.Thread(target=_grid_control_listener, daemon=True).start()

# ---------------------------------------------------------------------------
# 100 Hz orientation-control thread (PRESENT / HOLD)
# ---------------------------------------------------------------------------
def _orientation_worker():
    """PID servo that drives the EE orientation to a SHARED SETPOINT at ORIENT_HZ,
    decoupled from the camera. PRESENT sets the setpoint to identity (level);
    ORIENT sets the pen-classified tilt, then level again.

    PID on the SO(3) body error e_rot = log(R_cur^T @ R_target): P for responsiveness,
    D (finite-difference of the rotation vector) for damping, and I (time integral of
    e_rot, clamped for anti-windup) to eliminate the steady-state offset that a pure
    velocity command leaves when arm stiction / load droop stops the last fraction of
    a degree from closing. The integral is dropped on every setpoint jump.

    The orientation error is taken as a proper SO(3) geodesic in the EE BODY
    frame — omega_b = log(R_cur^T @ R_target) is exactly the body angular
    velocity that rotates the current pose onto the target — rather than feeding
    a raw ZYX-Euler triple into three independent P loops. This:
      (a) matches the body-frame angular-velocity convention the SDK integrates,
          the same convention se3_control already uses (cmd = +Kp * log error);
      (b) takes the SHORTEST rotation to the target (the three Euler angles are
          about coupled intermediate axes, so per-axis P is not a true descent);
      (c) stays smooth through pitch = +/-90 deg (Euler gimbal lock) and the yaw
          +/-pi wrap, where the Euler triple and its finite-difference derivative
          blow up. For R_target = I it reduces to the level-and-hold law.
    """
    global _latest_rpy, _erot_prev, _erot_prev_t, _erot_int, _orient_reset_d
    dt_ctrl = 1.0 / ORIENT_HZ
    cmd_prev = np.zeros(3)                       # last angular cmd, for slew limiting
    max_step = ORIENT_MAX_ANG_ACC * dt_ctrl      # max change per tick (rad/s)
    while not _orient_stop.is_set():
        t0  = time.time()
        pos = get_z1_pos()
        if pos is not None:
            roll, pitch, yaw = pos[0], pos[1], pos[2]
            now = time.time()
            with _orient_lock:
                _latest_rpy = (roll, pitch, yaw)
                R_tgt = _orient_target_R          # current setpoint (base frame)
                kp, ki, kd = _orient_Kp, _orient_Ki, _orient_Kd
                if _orient_reset_d:               # setpoint just jumped -> drop D and I
                    _erot_prev = None; _erot_prev_t = None
                    _erot_int = np.zeros(3); _orient_reset_d = False
            # Body-frame rotation that carries the current EE pose onto R_tgt.
            R_cur = rpy_to_R(roll, pitch, yaw)
            e_rot = so3_log(R_cur.T @ R_tgt)       # axis*angle; |e_rot| == error angle
            # Time step since the last tick (0 right after a setpoint jump, so the
            # first I/D sample after a jump contributes nothing).
            dt = (now - _erot_prev_t) if _erot_prev_t is not None else 0.0
            if dt < 0.0:
                dt = 0.0
            # Derivative on the rotation-vector error: continuous, no Euler wrap.
            if _erot_prev is not None and dt > 1e-6:
                d_rot = (e_rot - _erot_prev) / dt
            else:
                d_rot = np.zeros(3)
            # Integral of the error with clamped anti-windup (per-axis).
            _erot_int = np.clip(_erot_int + e_rot * dt,
                                -ORIENT_I_CLAMP, ORIENT_I_CLAMP)
            _erot_prev   = e_rot
            _erot_prev_t = now
            cmd = np.zeros(6)
            # PID: proportional + integral + derivative on the SO(3) body error.
            ang = np.clip(kp * e_rot + ki * _erot_int + kd * d_rot,
                          -MAX_ANG, MAX_ANG)          # EE body angular velocity
            # Slew-rate limit: ramp the command toward the PID target instead of
            # stepping to it, so the first tick after a setpoint jump eases in.
            ang = cmd_prev + np.clip(ang - cmd_prev, -max_step, max_step)
            cmd_prev = ang
            cmd[0:3] = ang
            send_cmd(cmd, GRIPPER_CLOSE)
        elapsed = time.time() - t0
        if elapsed < dt_ctrl:
            time.sleep(dt_ctrl - elapsed)

# ---------------------------------------------------------------------------
# SE(3) screw servo — controls FLANGE (EE), with TCP correction  (from 1706)
# ---------------------------------------------------------------------------
# SE(3) servo PID state (integral + derivative on the twist error). Reset at each
# new servo target so a fresh move does not inherit the previous integral.
_se3_int_ang  = np.zeros(3)
_se3_int_lin  = np.zeros(3)
_se3_prev_ang = None
_se3_prev_lin = None
_se3_prev_t   = None

def _reset_se3_pid():
    """Drop the SE(3) servo integral/derivative state (DETECT->SERVO, SERVO->GRASP,
    LIFT), so a new target starts with a clean integrator."""
    global _se3_int_ang, _se3_int_lin, _se3_prev_ang, _se3_prev_lin, _se3_prev_t
    _se3_int_ang = np.zeros(3); _se3_int_lin = np.zeros(3)
    _se3_prev_ang = None; _se3_prev_lin = None; _se3_prev_t = None

def se3_control(tip_target_T, T_base_EE, kp_lin, kp_ang, return_debug=False,
                ki_lin=None, kd_lin=None, ki_ang=None, kd_ang=None, max_lin=None):
    """PID twist along the SE(3) screw geodesic (theta-u PBVS,
    Chaumette & Hutchinson 2006).

    Maps the desired TIP pose to the EE/flange pose via T_EE_tip before
    computing the log error. This is critical: the SDK integrates angular
    velocity in the EE body frame and linear velocity as the EE-origin
    world velocity. Sending the tip-origin velocity on the linear channel
    while omega acts at the flange introduces an omega x r lever-arm error
    that this mapping removes. (Lynch & Park 2017, adjoint; Murray, Li &
    Sastry 1994.)

    Returns (cmd6, pos_err, rot_err); pos_err is the TIP position error.
    If return_debug=True, also returns a 4th element: a diagnostics dict.
    """
    global _se3_int_ang, _se3_int_lin, _se3_prev_ang, _se3_prev_lin, _se3_prev_t
    ee_target_T = tip_target_T @ T_tip_EE_inv      # tip -> EE target
    T_err = inv_T(T_base_EE) @ ee_target_T          # relative pose in EE body
    omega_b, v_b = se3_log(T_err)

    tip_now = (T_base_EE @ T_EE_tip)[:3, 3]
    pos_err = float(np.linalg.norm(tip_target_T[:3, 3] - tip_now))
    rot_err = float(np.linalg.norm(omega_b))

    if SCREW_COUPLED:
        v_world = T_base_EE[:3, :3] @ v_b            # EE-origin world velocity
    else:
        v_world = ee_target_T[:3, 3] - T_base_EE[:3, 3]  # straight world line

    # --- PID on the two error signals: angular = omega_b (EE body), linear = v_world.
    # I removes the steady-state residual a pure-P velocity servo leaves; D damps.
    # I/D gains scale with the caller's P gain, so GRASP's half-P also halves I/D.
    now = time.time()
    dt = (now - _se3_prev_t) if _se3_prev_t is not None else 0.0
    if dt < 0.0 or dt > 0.5:                        # first call / long gap -> skip I/D
        dt = 0.0
    # I/D gains default to scaling with the caller's P gain (so GRASP's half-P also
    # halves I/D), but any can be overridden explicitly for a task-specific tuning
    # (e.g. GOHOME) without touching module globals.
    sc_ang = kp_ang / Kp_rot if Kp_rot > 1e-9 else 1.0
    sc_lin = kp_lin / Kp_pos if Kp_pos > 1e-9 else 1.0
    ki_a = Ki_rot * sc_ang if ki_ang is None else ki_ang
    kd_a = Kd_rot * sc_ang if kd_ang is None else kd_ang
    ki_l = Ki_pos * sc_lin if ki_lin is None else ki_lin
    kd_l = Kd_pos * sc_lin if kd_lin is None else kd_lin
    lim_lin = MAX_LIN if max_lin is None else max_lin
    if _se3_prev_ang is not None and dt > 1e-6:
        d_ang = (omega_b - _se3_prev_ang) / dt
        d_lin = (v_world - _se3_prev_lin) / dt
    else:
        d_ang = np.zeros(3); d_lin = np.zeros(3)
    _se3_int_ang = np.clip(_se3_int_ang + omega_b * dt, -SE3_I_CLAMP_ANG, SE3_I_CLAMP_ANG)
    _se3_int_lin = np.clip(_se3_int_lin + v_world * dt, -SE3_I_CLAMP_LIN, SE3_I_CLAMP_LIN)
    _se3_prev_ang = omega_b.copy(); _se3_prev_lin = v_world.copy(); _se3_prev_t = now

    cmd = np.zeros(6)
    ang_raw = kp_ang * omega_b + ki_a * _se3_int_ang + kd_a * d_ang
    lin_raw = kp_lin * v_world + ki_l * _se3_int_lin + kd_l * d_lin
    cmd[0:3] = np.clip(ang_raw, -MAX_ANG, MAX_ANG)  # EE body angular
    cmd[3:6] = np.clip(lin_raw, -lim_lin, lim_lin)  # EE-origin world linear
    if return_debug:
        dbg = {
            "tip_now":    tip_now,
            "tip_target": tip_target_T[:3, 3].copy(),
            "err_vec":    tip_target_T[:3, 3] - tip_now,   # base frame, m
            "omega_b":    omega_b,                          # EE body, rad
            "v_b":        v_b,                              # EE body, m
            "v_world":    v_world,                          # base frame, m/s-ish
            "ang_raw":    ang_raw,   "ang_cmd": cmd[0:3].copy(),
            "lin_raw":    lin_raw,   "lin_cmd": cmd[3:6].copy(),
            "ang_clipped": bool(np.any(np.abs(ang_raw) > MAX_ANG + 1e-9)),
            "lin_clipped": bool(np.any(np.abs(lin_raw) > lim_lin + 1e-9)),
        }
        return cmd, pos_err, rot_err, dbg
    return cmd, pos_err, rot_err

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
STATE_IDLE          = "IDLE"
STATE_DETECT        = "DETECT"
STATE_SERVO         = "SERVO"
STATE_GRASP         = "GRASP"
STATE_CLOSE_GRIPPER = "CLOSE_GRIPPER"
STATE_LIFT          = "LIFT"
STATE_PRESENT       = "PRESENT"
STATE_ORIENT        = "ORIENT"
STATE_HOLD          = "HOLD"
STATE_HANDING       = "HANDING"     # finger touched the object -> open gripper (release)
STATE_FAULT         = "FAULT"       # servo stalled / aborted: arm halted, waits for reset
STATE_GOHOME        = "GOHOME"      # servo to the fixed start/finish pose
STATE_TRACK_HANDS   = "TRACK_HANDS" # study finished: hold at home, track hands, finger-touch release
STATE_HOLD_WAIT     = "HOLD_WAIT"   # BETWEEN trials: hold object at start pose (no release), wait for next START

state           = STATE_IDLE
stable_count    = 0
lock_count      = 0
close_gripper_t = None
lift_start_t    = None
lift_target_T   = None
hold_printed    = False
current_gripper = GRIPPER_OPEN
_gohome_next    = STATE_IDLE   # state to enter once the GOHOME pose is reached
_gohome_settle  = 0            # consecutive in-tolerance frames during GOHOME
_handing_done_t = None         # wall-clock when the gripper opened in HANDING
_present_t      = None         # wall-clock when PRESENT leveling began
_start_orient_at = None        # wall-clock at which to actually enter ORIENT (START + delay)
START_ORIENT_DELAY_S = 3.2     # wait after the runner's START before orienting -> covers the
                               # runner's 3 s on-screen countdown (+0.2 s slack)
PRESENT_LEVEL_TIMEOUT = 5.0    # s to wait for level before proceeding anyway
                               # (a joint limit can prevent full leveling — don't hang)
pre_grasp_T     = None
grasp_T         = None
grasp_T_locked  = None
grasp_committed = False     # merged approach+grasp: True once the computed grasp pose
                            # has been frozen and the servo switched to gentle insertion
grasp_EE_R_lock = None      # EE rotation (base) frozen at GRASP entry — drift ref
grasp_tip_lock  = None      # tip position (base) frozen at GRASP entry
_grasp_prev_tip = None      # tip position at previous debug print (motion check)
_grasp_prev_t   = None      # wall-clock at previous debug print
# --- object-centre tracking (survives full marker occlusion after grasp) ---
grasp_obj_center_base = None  # object centre in BASE, captured at the SERVO->GRASP lock
grasp_obj_R_base      = None  # object rotation in BASE, captured at the same lock
obj_pose_cam          = None  # object pose (4x4 T_cam_obj) frozen at grasp (constant after)
obj_tracked           = False # True once the object centre has been captured
present_leveled = False     # PRESENT reached level; armed for the 'o' ORIENT trigger
# --- pen-driven ORIENT state ---
orient_phase      = None    # None | "WATCH" | "TILT" | "RETURN"
orient_tilt_label = None    # committed tilt label currently being held
orient_tilt_t     = None    # wall-clock start of the ORIENT_HOLD_T tilt hold
orient_interrupt_t = None   # wall-clock time a pair press latched (start of settle delay)
orient_return_t   = None    # wall-clock start of the return-to-level settle
_pen_cand_label   = None    # candidate tilt currently accumulating stability
_pen_cand_t       = None    # wall-clock the candidate first appeared
_pen_vhist        = []      # recent mean-vector samples (steadiness window)
_pen_chist        = []      # recent midpoint samples (steadiness window)
_pen_miss         = 0       # consecutive frames with <2 pens detected (dropout run)
# --- continuous pen-tracking servo target (base frame), low-passed + held on dropout ---
_orient_tgt_R_base = None   # smoothed desired board rotation (base)
_orient_tgt_t_base = None   # smoothed desired board centre   (base)
_orient_tgt_vel     = np.zeros(3)  # estimated target-centre velocity (base, m/s) for feed-forward
_orient_tgt_prev_t  = None         # previous smoothed target centre (finite-difference reference)
_orient_tgt_prev_wt = None         # wall-clock of the previous target sample
_orient_retreat_from = None # tip position (base) where both pens were lost (retreat anchor)
_orient_pens_lost_t = None  # wall-clock both pens were first lost (None while visible); slow-home after >1s
# --- PER-C2-TRIAL telemetry accumulators (reset every ORIENT entry; summarised at STOP) ---
_orient_frame_count = 0       # total ORIENT frames this trial (denominator for OOF fractions)
_pen_oof_frames  = 0          # frames with < 2 pens tracked (a marker/pen out of frame)
_pen_oof_events  = 0          # number of DROPOUT EPISODES (2-pens -> <2-pens transitions)
_penA_miss_frames = 0         # frames penA was not tracked
_penB_miss_frames = 0         # frames penB was not tracked
_pen_prev_full   = True       # were both pens visible on the previous frame? (episode edge detector)
_xint_n     = 0               # count of frames with a valid intersection (both pens, well-conditioned)
_xint_sum   = np.zeros(3)     # running sum of intersection position (base) for the mean
_xint_sumsq = np.zeros(3)     # running sum of squares for the per-axis variance / RMS jitter
_xint_prev  = None            # previous intersection sample, for path-length accumulation
_xint_path  = 0.0             # cumulative path length of the intersection (base, m) = total wander
_fd_list    = []              # per-face functional delays (s) this trial
_pt_list    = []              # per-face present (settle) times (s) this trial
_orient_T_EE_obj   = None   # object->EE transform captured ONCE at ORIENT entry (rigid grip)
_orient_slip_run   = 0      # consecutive frames the FK-vs-vision board pose has exceeded the slip gate
_orient_ref_R      = None   # board orientation captured at ORIENT entry (level ref for the tilt clamp)
_orient_stall_ref  = None   # tip position ref for the stuck detector
_orient_stall_rot  = None   # rot_err ref for the stuck detector (rotation-aware progress)
_orient_stall_t    = None   # wall-clock of the last tip/rotation progress
_orient_trans_only_until = 0.0  # translation-only (rotation dropped) until this wall-clock time
_orient_button_hold_until = 0.0  # hold the arm still after a PAIR press until this wall-clock time
_orient_prev_ang   = np.zeros(3)  # previous angular command (base) for the slew-rate limiter
_orient_prev_ang_t = None         # wall-clock of the previous angular command
orient_frozen      = False  # 'b' key toggles freeze/unfreeze on the SAME face (press again to resume)
_orient_return_home = False # set on a button press (PAIR): drive to the initial pose, then resume
_orient_return_t0   = 0.0   # wall-clock when the return-to-initial move began (for the timeout)
# --- finger-touch handover state ---
_hand_touch_count = 0       # consecutive frames a fingertip was on the object
_orient_hands_cache = []    # last MediaPipe hands result (reused between throttled detections)
_pens_in_view     = False   # a pen was tracked this WATCH frame -> block the release
hand_open_printed = False   # one-shot log flag for HANDING
# Per-hand 3D-touch state (keyed by handedness label), ported from finger_touch_check.
_dist_ema    = {}           # label -> {tid: smoothed signed distance to plane (mm)}
_infoot_last = {}           # label -> {tid: last in-footprint bool}
_ema_stale   = {}           # label -> consecutive frames with no accepted pose
# --- servo stall watchdog / fault state ---
_stall_best       = None    # best (smallest) pos_err seen since the move began
_stall_t          = None    # wall-clock of the last real improvement
fault_printed     = False   # one-shot log flag for STATE_FAULT
_dbg_frame        = 0

# ── FPS / per-section profiling ───────────────────────────────────────────────
# Measures where each frame's time goes so a slow loop can be diagnosed: is it the
# CAMERA (cap.read blocks -> long exposure / USB / resolution) or PROCESSING (board
# ArUco, pen ArUco, MediaPipe hands)? Totals are averaged and printed every
# DEBUG_EVERY frames, and the live FPS is drawn on the display.
_fps_ema      = None        # smoothed frames-per-second for the overlay
_prof_t_read  = 0.0         # accumulated cap.read() seconds since last report
_prof_t_board = 0.0         # accumulated board detectMarkers seconds
_prof_t_pen   = 0.0         # accumulated pen dt.detect seconds
_prof_t_hands = 0.0         # accumulated MediaPipe _detect_hands seconds
_prof_t_total = 0.0         # accumulated whole-frame seconds
_prof_n       = 0           # frames accumulated since last report
_loop_t_prev  = None        # perf_counter at the previous frame end
_board_cache  = None        # last (corners_list, ids) for the ORIENT board-detect throttle

# ── Fluency timing (per presented face) ───────────────────────────────────────
# All timestamps use pylsl.local_clock() so they line up with the runner's LSL markers
# (FACE/PAIR/FREEZE) in the recorded XDF. func_delay = MOVE_START - face-request time.
_fd_face          = None    # face currently being timed
_fd_request_t     = None    # clock when this face was requested (robot free to move)
_fd_move_start_t  = None    # clock when the robot began moving to present it
_fd_ready_t       = None    # clock when it settled at the presentation pose
_fd_move_emitted  = False
_fd_ready_emitted = False

def _fmt(T):
    p = T[:3, 3] * 1000
    return f"[{p[0]:+.0f} {p[1]:+.0f} {p[2]:+.0f}]mm"

def _stop_orient_thread():
    """Signal the 100 Hz PD worker to stop and JOIN it, so a stale GRIPPER_CLOSE
    from the worker cannot race a release / safe-stop command. Main-thread only."""
    global _orient_thread
    _orient_stop.set()
    if _orient_thread is not None:
        _orient_thread.join(timeout=0.5)
        _orient_thread = None

def emergency_stop():
    """Halt the arm on ANY exit path: stop the worker, then command zero twist with
    a zero (neutral) gripper rate so the arm stops and the gripper HOLDS its current
    opening (never drops a grasped object). Registered with atexit so it also fires
    on unhandled exceptions / Ctrl-C — where the C++ bridge would otherwise keep
    re-issuing the last twist forever. Idempotent; never raises."""
    try:
        _stop_orient_thread()
    except Exception:
        pass
    try:
        send_zeros(GRIPPER_NEUTRAL)
    except Exception:
        pass

atexit.register(emergency_stop)

def _reset_stall():
    """Restart the stall watchdog at the start of a new SERVO/GRASP move."""
    global _stall_best, _stall_t
    _stall_best = None
    _stall_t = None

def _servo_stalled(e_pos):
    """True only if the arm is stuck FAR from target (no pos_err improvement for
    STALL_TIMEOUT while still beyond STALL_NEAR_TARGET). Near the target a plateau
    is convergence waiting on the orientation lock, not a stall."""
    global _stall_best, _stall_t
    now = time.time()
    if e_pos < STALL_NEAR_TARGET:            # essentially arrived -> never a stall
        _stall_best = e_pos
        _stall_t = now
        return False
    if _stall_best is None or e_pos < _stall_best - STALL_MIN_IMPROVE:
        _stall_best = e_pos
        _stall_t = now
        return False
    return (now - _stall_t) > STALL_TIMEOUT

def reset_all():
    global state, stable_count, lock_count, pre_grasp_T, grasp_T, grasp_T_locked
    global grasp_committed
    global grasp_EE_R_lock, grasp_tip_lock, _grasp_prev_tip, _grasp_prev_t
    global hold_printed, current_gripper, close_gripper_t, lift_start_t, lift_target_T
    global _prev_board_R, _z1_pos_inlet, _orient_thread, _latest_rpy, _erot_prev, _erot_prev_t, _erot_int
    global present_leveled, orient_phase, orient_tilt_label, orient_tilt_t, _start_orient_at
    global orient_return_t, _pen_cand_label, _pen_cand_t, _pen_vhist, _pen_chist, _pen_miss
    global _orient_tgt_R_base, _orient_tgt_t_base, _orient_retreat_from, orient_frozen
    global _orient_tgt_vel, _orient_tgt_prev_t, _orient_tgt_prev_wt
    global _hand_touch_count, _pens_in_view, hand_open_printed, _hand_last_hull, _hand_hull_stale
    global _stall_best, _stall_t, fault_printed
    global grasp_obj_center_base, grasp_obj_R_base, obj_pose_cam, obj_tracked
    global _dist_ema, _infoot_last, _ema_stale
    # Stop orientation thread cleanly before resetting any state.
    _stop_orient_thread()
    _orient_stop.clear()
    _latest_rpy = _erot_prev = _erot_prev_t = None
    _erot_int = np.zeros(3)
    set_orient_target(0.0, 0.0, 0.0, Kp_present, Ki_present, Kd_present)   # default: level
    present_leveled = False
    orient_phase = None
    orient_tilt_label = None
    orient_tilt_t = orient_return_t = None
    _pen_cand_label = None
    _pen_cand_t = None
    _pen_vhist = []
    _pen_chist = []
    _pen_miss = 0
    _orient_tgt_R_base = None
    _orient_tgt_t_base = None
    _orient_tgt_vel = np.zeros(3); _orient_tgt_prev_t = None; _orient_tgt_prev_wt = None
    _orient_retreat_from = None
    orient_frozen = False
    _hand_touch_count = 0
    _pens_in_view = False
    hand_open_printed = False
    _hand_relock()                 # drop per-hand PnP locks
    _dist_ema = {}; _infoot_last = {}; _ema_stale = {}   # (offsets persist across reset)
    _task_finished.clear()                               # drop any queued grid STOP
    _start_requested.clear()                             # drop any queued logger START
    _start_orient_at = None                              # cancel any pending START->ORIENT countdown
    _stall_best = _stall_t = None
    fault_printed = False
    grasp_obj_center_base = None
    grasp_obj_R_base = None
    obj_pose_cam = None
    obj_tracked = False
    _hand_last_hull = None
    _hand_hull_stale = 0
    state = STATE_IDLE
    stable_count = lock_count = 0
    grasp_committed = False
    pre_grasp_T = grasp_T = grasp_T_locked = None
    grasp_EE_R_lock = grasp_tip_lock = _grasp_prev_tip = _grasp_prev_t = None
    hold_printed = False
    current_gripper = GRIPPER_OPEN
    close_gripper_t = lift_start_t = lift_target_T = None
    _prev_board_R = None
    _z1_pos_inlet = None
    pose_filter.reset()
    _reset_se3_pid()
    send_zeros(GRIPPER_OPEN)

# ---------------------------------------------------------------------------
# Fixed start/finish pose  (GOHOME)
# ---------------------------------------------------------------------------
# TIP target in the BASE frame as an endPosture sextuple [roll,pitch,yaw,x,y,z]
# (rad, m) — the same convention the z1_pos stream reports. The arm servos here
# with the existing se3_control PID at the start of every run and again after the
# object is released, so each experiment begins and ends from the same place.
HOME_RPY_XYZ = (0.0, 0.0, 0.0,  0.4, 0.0, 0.3)   # x 0.5 -> 0.4: 's' home pose 10 cm further BACK (toward base)
HOME_POSE_T  = make_T(rpy_to_R(*HOME_RPY_XYZ[:3]), np.array(HOME_RPY_XYZ[3:], float))
HOME_GRIPPER = GRIPPER_CLOSE         # 7th channel: hold the gripper CLOSED while homing
HOME_POS_TOL = 0.010                 # 10 mm arrival tolerance (was 8; commit sooner)
HOME_ROT_TOL = np.deg2rad(2.5)       # 2.5 deg arrival tolerance
HOME_SETTLE_FRAMES = 3               # consecutive in-tolerance frames before stopping

# EE-POSITION SAFETY BOX (base frame), centred at the start / home pose. During ORIENT the
# commanded tip position is kept INSIDE this box, so pen-following (or the reacquire retreat)
# can never drive the arm out to a singularity or joint limit. 40 cm box -> ±0.20 m per axis;
# edit EE_BOX_HALF per-axis if you want a different height/footprint.
EE_BOX_CENTER = HOME_POSE_T[:3, 3].copy()          # centred at the start pose
EE_BOX_HALF   = np.array([0.25, 0.25, 0.25])       # half-extents (m): the tip may move up to 25 cm
                                                   # from the start pose on each axis, then no further

def clamp_pos_to_box(p):
    """Clamp a base-frame position into the EE safety box."""
    return np.clip(np.asarray(p, float),
                   EE_BOX_CENTER - EE_BOX_HALF, EE_BOX_CENTER + EE_BOX_HALF)

def pos_in_box(p):
    """True if a base-frame position is inside the EE safety box."""
    p = np.asarray(p, float).reshape(3)
    return bool(np.all(p >= EE_BOX_CENTER - EE_BOX_HALF) and
                np.all(p <= EE_BOX_CENTER + EE_BOX_HALF))

def box_guard_vel(v6, tip_pos):
    """Zero any world-linear (v6[3:6], base frame) velocity component that would push the tip
    further OUTSIDE the EE safety box — a hard guarantee the EE position can't leave the box."""
    v6 = np.asarray(v6, float).copy()
    lo = EE_BOX_CENTER - EE_BOX_HALF
    hi = EE_BOX_CENTER + EE_BOX_HALF
    tip_pos = np.asarray(tip_pos, float).reshape(3)
    for i in range(3):
        if tip_pos[i] >= hi[i] and v6[3 + i] > 0.0:
            v6[3 + i] = 0.0
        elif tip_pos[i] <= lo[i] and v6[3 + i] < 0.0:
            v6[3 + i] = 0.0
    return v6
# Basic PID for the point-to-point home move: P drives to the target, I removes the
# last steady-state error, D damps. The wiggle was underdamped integral overshoot, so
# I is kept modest and D is raised to settle it — se3_control's anti-windup clamp caps
# the integral so it can't wind up. GOHOME runs on clean encoder FK (not the noisy
# monocular pose), so the derivative damping is safe. Speed capped lower to ease in.
HOME_KP_POS, HOME_KP_ROT = 2.0, 0.5     # P (angular halved to curb overshoot returning to start)
HOME_KI_POS, HOME_KI_ROT = 0.15, 0.03   # I (angular integral cut — the main angular overshoot source)
HOME_KD_POS, HOME_KD_ROT = 0.12, 0.20   # D (more angular damping so the rotation settles, no oscillation)
HOME_MAX_LIN = 0.20                      # slower linear cap for the home move (was 0.30)
HOME_MAX_ANG = 0.40                      # rad/s cap on the home ROTATION (was uncapped at MAX_ANG 1.0)
                                         # — eases the angles in so they don't overshoot the start pose

def go_home(next_state=STATE_IDLE, gripper=None):
    """Retract the tip to HOME_POSE_T with the existing SE(3) PID, then advance to
    next_state. Stops the orientation worker first so its 100 Hz twist can't fight
    the servo, and resets the servo integrator + stall watchdog for a clean move."""
    global state, _gohome_next, HOME_GRIPPER, _gohome_settle
    _stop_orient_thread()
    _orient_stop.clear()
    _reset_stall(); _reset_se3_pid()
    if gripper is not None:
        HOME_GRIPPER = gripper
    _gohome_next = next_state
    _gohome_settle = 0
    state = STATE_GOHOME
    print(f"[→ GOHOME] driving to start pose {_fmt(HOME_POSE_T)}, then → {next_state}")

def start_orient():
    """Enter the pen-driven ORIENT state (WATCH phase). Called automatically as soon
    as PRESENT has leveled (no key needed); the 'o' key also calls it as a fallback.
    The 100 Hz orientation worker is already running from PRESENT."""
    global state, orient_phase, orient_tilt_label, orient_tilt_t, orient_return_t
    global _pen_cand_label, _pen_cand_t, _pen_vhist, _pen_chist, _pen_miss
    global _orient_tgt_R_base, _orient_tgt_t_base, _orient_retreat_from, orient_frozen
    global _orient_tgt_vel, _orient_tgt_prev_t, _orient_tgt_prev_wt
    global _orient_T_EE_obj, _orient_slip_run, _orient_ref_R, _orient_pens_lost_t
    global _orient_frame_count, _pen_oof_frames, _pen_oof_events, _penA_miss_frames
    global _penB_miss_frames, _pen_prev_full, _xint_n, _xint_sum, _xint_sumsq
    global _xint_prev, _xint_path, _fd_list, _pt_list, _csv_event
    global _orient_prev_ang, _orient_prev_ang_t
    global _orient_stall_ref, _orient_stall_rot, _orient_stall_t, _orient_trans_only_until
    global _orient_button_hold_until
    global _orient_return_home, _orient_return_t0
    global _hand_touch_count, _pens_in_view, _hand_last_hull, _hand_hull_stale
    global hand_open_printed, _orient_box_R, _orient_box_t, _orient_hands_cache
    # Seed the smoothed ORIENT box pose from the pose tracked through grasp/lift/present
    # (continuous, so it won't start on a flipped live frame). Live PnP refines it.
    if obj_pose_cam is not None:
        _orient_box_R = obj_pose_cam[:3, :3].copy()
        _orient_box_t = obj_pose_cam[:3, 3].copy()
    else:
        _orient_box_R = _orient_box_t = None
    orient_phase = "WATCH"
    orient_tilt_label = None
    orient_tilt_t = orient_return_t = None
    _pen_cand_label = None
    _pen_cand_t = None
    _pen_vhist = []
    _pen_chist = []
    _pen_miss = 0
    _orient_tgt_R_base = None       # start with no target -> first good frame seeds it
    _orient_tgt_t_base = None
    _orient_tgt_vel = np.zeros(3); _orient_tgt_prev_t = None; _orient_tgt_prev_wt = None
    _orient_retreat_from = None     # no retreat anchor until both pens are first lost
    _orient_pens_lost_t = None      # pens assumed visible on entry (no slow-home pending)
    # Fresh per-trial telemetry accumulators (this ORIENT run == one C2 trial).
    _orient_frame_count = 0
    _pen_oof_frames = 0; _pen_oof_events = 0
    _penA_miss_frames = 0; _penB_miss_frames = 0; _pen_prev_full = True
    _xint_n = 0; _xint_sum = np.zeros(3); _xint_sumsq = np.zeros(3)
    _xint_prev = None; _xint_path = 0.0
    _fd_list = []; _pt_list = []
    _csv_event = "START"            # tag the first ORIENT frame of this trial as the START event
    _orient_T_EE_obj = None         # re-capture the fixed object->EE grip on the first good frame
    _orient_slip_run = 0
    _orient_ref_R = None            # re-capture the level orientation on the first good frame
    _orient_prev_ang = np.zeros(3); _orient_prev_ang_t = None   # fresh slew-limiter state
    _orient_stall_ref = None; _orient_stall_rot = None; _orient_stall_t = None; _orient_trans_only_until = 0.0  # fresh stuck state
    _orient_button_hold_until = 0.0    # no post-button hold pending on a fresh orient
    _orient_return_home = False     # not returning to the initial pose yet
    _orient_return_t0 = 0.0
    orient_frozen = False           # start ORIENT actively tracking (not frozen)
    _orient_hands_cache = []        # fresh hand-detection cache
    _hand_touch_count = 0
    _pens_in_view = True          # assume pens present until a clear tracking frame
    _hand_last_hull = None
    _hand_hull_stale = 0
    hand_open_printed = False
    _hand_relock()                                    # fresh per-hand PnP locks
    _dist_ema.clear(); _infoot_last.clear(); _ema_stale.clear()
    _orient_interrupt.clear()                         # drop any press queued before ORIENT
    _task_finished.clear()                            # drop any STOP queued before ORIENT
    # Continuous 6-DOF pen tracking runs from the MAIN loop via se3_control, so stop the
    # 100 Hz orientation-only worker (it commands the angular channel and would fight the
    # linear channel), halt the arm, and start the SE(3) servo integrator clean.
    _stop_orient_thread()
    _orient_stop.clear()
    _reset_se3_pid()
    send_zeros(GRIPPER_CLOSE)
    state = STATE_ORIENT
    if PENS_AVAILABLE:
        print("[→ ORIENT]  two-pen: the SELECTED FACE point (22 mm above the board) -> the pen-line "
              "intersection, board NEAR-PARALLEL to the pen plane (+15 deg). Show both pens to pose "
              "the object; 'b' toggles freeze; a button press returns to the initial pose.")
    else:
        print("[→ ORIENT]  pens unavailable — board tracking disabled; watching for the "
              "finger-touch release. Lay >=2 fingertips on the board to release.")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
# 2K to match calibrate_camera.py / handeye_calib.py. MJPEG so 2K fits over USB
# at a usable frame rate. This is the ONLY change from the original 720p 2006
# file — every controller/tuning parameter below is identical.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # freshest frame only; avoids control latency
# MOTION-BLUR: a short exposure (fast shutter) is the direct fix for blur while the arm moves —
# blur ~ exposure_time x camera_speed. Lock auto-exposure OFF and set a fast exposure; compensate
# with MORE LIGHT (a short exposure needs a bright scene). On macOS the AVFoundation backend often
# IGNORES these (like focus) — if so, set a fast shutter in the Anker/Camera app instead.
CAMERA_MANUAL_EXPOSURE = True     # try to force a fixed, fast shutter (else leave auto)
CAMERA_EXPOSURE_VALUE  = -7       # smaller/more-negative = faster shutter = less blur (units vary)
if CAMERA_MANUAL_EXPOSURE:
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)     # 0.25=manual on many backends (0.75=auto)
        cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE_VALUE)
        print(f"[camera] tried manual exposure {CAMERA_EXPOSURE_VALUE} "
              f"(read back {cap.get(cv2.CAP_PROP_EXPOSURE)}). If unchanged, set a fast shutter in the Anker app.")
    except Exception as _ee:                          # noqa: BLE001
        print(f"[camera] exposure control not available ({_ee}); set it in the Anker app.")
_aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); _ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened at {_aw}×{_ah}")
if (_aw, _ah) != (2560, 1440):
    print(f"[WARN] Requested 2560×1440 but got {_aw}×{_ah} — pose will be wrong "
          f"unless camera_matrix.npy was calibrated at {_aw}×{_ah}.")

# MediaPipe Tasks HandLandmarker instance for the finger-touch handover release
# (created once at startup, exactly like test2.py). VIDEO running mode so we feed
# frames with monotonically increasing timestamps from _detect_hands.
if MP_AVAILABLE:
    try:
        # Drop a partial/corrupt model left by a previously-failed download
        # (the real asset is ~7 MB; anything much smaller is an error page).
        if os.path.exists(HAND_MODEL_PATH) and os.path.getsize(HAND_MODEL_PATH) < 1_000_000:
            os.remove(HAND_MODEL_PATH)
        if not os.path.exists(HAND_MODEL_PATH):
            print(f"[MediaPipe] downloading hand model -> ./{HAND_MODEL_PATH} ...")
            _download_file(HAND_MODEL_URL, HAND_MODEL_PATH)
            print(f"[MediaPipe] model saved ({os.path.getsize(HAND_MODEL_PATH)//1024} KB).")
        _opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        _hands = vision.HandLandmarker.create_from_options(_opts)
        print("MediaPipe HandLandmarker (Tasks, VIDEO mode, full frame — test2.py "
              "config) ready — hands drawn during ORIENT; lay >=2 fingertips on the "
              "board to release.")
    except Exception as _e:                          # noqa: BLE001
        _hands = None; MP_AVAILABLE = False
        print(f"[WARN] HandLandmarker init failed ({_e}); use 'h' to release.\n"
              f"       Model can be downloaded manually to ./{HAND_MODEL_PATH} from:\n"
              f"       {HAND_MODEL_URL}")
else:
    print("[WARN] MediaPipe not available — finger-touch release off; use 'h' to release.")

# ---------------------------------------------------------------------------
# Operator per-finger hand-size calibration (for the 3D finger-touch release)
# ---------------------------------------------------------------------------
# The 3D touch test recovers each hand's true camera-frame pose by scaling MediaPipe's
# metric hand model to the operator's actual hand, per finger. Measure each finger
# from its BASE KNUCKLE (MCP) to the FINGERTIP, in mm (the knuckle apex sits over the
# MCP joint, which is where MediaPipe's landmark is).
if MP_AVAILABLE:
    print("Measure each finger from its BASE KNUCKLE to the FINGERTIP, in mm "
          "(press Enter to keep the default):")
    for _f in FINGER_ORDER:
        _default = HAND_FINGER_M[_f] * 1000.0
        try:
            _resp = input(f"  {_f:6s} base-knuckle->tip [{_default:.0f}]: ").strip()
            if _resp:
                _val = float(_resp)
                if 20.0 <= _val <= 140.0:
                    HAND_FINGER_M[_f] = _val / 1000.0
                else:
                    print(f"    {_val:.0f} mm out of range (20-140); "
                          f"keeping {_default:.0f} mm.")
        except (EOFError, ValueError):
            print(f"    no/invalid input; keeping {_default:.0f} mm.")
    print("[hand-size] finger base-knuckle->tip (mm): "
          + ", ".join(f"{_f}={HAND_FINGER_M[_f]*1000:.0f}" for _f in FINGER_ORDER))
    print(f"[touch] a fingertip counts as touching within {TOUCH_PLANE_TOL*1000:.0f} mm "
          f"of the object plane.")

print("IDLE — S: go to home pose | Enter: start grasping | r: reset | "
      "g: toggle gripper | Esc: quit")

# ---------------------------------------------------------------------------
# Window placement: keep the operator's approach view on the MAIN (primary) Mac
# screen, so the participant-facing session_runner grid can own the extended
# display. Uses the primary monitor's origin from screeninfo; falls back to (0,0).
# ---------------------------------------------------------------------------
def _place_approach_window(name="approach6dof"):
    px, py, pw, ph = 0, 0, 1920, 1080
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        prim = next((m for m in mons if getattr(m, "is_primary", False)),
                    mons[0] if mons else None)
        if prim is not None:
            px, py, pw, ph = int(prim.x), int(prim.y), int(prim.width), int(prim.height)
    except Exception as e:                                # noqa: BLE001
        print(f"[display] screeninfo unavailable ({e}); placing window at (0,0).")
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)  # force WINDOWED (never fullscreen)
    _w = min(1280, max(640, pw - 120)); _h = min(760, max(480, ph - 160))
    cv2.resizeWindow(name, _w, _h)
    cv2.moveWindow(name, px + 40, py + 60)                # top-left of the laptop/primary screen
    print(f"[display] approach window WINDOWED {_w}x{_h} on primary/laptop at ({px+40},{py+60})")

_place_approach_window()

# ---------------------------------------------------------------------------
# LSL link check — tells you, every run, whether the arm bridge is actually
# connected. z1_pos (arm -> here) and z1_cmd (here -> arm) both live on the C++
# SDK bridge; if z1_pos isn't visible the bridge is down/unreachable and the arm
# will not move (GOHOME will FAULT), no matter what this controller sends.
# ---------------------------------------------------------------------------
def _lsl_link_check():
    print("\n[LSL check] resolving streams (2s) ...")
    try:
        streams = pylsl.resolve_streams(wait_time=2.0)
    except Exception as e:                                    # noqa: BLE001
        print(f"[LSL check] resolve failed: {e}"); streams = []
    names = sorted({s.name() for s in streams})
    print(f"[LSL check] visible LSL streams: {names or 'NONE'}")
    if "z1_pos" in names:
        print("[LSL check]   z1_pos : FOUND — the arm bridge is publishing pose.")
    else:
        print("[LSL check]   z1_pos : *** NOT FOUND *** — the C++ SDK bridge is NOT running or not")
        print("[LSL check]            reachable. The arm cannot move and GOHOME will FAULT. Start the")
        print("[LSL check]            bridge; if it IS running, it's an LSL network/interface problem")
        print("[LSL check]            (see lsl_api.cfg).")
    try:
        has = outlet.wait_for_consumers(2.0)                  # bridge subscribed to z1_cmd?
    except Exception:                                        # noqa: BLE001
        try:
            has = outlet.have_consumers()
        except Exception:                                    # noqa: BLE001
            has = False
    if has:
        print("[LSL check]   z1_cmd : a consumer IS subscribed — the bridge is receiving commands.")
    else:
        print("[LSL check]   z1_cmd : *** NO consumer *** — nothing is receiving z1_cmd (bridge down,")
        print("[LSL check]            or LSL can't cross your network interfaces — see lsl_api.cfg).")
    print("[LSL check] done.\n")

_lsl_link_check()

# ---------------------------------------------------------------------------
# z1_cmd consumer gate. On a RE-RUN this controller makes a NEW z1_cmd outlet,
# but an already-running C++ bridge stays bound to the previous (now-dead) outlet
# and won't auto-reconnect — so nothing receives commands and every move FAULTs.
# Wait here until the bridge subscribes; (re)starting the bridge now makes it
# resolve this new outlet and connect. Ctrl-C skips (e.g. vision-only testing).
# ---------------------------------------------------------------------------
def _wait_for_z1cmd_consumer(timeout=10.0):
    # NON-BLOCKING grace period: the bridge normally subscribes within a couple of seconds of
    # this outlet being created, so give it a short window and then CONTINUE regardless (no need
    # to press anything). "No consumer" for the first second or two is normal, not an error.
    try:
        if outlet.have_consumers():
            print("[z1_cmd] bridge subscribed — good.\n"); return True
    except Exception:                                    # noqa: BLE001
        return True                                      # can't tell -> don't block
    print(f"[z1_cmd] no consumer yet — the bridge usually subscribes within a few seconds "
          f"(or (re)start it now). Waiting up to {timeout:.0f}s, then continuing anyway...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if outlet.wait_for_consumers(1.0):
                print("[z1_cmd] bridge subscribed — good.\n"); return True
        except Exception:                                # noqa: BLE001
            break
    print("[z1_cmd] continuing. If the arm doesn't move, (re)start the bridge; after a GOHOME "
          "FAULT press 'r' to retry.\n")
    return False

_wait_for_z1cmd_consumer()

# ---------------------------------------------------------------------------
# NO-APPROACH BOOT
# ---------------------------------------------------------------------------
# This build does NOT approach or grasp. It drives once to the start ('s') pose with the
# gripper OPEN, then parks in HOLD_WAIT: you place the object in the gripper, press 'g' to
# close, and ORIENT begins when the runner sends START. After each trial it returns here,
# still holding the object, and re-orients on the next START.
state          = STATE_GOHOME
_gohome_next   = STATE_HOLD_WAIT
HOME_GRIPPER   = GRIPPER_OPEN          # home with the gripper OPEN so you can place the object
current_gripper = GRIPPER_OPEN
print("[boot] NO-APPROACH mode: homing (gripper OPEN). Place the object -> press 'g' to close "
      "-> start the trial on the runner (START). 'g' opens again to remove it.")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    _t_frame0 = time.perf_counter()
    ret, frame = cap.read()
    _prof_t_read += time.perf_counter() - _t_frame0
    if not ret:
        break
    _dbg_frame += 1
    frame_clean = frame.copy()   # pristine copy for MediaPipe (no overlays over the hand)

    # PEN DETECTION on the PRISTINE frame — before any board markers/axes/labels
    # are drawn onto it — so pen ArUco recognition is byte-for-byte identical to
    # the standalone pens_mean_tilt_v2.py (drawing on the image before detection
    # was what degraded recognition). Detection uses the pen tracker's own
    # detector (dt.detect) and calibration; only runs while orienting.
    if state == STATE_ORIENT and PENS_AVAILABLE:
        pen_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _t_pen0 = time.perf_counter()
        pen_corners, pen_ids, _ = dt.detect(frame)
        _prof_t_pen += time.perf_counter() - _t_pen0
    else:
        pen_gray = pen_corners = pen_ids = None

    # ONE FK sample per frame: both T_base_cam and T_base_tip come from the
    # same get_T_base_EE() call, so they refer to the same instant.
    T_base_EE  = tm.get_T_base_EE()
    T_base_cam = T_base_EE @ tm.T_EE_cam
    T_base_tip = T_base_EE @ T_EE_tip

    # --- Perception: detect -> single board PnP -> base frame -> filter ----
    # Once ORIENT starts the board is assumed fixed (grip is rigid, control is FK-driven,
    # no slip) until the last button is pressed — so DON'T detect the board at all during
    # ORIENT. Reuse the markers cached at ORIENT entry; the board detect (the biggest
    # per-frame cost) is skipped entirely, leaving only the pen tracking. In every other
    # state the board pose IS the control input, so detect every frame.
    _skip_board = (state == STATE_ORIENT and not ORIENT_DETECT_BOARD
                   and _board_cache is not None)
    if _skip_board:
        corners_list, ids = _board_cache
    else:
        _t_board0 = time.perf_counter()
        corners_list, ids, _ = detector.detectMarkers(frame)
        _prof_t_board += time.perf_counter() - _t_board0
        _board_cache = (corners_list, ids)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners_list, ids)

    R_obj_cam, t_obj_cam, reproj = board_pose(ids, corners_list)

    # Per-marker visualisation (opus style): frame axes + ID at each detected
    # marker, positions derived from the board pose (no redundant per-marker PnP).
    if R_obj_cam is not None and ids is not None:
        rv_board = cv2.Rodrigues(R_obj_cam)[0]
        # Board-frame axes (larger, 30 mm)
        cv2.drawFrameAxes(frame, K, dist, rv_board, t_obj_cam.reshape(3, 1), 0.030)
        # Per-marker axes (15 mm) at each marker's centre
        for mid in [int(m) for m in ids.flatten() if int(m) in MARKER_LAYOUT]:
            t_m = t_obj_cam + R_obj_cam @ MARKER_LAYOUT[mid]
            cv2.drawFrameAxes(frame, K, dist, rv_board, t_m.reshape(3, 1), 0.015)
            px = project_pt(rv_board, t_m)
            put(frame, f"ID{mid}", (px[0] + 5, px[1] - 5), (160, 160, 160), scale=0.40)

    if R_obj_cam is not None:
        T_base_obj = T_base_cam @ make_T(R_obj_cam, t_obj_cam)
        pose_filter.update(T_base_obj[:3, :3], T_base_obj[:3, 3])
    else:
        pose_filter.hold()

    have_obj = pose_filter.valid()

    # Refresh 6-DOF targets while pose is valid and pre-insertion
    if have_obj and state in (STATE_IDLE, STATE_DETECT, STATE_SERVO):
        R_obj_base, t_obj_base = pose_filter.R, pose_filter.t
        pre_grasp_T, grasp_T = compute_grasp(R_obj_base, t_obj_base, T_base_tip)

    e_pos = 0.0
    e_rot = 0.0

    # Refresh the tracked object pose from LIVE markers. The object is rigid to the
    # arm, so its camera-frame pose is ~static; whenever >= OBJ_LIVE_MIN_MARKERS
    # board markers are visible (the 8-marker board keeps >=2 in view while held) we
    # blend the box pose toward the live PnP pose, which corrects grasp-estimate
    # error / slip. Under full occlusion obj_pose_cam simply holds its last value.
    n_board = 0 if ids is None else int(sum(int(m) in MARKER_LAYOUT for m in ids.flatten()))
    if obj_tracked and R_obj_cam is not None and n_board >= OBJ_LIVE_MIN_MARKERS:
        T_live = make_T(R_obj_cam, t_obj_cam)          # live object pose in CAMERA frame
        if obj_pose_cam is None:
            obj_pose_cam = T_live
        else:
            # Reject a flipped live solve (two-fold planar ambiguity): if the live
            # rotation is a big jump from the tracked pose, skip it — blending it in
            # is what throws the box's lines across the frame. Small updates still
            # correct grasp-estimate error / slip as before.
            dR = float(np.linalg.norm(so3_log(obj_pose_cam[:3, :3].T @ T_live[:3, :3])))
            if dR <= OBJ_LIVE_MAX_JUMP:
                a = OBJ_LIVE_ALPHA
                Rp, tp = obj_pose_cam[:3, :3], obj_pose_cam[:3, 3]
                Rn = orthonormalize(Rp @ so3_exp(a * so3_log(Rp.T @ T_live[:3, :3])))
                tn = (1.0 - a) * tp + a * T_live[:3, 3]
                obj_pose_cam = make_T(Rn, tn)

    # Draw the 3D box around the markers and expose its projected front face
    # (obj_quad) + centre (obj_px) for the open-palm release test.
    obj_px = None; obj_quad = None
    if obj_tracked and obj_pose_cam is not None and state != STATE_ORIENT:
        obj_quad, obj_px = _draw_board_box(frame, obj_pose_cam)
        # ORIENT draws its own live, marker-estimated box (see the STATE_ORIENT block).

    # ── State logic ────────────────────────────────────────────────────────
    if state == STATE_IDLE:
        pass

    elif state == STATE_DETECT:
        if have_obj:
            stable_count += 1
            if stable_count >= STABLE_FRAMES and grasp_T is not None:
                stable_count = 0; lock_count = 0
                grasp_committed = False
                _reset_stall()
                _reset_se3_pid()
                state = STATE_SERVO
                # Merged approach+grasp: target the FINAL grasp pose directly (no standoff
                # hover) so the EE heads straight at the object from here.
                e0 = np.linalg.norm(grasp_T[:3, 3] - T_base_tip[:3, 3])
                print(f"[→ SERVO]  target={_fmt(grasp_T)}  "
                      f"tip={_fmt(T_base_tip)}  pos_err={e0*1000:.0f}mm  "
                      f"(continuous approach→grasp)")
        else:
            stable_count = 0

    elif state == STATE_SERVO:
        # ── MERGED APPROACH + GRASP (adapts to a moving object) ──────────────
        # One continuous SE(3) servo straight to the FINAL grasp pose (no standoff hover).
        # PRE-COMMIT: the target grasp pose (position + orientation) is recomputed live from
        # the tracked object every frame, so the arm follows the object as it MOVES. Near
        # contact it COMMITS: it freezes the ACHIEVED tip orientation (not the computed one)
        # and switches to gentle insertion gains — but the target POSITION keeps tracking the
        # live object, so it still adapts to the object moving through the insertion. Freezing
        # the achieved orientation makes the last stretch a pure straight-in translation with
        # ~zero rotation error, which removes the post-commit orientation drift + screw-coupled
        # lateral oscillation (the old stall/FAULT), and stays robust when the eye-in-hand
        # markers leave frame at close range (position then holds its last value).
        target_T = grasp_T_locked if grasp_committed else grasp_T
        if target_T is None:
            send_zeros(GRIPPER_OPEN)
        else:
            if grasp_committed:
                # STRAIGHT-IN insertion: the ORIENTATION is frozen at commit (grasp_T_locked[:3,:3]
                # = the achieved tip orientation), but the POSITION still tracks the live object
                # grasp point (grasp_T is refreshed live while the pose is valid, and holds its last
                # value when markers leave frame). So the arm keeps ADAPTING to a moving object
                # right through the insertion, while the frozen orientation removes the post-commit
                # rotation drift + screw-coupled lateral oscillation that was causing the stall/FAULT.
                _live_pos = grasp_T[:3, 3] if grasp_T is not None else grasp_T_locked[:3, 3]
                insert_target = make_T(grasp_T_locked[:3, :3], _live_pos)
                v, e_pos, e_rot = se3_control(
                    insert_target, T_base_EE, Kp_pos * 0.6, Kp_rot * 0.6,
                    ki_lin=GRASP_KI_POS, kd_lin=GRASP_KD_POS,
                    ki_ang=GRASP_KI_ROT, kd_ang=GRASP_KD_ROT,
                    max_lin=GRASP_MAX_LIN)
            else:
                v, e_pos, e_rot = se3_control(target_T, T_base_EE, Kp_pos, Kp_rot)
                # Commit near contact (close + lined up, or unconditionally within HARD_COMMIT_DIST
                # so the last cm doesn't ride a jittery live orientation). At commit we freeze the
                # ACHIEVED tip orientation (not the computed one): the arm then inserts STRAIGHT in
                # with ~zero rotation error, which is what removes the drift/oscillation. Position
                # keeps tracking the live object (see the committed branch above).
                if e_pos < COMMIT_DIST and e_rot < ROT_THRESH and have_obj:
                    lock_count += 1
                else:
                    lock_count = 0
                if lock_count >= LOCK_FRAMES or e_pos < HARD_COMMIT_DIST:
                    lock_count = 0
                    grasp_committed = True
                    grasp_T_locked  = grasp_T.copy()
                    grasp_T_locked[:3, :3] = T_base_tip[:3, :3].copy()   # FREEZE the achieved tip
                                                                         # orientation -> straight-in
                    grasp_EE_R_lock = T_base_EE[:3, :3].copy()   # drift reference (should stay ~0 now)
                    grasp_tip_lock  = T_base_tip[:3, 3].copy()
                    _grasp_prev_tip = T_base_tip[:3, 3].copy()
                    _grasp_prev_t   = time.time()
                    # Object pose in BASE while markers are still clearly visible.
                    grasp_obj_center_base = (pose_filter.t.copy()
                                             if pose_filter.t is not None else None)
                    grasp_obj_R_base = (pose_filter.R.copy()
                                        if pose_filter.R is not None else None)
                    _reset_stall()
                    _reset_se3_pid()
                    e0 = np.linalg.norm(grasp_T_locked[:3, 3] - T_base_tip[:3, 3])
                    print(f"[COMMIT]  grasp pose frozen — target={_fmt(grasp_T_locked)}  "
                          f"tip={_fmt(T_base_tip)}  pos_err={e0*1000:.0f}mm  "
                          f"(gentle straight-in grasp)")

            # Telemetry: detailed once committed, brief while approaching.
            if _dbg_frame % DEBUG_EVERY == 0:
                if grasp_committed:
                    drift_deg = 0.0
                    if grasp_EE_R_lock is not None:
                        drift_deg = np.rad2deg(np.linalg.norm(
                            so3_log(grasp_EE_R_lock.T @ T_base_EE[:3, :3])))
                    print(f"[GRASP]  pos_err={e_pos*1000:6.1f}mm  "
                          f"rot_err={np.rad2deg(e_rot):4.1f}deg  "
                          f"EE_drift={drift_deg:4.1f}deg  stale={pose_filter.stale}")
                    if drift_deg > 2.0:
                        print("         *** WARNING: EE orientation drifting since "
                              "commit — screw coupling will inject lateral error ***")
                else:
                    print(f"[SERVO]  pos_err={e_pos*1000:.1f}mm  "
                          f"rot_err={np.rad2deg(e_rot):.1f}deg  stale={pose_filter.stale}")

            # Contact → close (STOP_DIST hard-stop on reaching the grasp point).
            if e_pos < STOP_DIST:
                # Freeze the full object POSE in the CAMERA frame at the grasp. It is
                # constant thereafter (object + camera both rigid to the arm), so the
                # 3D box tracks the object through full marker occlusion.
                if grasp_obj_center_base is not None and grasp_obj_R_base is not None:
                    obj_pose_cam = inv_T(T_base_cam) @ make_T(grasp_obj_R_base,
                                                             grasp_obj_center_base)
                    obj_tracked = True
                    print(f"[OBJECT] pose frozen in camera frame (depth "
                          f"{obj_pose_cam[2, 3]*1000:.0f}mm) — 3D box tracked through occlusion.")
                send_zeros(GRIPPER_CLOSE)
                state = STATE_CLOSE_GRIPPER
                close_gripper_t = time.time()
                print("[→ CLOSE_GRIPPER]")
            elif _servo_stalled(e_pos):
                lbl = "GRASP" if grasp_committed else "SERVO"
                print(f"[{lbl}] no progress for {STALL_TIMEOUT:.0f}s "
                      f"(pos_err={e_pos*1000:.0f}mm) — FAULT safe-hold. (r: reset)")
                emergency_stop()
                state = STATE_FAULT
            else:
                send_cmd(v, GRIPPER_OPEN)

    elif state == STATE_CLOSE_GRIPPER:
        send_zeros(GRIPPER_CLOSE)
        if time.time() - close_gripper_t >= 1.0:      # was 1.5 s gripper-close settle
            state = STATE_LIFT
            lift_start_t = time.time(); lift_target_T = None
            print("[→ LIFT] waiting 0.3 s then lifting -10 mm")

    elif state == STATE_LIFT:
        if time.time() - lift_start_t < 0.3:          # pre-lift pause (was 1.0 s)
            send_zeros(GRIPPER_CLOSE)
        else:
            if lift_target_T is None:
                lift_target_T = T_base_tip.copy()
                lift_target_T[2, 3] += -0.01    # lift -10 mm in world Z
                _reset_se3_pid()               # fresh integrator for the lift move
                print(f"[LIFT] target={_fmt(lift_target_T)}")
            v, e_pos, e_rot = se3_control(lift_target_T, T_base_EE, Kp_pos, Kp_rot)
            send_cmd(v, GRIPPER_CLOSE)
            if e_pos < POS_THRESH:
                # PRESENT now first RETURNS the object to the start (home) pose, then levels
                # and hands off to ORIENT. Drive there via GOHOME (gripper stays closed so the
                # object is carried), then enter PRESENT — which starts its own worker.
                print("[→ returning object to start pose, then PRESENT]")
                go_home(next_state=STATE_PRESENT, gripper=GRIPPER_CLOSE)

    elif state == STATE_GOHOME:
        # Point-to-point move to the fixed start/finish pose. Drives on FK alone
        # (no perception), so it works with no markers in view. Uses well-damped,
        # low-integral gains and a lower speed cap so the arm eases in and STOPS
        # instead of overshooting and hunting — passed explicitly, no global state.
        v, e_pos, e_rot = se3_control(
            HOME_POSE_T, T_base_EE, HOME_KP_POS, HOME_KP_ROT,
            ki_lin=HOME_KI_POS, kd_lin=HOME_KD_POS,
            ki_ang=HOME_KI_ROT, kd_ang=HOME_KD_ROT, max_lin=HOME_MAX_LIN)
        v[0:3] = np.clip(v[0:3], -HOME_MAX_ANG, HOME_MAX_ANG)   # ease the rotation in (no angle overshoot)
        if _dbg_frame % DEBUG_EVERY == 0:
            print(f"[GOHOME]  pos_err={e_pos*1000:.1f}mm  "
                  f"rot_err={np.rad2deg(e_rot):.1f}deg  settle={_gohome_settle}")
        if _servo_stalled(e_pos):
            print(f"[GOHOME] no progress for {STALL_TIMEOUT:.0f}s "
                  f"(pos_err={e_pos*1000:.0f}mm) — FAULT safe-hold. (r: reset)")
            emergency_stop(); state = STATE_FAULT
        elif e_pos < HOME_POS_TOL and e_rot < HOME_ROT_TOL:
            # Within tolerance: stop the arm dead and require a few stable frames
            # before committing, so a last flicker can't read as motion.
            send_zeros(HOME_GRIPPER)
            _gohome_settle += 1
            if _gohome_settle >= HOME_SETTLE_FRAMES:
                print(f"[GOHOME] arrived at start pose — → {_gohome_next}")
                state = _gohome_next
        else:
            _gohome_settle = 0
            send_cmd(v, HOME_GRIPPER)

    elif state == STATE_PRESENT:
        # First PRESENT frame (arrived from GOHOME at the start pose): start the orientation
        # worker so it holds level + reports rpy. go_home stopped/joined the worker (thread=None),
        # so a None thread here means we just entered PRESENT and must (re)start it.
        if _orient_thread is None:
            present_leveled = False
            _present_t = time.time()
            _orient_stop.clear()
            _latest_rpy = _erot_prev = _erot_prev_t = None
            _erot_int = np.zeros(3)
            set_orient_target(0.0, 0.0, 0.0, Kp_present, Ki_present, Kd_present)
            _orient_thread = threading.Thread(target=_orientation_worker, daemon=True)
            _orient_thread.start()
        # Orientation thread holds level at 100 Hz; main loop reads its status.
        # Once leveled, arm the manual 'o' trigger and WAIT (do not auto-HOLD).
        with _orient_lock:
            rpy = _latest_rpy
        if rpy is not None:
            roll, pitch, yaw = rpy
            if _dbg_frame % DEBUG_EVERY == 0:
                print(f"[PRESENT]  roll={np.rad2deg(roll):.1f}°  "
                      f"pitch={np.rad2deg(pitch):.1f}°  yaw={np.rad2deg(yaw):.1f}°")
            level_now = (abs(roll) < ROT_THRESH and abs(pitch) < ROT_THRESH
                         and abs(yaw) < ROT_THRESH)
            timed_out = (_present_t is not None
                         and time.time() - _present_t > PRESENT_LEVEL_TIMEOUT)
            if not present_leveled and (level_now or timed_out):
                present_leveled = True
                if level_now:
                    print(f"[PRESENT] leveled (r={np.rad2deg(roll):.2f}° "
                          f"p={np.rad2deg(pitch):.2f}° y={np.rad2deg(yaw):.2f}°) — "
                          f"waiting for logger START (LSL) to begin ORIENT ('o' to override).")
                else:
                    # Couldn't fully level — almost always a wrist joint hard limit.
                    # Don't stall the experiment: continue with whatever level we reached.
                    print(f"[PRESENT] could not fully level in {PRESENT_LEVEL_TIMEOUT:.0f}s "
                          f"(r={np.rad2deg(roll):.1f}° p={np.rad2deg(pitch):.1f}° "
                          f"y={np.rad2deg(yaw):.1f}°) — likely a joint limit; waiting for "
                          f"logger START (LSL) to begin ORIENT ('o' to override).")
            # Enter ORIENT ONLY when the logger's START marker has arrived (and we're leveled),
            # AFTER a START_ORIENT_DELAY_S wait so we don't move during the runner's 3 s countdown.
            # 'o' remains a manual override in the key handler below.
            if present_leveled and _start_requested.is_set():
                if _start_orient_at is None:
                    _start_orient_at = time.time() + START_ORIENT_DELAY_S
                    print(f"[PRESENT] START received from logger — entering ORIENT in "
                          f"{START_ORIENT_DELAY_S:.1f}s (runner countdown).")
                elif time.time() >= _start_orient_at:
                    _start_requested.clear()
                    _start_orient_at = None
                    print("[PRESENT] START delay elapsed — entering ORIENT.")
                    start_orient()

    elif state == STATE_HOLD_WAIT:
        # BETWEEN TRIALS. The object stays GRIPPED and the arm holds at the start ('s') pose.
        # No finger-touch release, no hand-tracking. When the runner broadcasts the next trial's
        # START, re-enter ORIENT after the same countdown (the grip is re-captured on ORIENT entry,
        # and the new Williams face order was set by the START marker in the listener).
        send_cmd(np.zeros(6), current_gripper)        # hold pose; 'g' can open it to remove the object
        # Only begin ORIENT if the gripper is CLOSED (object actually held). If START arrives with
        # the gripper open, ignore it and prompt to place + 'g' first.
        if _start_requested.is_set() and current_gripper == GRIPPER_OPEN:
            _start_requested.clear(); _start_orient_at = None
            print("[HOLD] START ignored — gripper is OPEN. Place the object and press 'g' first.")
        elif _start_requested.is_set():
            if _start_orient_at is None:
                _start_orient_at = time.time() + START_ORIENT_DELAY_S
                print(f"[HOLD] next-trial START received — re-entering ORIENT in "
                      f"{START_ORIENT_DELAY_S:.1f}s (runner countdown).")
            elif time.time() >= _start_orient_at:
                _start_requested.clear()
                _start_orient_at = None
                print("[HOLD] START delay elapsed — re-entering ORIENT (object still held).")
                start_orient()
        if current_gripper == GRIPPER_OPEN:
            put(frame, "PLACE object in gripper, press 'g' to close — then start trial (START)",
                (10, 28), (0, 220, 90))
        else:
            put(frame, "HOLDING object at start pose — waiting for next trial (START)",
                (10, 28), (0, 200, 255))

    elif state == STATE_ORIENT:
        # PEN-DRIVEN ORIENT. The 100 Hz PD thread (already running from PRESENT)
        # tracks the shared setpoint; here we derive that setpoint from the two
        # DodecaPens instead of a scripted sweep, in three phases:
        #   WATCH  — track both pens, average their long-axis directions, classify
        #            the tilt they point to. Once ONE non-level tilt holds steady
        #            (pens physically still AND same label) for ORIENT_STABLE_T,
        #            commit it.
        #   TILT   — command that tilt (±30° pitch or roll) for ORIENT_HOLD_T s.
        #   RETURN — go back to level (the initial position), then resume WATCH.
        #
        # TRIAL FINISHED (grid 'STOP' on GridControl) -> return to the home ('s') pose,
        # keeping the object GRIPPED. This build does NOT release / hand-track: it parks at home
        # holding the object and waits for the runner's next START to re-enter ORIENT.
        if _task_finished.is_set():
            _task_finished.clear()
            # C2 TRIAL SUMMARY: emit ONE consolidated marker (intersection-motion jitter, pen
            # out-of-frame counts, functional-delay/present-time) so the whole trial is recoverable
            # from a single line in the XDF, in addition to the per-frame OrientTelemetry stream.
            _emit_marker(_orient_trial_summary())
            # STOP event row in the CSV (marks the trial end for completion-time in the flat file).
            if ORIENT_CSV_LOG:
                _tp6 = _pose6(T_base_tip)
                _bd6 = _pose6((T_base_EE @ _orient_T_EE_obj) if _orient_T_EE_obj is not None else None)
                _orient_csv_write({
                    "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "t_lsl": f"{pylsl.local_clock():.6f}", "event": "STOP",
                    "participant": _trial_part, "trial": _trial_id, "cond": _trial_cond,
                    "block": _trial_block,
                    "tip_x": _tp6[0], "tip_y": _tp6[1], "tip_z": _tp6[2],
                    "tip_roll": _tp6[3], "tip_pitch": _tp6[4], "tip_yaw": _tp6[5],
                    "board_x": _bd6[0], "board_y": _bd6[1], "board_z": _bd6[2],
                    "board_roll": _bd6[3], "board_pitch": _bd6[4], "board_yaw": _bd6[5]})
            _hand_touch_count = 0
            _start_requested.clear()          # ignore any stale START; wait for the NEXT trial's START
            _start_orient_at = None
            current_gripper = GRIPPER_CLOSE    # keep holding the object across the gap
            # Drive fully back to the start pose with the normal GOHOME servo (all 6 DOF), then hold
            # there (gripper CLOSED) in STATE_HOLD_WAIT until the next trial's START arrives.
            go_home(next_state=STATE_HOLD_WAIT, gripper=GRIPPER_CLOSE)
            print("[ORIENT] trial finished (STOP) -> GOHOME to start pose, HOLD (no release), "
                  "wait for next START.")
            cv2.imshow("approach6dof", frame); cv2.waitKey(1)
            continue

        # END-STATE (ORIENT): 3D FINGER-TOUCH release. Detect + DRAW the hand, recover
        # its true camera-frame pose from the operator's measured hand length (see
        # _hand_tip_positions_cam), and test each fingertip's 3D distance to the board
        # plane. Release when any SINGLE hand has >= TOUCH_FINGERS_MIN fingertips
        # actually touching the board for HAND_CONFIRM frames -> STATE_HANDING (open
        # the gripper). The person lays >=2 fingers on the presented board to receive
        # it (their pens are set down by then).
        square = _board_square(ids, corners_list)            # reference outline only
        if square is not None:
            cv2.polylines(frame, [square], True, (0, 165, 255), 2)   # object square
        # STABLE object box + pose. Update the heavily-smoothed ORIENT pose from the
        # markers seen this frame (board_pose gives a full pose from any subset via the
        # known layout), then use it for BOTH the drawn box and the touch test. This is
        # what keeps the box put when a marker drops and lets it work down to a single
        # marker; only under total loss (and no seed) does bp fall back to the raw pose.
        _orient_update_box(R_obj_cam, t_obj_cam)
        if _orient_box_R is not None:
            bp = (_orient_box_R, _orient_box_t)
        else:
            bp = _board_pose_cam(R_obj_cam, t_obj_cam)
        if bp is not None:
            _draw_box_from_pose(frame, bp[0], bp[1],
                                MARK_HALF_X, MARK_HALF_Y, BOARD_BOX_THICK / 2.0)
        # RIGID GRIP: capture the object->EE transform ONCE from the (smoothed) board pose, then drive
        # the servo from FK + this fixed grip — the object can't move relative to the EE, so per-frame
        # ArUco jitter is kept out of the control loop. bp stays only for the drawn box + slip watchdog.
        if _orient_T_EE_obj is None and bp is not None:
            _orient_T_EE_obj = inv_T(T_base_EE) @ (T_base_cam @ make_T(bp[0], bp[1]))
            print("[ORIENT] captured fixed object->EE grip (servo now FK-driven, vision decoupled)")
        # SLIP WATCHDOG (warning only — never perturbs the command): compare the FK-predicted board
        # pose with the live vision pose; a sustained divergence means the object shifted in the jaws.
        if _orient_T_EE_obj is not None and bp is not None:
            _obj_fk  = T_base_EE @ _orient_T_EE_obj
            _obj_vis = T_base_cam @ make_T(bp[0], bp[1])
            _slip_dp = float(np.linalg.norm(_obj_fk[:3, 3] - _obj_vis[:3, 3]))
            _slip_dr = float(np.rad2deg(np.linalg.norm(so3_log(
                orthonormalize(_obj_fk[:3, :3]).T @ orthonormalize(_obj_vis[:3, :3])))))
            if _slip_dp > ORIENT_SLIP_POS or _slip_dr > ORIENT_SLIP_ANG:
                _orient_slip_run += 1
                if _orient_slip_run == ORIENT_SLIP_FRAMES:
                    print(f"[ORIENT] WARNING grip/vision diverged {_slip_dp*1000:.0f}mm/{_slip_dr:.0f}deg"
                          f" — possible slip" + ("; recapturing grip" if ORIENT_SLIP_RECAPTURE else ""))
                    if ORIENT_SLIP_RECAPTURE:
                        _orient_T_EE_obj = inv_T(T_base_EE) @ _obj_vis
                        _orient_slip_run = 0
            else:
                _orient_slip_run = 0
        # Keep ORIENT focused on pen tracking. MediaPipe hand detection is CPU-heavy,
        # so it is deferred until ORIENT finishes and STATE_TRACK_HANDS takes over.
        hands = []
        _orient_hands_cache = hands
        touch_release = False
        best_touch = 0
        _orient_hand_results = []        # per-hand: (label, smoothed dict, nt)
        for hnd in hands:
            label  = hnd["label"]
            pts_px = hnd["px"]
            raw_info = {}
            if bp is not None:
                pts_cam, _hreproj = _hand_tip_positions_cam(hnd["world"], pts_px, label)
                if pts_cam is not None:
                    _, raw_info = _fingertips_touching(pts_cam, bp[0], bp[1])
            # Per-hand EMA smoothing (keyed by label); HOLD through short dropouts.
            ema  = _dist_ema.setdefault(label, {})
            foot = _infoot_last.setdefault(label, {})
            if raw_info:
                _ema_stale[label] = 0
                for tid in FINGER_TIP_IDS:
                    d = raw_info[tid]["dist_mm"]
                    ema[tid] = (d if tid not in ema
                                else DIST_EMA_ALPHA * d + (1 - DIST_EMA_ALPHA) * ema[tid])
                    foot[tid] = raw_info[tid]["in_foot"]
            else:
                _ema_stale[label] = _ema_stale.get(label, 0) + 1
                if _ema_stale[label] > EMA_MAX_STALE:
                    ema.clear(); foot.clear()
            # Touch from this hand's smoothed distances to the object plane.
            sm = {}; nt = 0
            for tid in FINGER_TIP_IDS:
                if tid in ema:
                    dd = ema[tid]
                    ft = foot.get(tid, False)
                    tc = bool(abs(dd) <= TOUCH_PLANE_TOL * 1000.0 and ft)
                    sm[tid] = {"dist_mm": dd, "in_foot": ft, "touch": tc}
                    nt += int(tc)
            best_touch = max(best_touch, nt)
            if nt >= TOUCH_FINGERS_MIN:
                touch_release = True
            _orient_hand_results.append((label, sm, nt))
            # Fingertip markers (red=touching, green=in-foot, amber=out) + mm labels.
            for tid in FINGER_TIP_IDS:
                s = sm.get(tid)
                col = ((0, 0, 255) if (s and s["touch"]) else
                       ((0, 220, 0) if (s and s["in_foot"]) else (0, 165, 255)))
                cv2.circle(frame, pts_px[tid], 9, col, -1)
                if s is not None:
                    put(frame, f"{s['dist_mm']:+.0f}mm" + ("" if s["in_foot"] else "!"),
                        (pts_px[tid][0] + 12, pts_px[tid][1] + 5), col, 0.45)

        # RELEASE GUARD: the object can be picked up at ANY point during ORIENT — the pen
        # guard is removed, so a finger-touch release fires whenever >=2 fingertips touch the
        # board, even while the pens are still in view. Only a valid board pose is required.
        blocked_reason = ("" if not touch_release else
                          ("no-board" if bp is None else ""))
        touch_release = touch_release and (bp is not None)

        # NOTE: hand detection is intentionally OFF during ORIENT (hands=[] above); MediaPipe
        # only runs in STATE_TRACK_HANDS after the study finishes (last button). No [HANDS]
        # print here so the log doesn't imply detection is happening during orienting.
        board_src = ('live' if (R_obj_cam is not None)
                     else ('tracked' if obj_pose_cam is not None else 'none'))
        status = (f"{best_touch} TIP(S) TOUCHING" if touch_release else
                  (f"HELD — {blocked_reason}" if blocked_reason else
                   f"{best_touch} touching"))
        put(frame,
            f"HANDS:{len(hands)}  board:{board_src}  tol{TOUCH_PLANE_TOL*1000:.0f}mm  {status}",
            (10, 30), (0, 255, 0) if touch_release else
            ((0, 165, 255) if blocked_reason else (200, 200, 200)), 0.7)
        # Per-hand, per-finger distance panel (smoothed).
        _py = 54
        for lbl, sm, nt in _orient_hand_results:
            put(frame, f"{lbl} ({nt} touching)", (10, _py), (0, 220, 220), 0.5); _py += 20
            for tid in FINGER_TIP_IDS:
                s = sm.get(tid)
                if s is None:
                    continue
                col = ((0, 0, 255) if s["touch"] else
                       ((0, 220, 0) if s["in_foot"] else (0, 165, 255)))
                put(frame, f"  {TIP_NAMES[tid]:6s}{s['dist_mm']:+7.1f}mm"
                           f"  [{'in ' if s['in_foot'] else 'OUT'}]"
                           f"{' TOUCH' if s['touch'] else ''}",
                    (10, _py), col, 0.45); _py += 18

        # RELEASE DISABLED DURING ORIENT: the object must stay gripped for the whole experiment.
        # It can ONLY be handed over after the task finishes (grid/logger STOP) — during the return
        # to start and at home (STATE_TRACK_HANDS). Hands are still detected/drawn for feedback.
        _hand_touch_count = 0

        # BUTTON PRESS -> RETURN TO INITIAL POSE. A 'PAIR' marker (the participant pressed the
        # face's button pair) makes the arm drive back to the start pose, then resume tracking
        # for the next face (the face pointer was advanced by the LSL listener on the same PAIR).
        # Any 'b' freeze is cleared; the smoothed pen target is dropped so it re-seeds fresh.
        if _orient_interrupt.is_set():
            _orient_interrupt.clear()
            if orient_frozen:
                orient_frozen = False
            # Brief hold after a button press: freeze the arm for ORIENT_BUTTON_HOLD_T so the
            # user can reposition their hands before the board moves to the next face. The target
            # keeps updating underneath; on release it resumes toward the fresh pose.
            _orient_button_hold_until = time.time() + ORIENT_BUTTON_HOLD_T
            _fd_face = None                 # re-arm fluency timing for the next face (even if same id)
            _orient_stall_ref = None; _orient_trans_only_until = 0.0   # fresh stuck state on resume
            _csv_event = f"PAIR:{target_face}"          # log the press inline in the CSV (for timing)
            print(f"[ORIENT] button press (PAIR) -> hold {ORIENT_BUTTON_HOLD_T:.1f}s "
                  f"(reposition), then resume tracking (next face)")

        # RUNNER FREEZE: 'b' pressed on the session-runner screen -> FREEZE:1/0 over LSL sets the
        # freeze state here directly, so the arm actually stops (works alongside the local 'b' key).
        if _orient_freeze_dirty.is_set():
            _orient_freeze_dirty.clear()
            orient_frozen = _orient_freeze_req.is_set()
            _csv_event = "FREEZE:1" if orient_frozen else "FREEZE:0"   # log freeze toggles inline
            print(f"[ORIENT] runner FREEZE -> {'FROZEN' if orient_frozen else 'resumed tracking'}")
            # FLUENCY: when the participant FREEZES (judges the face ready to work on), log the
            # interval since the robot started moving (participant-perceived readiness latency) and
            # since it settled. Combined with MOVE_START/READY this gives the full timing per face.
            if orient_frozen and _fd_move_start_t is not None:
                _tn = pylsl.local_clock()
                _sm = _tn - _fd_move_start_t
                _sr = (_tn - _fd_ready_t) if _fd_ready_t is not None else float("nan")
                _emit_marker(f"FREEZE_AT:{_fd_face} trial={_trial_id} cond={_trial_cond} "
                             f"since_move={_sm:.3f} since_ready={_sr:.3f}")

        # ── CONTINUOUS PEN-TRACKING SERVO ──────────────────────
        # Track BOTH pens on the pristine frame (pen_gray/pen_corners/pen_ids from the
        # top of the loop). Build each pen's directional line (tip + long axis), find the
        # closest-point "intersection" of the two lines and the plane perpendicular to the
        # pens (normal = mean/bisector of the axes), then drive the grasped board so its
        # CENTRE reaches that point and its plane (board Z normal) aligns to that normal.
        # TWO-PEN: track both pens; the target is the closest-point "intersection" of their two
        # lines, and the plane normal is the mean/bisector of their axes.
        axes = {}
        gap_mm = float("nan")
        if PENS_AVAILABLE:
            for pen in (penA, penB):
                res = pen.track(pen_gray, pen_corners, pen_ids)
                if res is None:
                    continue
                a = np.asarray(res["axis"], float); a /= (np.linalg.norm(a) + 1e-12)
                tip = np.asarray(res["tip"], float).reshape(3)
                axes[pen.name] = (a, tip)
                _pen_draw_vector(frame, tip, a, 0.06, pen.color, pen.name)
                cv2.drawFrameAxes(frame, dt.K, dt.dist, res["rvec"], res["tvec"], 0.02)
        _pens_in_view = len(axes) >= 1

        # --- PEN OUT-OF-FRAME accounting (per C2 trial) + per-frame telemetry defaults ---
        _orient_frame_count += 1
        _seenA = penA is not None and penA.name in axes
        _seenB = penB is not None and penB.name in axes
        _npens = int(_seenA) + int(_seenB)
        if not _seenA:
            _penA_miss_frames += 1
        if not _seenB:
            _penB_miss_frames += 1
        if _npens < 2:
            _pen_oof_frames += 1
            if _pen_prev_full:            # falling edge (both->partial): a NEW dropout episode
                _pen_oof_events += 1
            _pen_prev_full = False
        else:
            _pen_prev_full = True
        # Telemetry values for THIS frame; the branches below fill in the ones they compute.
        _tlm_Xint = np.array([np.nan, np.nan, np.nan]); _tlm_gap = float("nan")
        _tlm_pen_angle = float("nan")
        _tlm_tgt = np.array([np.nan, np.nan, np.nan])
        _tlm_pos_err_mm = float("nan"); _tlm_rot_err_deg = float("nan")
        _csv_func_delay = float("nan"); _csv_present_time = float("nan")   # set on MOVE_START/READY frames
        _tlm_cmd_lin = 0.0; _tlm_cmd_ang = 0.0; _tlm_phase = "watch"       # commanded twist + servo phase

        # TARGET FACE from the runner's LoggerMarkers LSL (set on START, advanced on each correct
        # PAIR). The pens no longer SELECT the face — they only pose it. p_face_local is that face's
        # point (20 mm above the board plane at its grid cell); the board is driven so THAT point
        # reaches the intersection. Unknown face (no START yet) -> board centre (old behaviour).
        with _orient_seq_lock:
            _order = list(_orient_seq_order) if _orient_seq_order else None
            _fidx  = _orient_face_idx
            _explicit = _orient_face_explicit
        if _explicit is not None:
            target_face = _explicit                      # authoritative: exactly what the grid displays
        elif _order:
            target_face = _order[_fidx] if _fidx < len(_order) else _order[-1]
        else:
            target_face = None
        p_face_local = face_point_local(target_face)

        # FLUENCY: a newly-requested face starts a fresh timing episode. _fd_request_t is the
        # moment the robot is free to present this face (functional-delay reference); MOVE_START
        # and READY are timestamped below once the servo actually moves / settles.
        if target_face is not None and target_face != _fd_face:
            _fd_face = target_face
            _fd_request_t = pylsl.local_clock()
            _fd_move_start_t = _fd_ready_t = None
            _fd_move_emitted = _fd_ready_emitted = False

        # SHOW THE TARGET BUTTON on the video: project the selected face's cell centre (on the board
        # surface, the physical button) and its 22 mm aim point through the current VISION board pose,
        # so the operator sees exactly which button the robot is aiming for. Drawn every frame (even
        # frozen / returning) whenever the board and a target face are known.
        if bp is not None and target_face is not None:
            _R_bp, _t_bp = bp[0], bp[1]
            _btn_cam = _R_bp @ np.array([p_face_local[0], p_face_local[1], 0.0]) + _t_bp  # button (surface)
            _aim_cam = _R_bp @ p_face_local + _t_bp                                        # 22 mm aim point
            _hh, _ww = frame.shape[:2]
            def _draw_pt(_pt, _ring, _dot, _col, _label=None):
                if _pt[2] <= 1e-3:
                    return
                _pp = _pen_project([_pt])
                if not np.all(np.isfinite(_pp)):
                    return
                _u, _v = float(_pp[0][0]), float(_pp[0][1])
                if abs(_u) >= 5 * _ww or abs(_v) >= 5 * _hh:
                    return
                _c = (int(_u), int(_v))
                if _ring:
                    cv2.circle(frame, _c, _ring, _col, 2)
                if _dot:
                    cv2.circle(frame, _c, _dot, _col, -1)
                if _label:
                    cv2.putText(frame, _label, (_c[0] + 14, _c[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _col, 2, cv2.LINE_AA)
            _draw_pt(_aim_cam, 6, 0, (0, 165, 255))                              # 22 mm aim point (amber)
            _draw_pt(_btn_cam, 15, 4, (0, 0, 255), f"FACE {target_face}")        # target button (red)

        tgt_ready = False
        if len(axes) == 2 and bp is not None:
            a1, t1 = axes[penA.name]
            a2, t2 = axes[penB.name]
            # 1) closest-point "intersection" of the two pen lines (CAMERA frame) — the point the
            #    SELECTED FACE is driven onto.
            X_cam, gap, f1, f2, _parallel = closest_point_two_lines(t1, a1, t2, a2)
            gap_mm = gap * 1000.0
            # 2) PEN-PLANE normal = cross of the two pen axes (CAMERA frame). The board is driven
            #    PARALLEL to the pens' plane (not perpendicular), then tilted a fixed 15 deg so the
            #    tool can reach the inclined face past the obstruction. |cross| = sin(pen angle) is
            #    ill-conditioned near-parallel; below ORIENT_MIN_PEN_SIN we HOLD the last target and
            #    ask the operator to spread the pens.
            _cx = np.cross(a1, a2)
            _sin_ab = float(np.linalg.norm(_cx))
            # PEN-USAGE telemetry, logged for EVERY 2-pen frame (even near-parallel ones, which the
            # tracking branch below rejects): gap_mm = how far the two pen LINES pass from each other
            # (the intersection sits gap/2 from each line), so gap≈0 means the operator genuinely
            # converges the pens on a point; a large gap = skew / out-of-plane. pen_angle_deg = the
            # acute angle between the pens; small -> near-parallel -> intersection ill-defined.
            _tlm_gap = gap_mm
            _tlm_pen_angle = float(np.degrees(np.arcsin(min(1.0, _sin_ab))))
            if _sin_ab >= ORIENT_MIN_PEN_SIN:
                n_cam = _cx / (_sin_ab + 1e-12)
                # Current board pose (base) + pick the working-face side (min flip) BEFORE the tilt.
                # From FK + the fixed grip (rigid object) so the reference is jitter-free; falls back to
                # vision only if the grip hasn't been captured yet.
                if _orient_T_EE_obj is not None:
                    T_base_obj = T_base_EE @ _orient_T_EE_obj
                else:
                    T_base_obj = T_base_cam @ make_T(bp[0], bp[1])
                R_obj_base = orthonormalize(T_base_obj[:3, :3])
                z_cur = R_obj_base[:, 2]
                n_base = T_base_cam[:3, :3] @ n_cam
                n_base /= (np.linalg.norm(n_base) + 1e-12)
                if np.dot(n_base, z_cur) < 0.0:                # keep the same face
                    n_cam = -n_cam
                # Fixed 15 deg presentation tilt, about the mean pen axis (which lies in the pen
                # plane, perpendicular to n_cam) so the board leans a consistent small amount.
                _tilt_axis = a1 + a2
                _tan = float(np.linalg.norm(_tilt_axis))
                if _tan > 1e-9 and abs(ORIENT_TILT_DEG) > 1e-6:
                    _tilt_axis = _tilt_axis / _tan
                    n_cam = so3_exp(_tilt_axis * np.deg2rad(ORIENT_TILT_DEG)) @ n_cam
                # Visual feedback: board-normal arrow, the board plane, the intersection point.
                _pen_draw_vector(frame, 0.5 * (t1 + t2), n_cam, 0.07, (0, 230, 0), "normal")
                _pen_draw_perp_plane(frame, X_cam, n_cam, 0.03)
                # Draw the intersection point ONLY if it is in front of the camera and projects to a
                # sane pixel (a behind-camera point projects to a huge value OpenCV 4.13 rejects).
                _xpx = _pen_project([X_cam])
                if X_cam[2] > 1e-3 and np.all(np.isfinite(_xpx)):
                    _hh, _ww = frame.shape[:2]
                    _px = _xpx[0]
                    if abs(float(_px[0])) < 5 * _ww and abs(float(_px[1])) < 5 * _hh:
                        _pc = (int(_px[0]), int(_px[1]))
                        cv2.circle(frame, _pc, 7, (0, 255, 0), -1)
                        cv2.circle(frame, _pc, 7, (0, 0, 0), 1)
                # Desired BOARD pose. Orientation = minimal rotation aligning the board normal to the
                # (tilted) n_base — the board never spins about its own normal. Position = place the
                # board so the SELECTED FACE point lands on the intersection: t = X - R * p_face_local.
                n_base = T_base_cam[:3, :3] @ n_cam
                n_base /= (np.linalg.norm(n_base) + 1e-12)
                if _orient_ref_R is None:
                    _orient_ref_R = R_obj_base.copy()   # level reference for the tilt clamp
                if ORIENT_USE_ROTATION:
                    R_des = orthonormalize(rot_between(z_cur, n_base) @ R_obj_base)
                    # Decompose the desired tilt (roll/pitch/yaw) about the level (entry) orientation.
                    # A LOCKED axis is held at its fixed setpoint (the board won't rotate on it); a
                    # free axis tracks the pens, clamped to +/-ORIENT_MAX_TILT_DEG so it can't swing far.
                    _rr, _pp, _yy = R_to_rpy(_orient_ref_R.T @ R_des)
                    _lim = np.deg2rad(ORIENT_MAX_TILT_DEG)
                    _rr = (np.deg2rad(ORIENT_LOCK_ROLL_DEG)  if ORIENT_LOCK_ROLL
                           else float(np.clip(_rr, -_lim, _lim)))
                    _pp = (np.deg2rad(ORIENT_LOCK_PITCH_DEG) if ORIENT_LOCK_PITCH
                           else float(np.clip(_pp, -_lim, _lim)))     # PITCH LOCKED (held level)
                    _yy = float(np.clip(_yy, -_lim, _lim))
                    R_des = orthonormalize(_orient_ref_R @ rpy_to_R(_rr, _pp, _yy))
                else:
                    R_des = R_obj_base          # translation-only: keep the current orientation
                X_base = (T_base_cam @ np.append(X_cam, 1.0))[:3]
                # TELEMETRY + jitter accumulation: this is a VALID intersection (both pens, well-
                # conditioned). Record it for the per-frame stream and the per-trial motion stats
                # (mean, per-axis RMS wander, and total path length = how much the point travelled).
                _tlm_Xint = X_base.copy()          # _tlm_gap / _tlm_pen_angle already set above
                _xint_n += 1
                _xint_sum += X_base
                _xint_sumsq += X_base * X_base
                if _xint_prev is not None:
                    _xint_path += float(np.linalg.norm(X_base - _xint_prev))
                _xint_prev = X_base.copy()
                # SIDE-APPROACH WEDGE: spin the board about its normal so the TARGET face's outward
                # radial turns toward the direction the tools come FROM (pen tips -> intersection),
                # keeping the button reachable only from its +/-ORIENT_APPROACH_HALF_DEG exterior wedge.
                # Only the excess beyond the wedge is corrected (tolerance). Position is unchanged.
                if (ORIENT_APPROACH_WEDGE and ORIENT_USE_ROTATION and target_face is not None
                        and float(np.linalg.norm(p_face_local[:2])) > 1e-6):
                    _zc = R_des[:, 2]                                   # board normal (base)
                    _uo = R_des @ np.array([p_face_local[0], p_face_local[1], 0.0])
                    _uo /= (np.linalg.norm(_uo) + 1e-12)               # target face outward radial (base)
                    _tips_mid = (T_base_cam @ np.append(0.5 * (t1 + t2), 1.0))[:3]
                    _app = _tips_mid - X_base                          # where the tools come FROM
                    _app -= np.dot(_app, _zc) * _zc                    # project into the board plane
                    _na = float(np.linalg.norm(_app))
                    if _na > 1e-6:
                        _app /= _na
                        _theta = float(np.arctan2(np.dot(np.cross(_uo, _app), _zc),
                                                  np.dot(_uo, _app)))   # signed angle: radial -> approach
                        if abs(_theta) > np.deg2rad(ORIENT_APPROACH_HALF_DEG):
                            _phi = _theta - np.sign(_theta) * np.deg2rad(
                                ORIENT_APPROACH_HALF_DEG - ORIENT_APPROACH_MARGIN_DEG)
                            R_des = orthonormalize(R_des @ so3_exp(_phi * np.array([0.0, 0.0, 1.0])))
                # Use the SMOOTHED orientation (not the raw, possibly-jittery R_des) for the face
                # offset, so rotation transients don't yank the POSITION target. The rotation ->
                # position coupling here was the runaway: R_des noise moved center_raw, which the
                # feed-forward amplified into a growing target velocity.
                _R_off = _orient_tgt_R_base if _orient_tgt_R_base is not None else R_des
                center_raw = X_base - _R_off @ p_face_local
                # Constant-pose SE(3) low-pass of the desired board pose; held across dropouts.
                if _orient_tgt_R_base is None:
                    _orient_tgt_R_base = R_des
                    _orient_tgt_t_base = center_raw.copy()
                else:
                    # SPEED-ADAPTIVE blend: alpha rises with the gap to the incoming target, so the
                    # board stays smooth/steady when the pens are still and snaps fluidly to fast pen
                    # motion — no lag, and no waiting for the pens to stabilise.
                    _w_r = so3_log(_orient_tgt_R_base.T @ R_des)                 # rotation to new target
                    _a_r = ORIENT_ALPHA_MIN + (ORIENT_ALPHA_MAX - ORIENT_ALPHA_MIN) * min(
                                1.0, float(np.linalg.norm(_w_r)) / ORIENT_ALPHA_REF_ANG)
                    _d_t = float(np.linalg.norm(center_raw - _orient_tgt_t_base))
                    _a_t = ORIENT_ALPHA_MIN + (ORIENT_ALPHA_MAX - ORIENT_ALPHA_MIN) * min(
                                1.0, _d_t / ORIENT_ALPHA_REF_POS)
                    _orient_tgt_R_base = orthonormalize(_orient_tgt_R_base @ so3_exp(_a_r * _w_r))
                    _orient_tgt_t_base = (1.0 - _a_t) * _orient_tgt_t_base + _a_t * center_raw
                # (2) TARGET VELOCITY for feed-forward: finite-difference the smoothed centre.
                _now_ff = time.time()
                if _orient_tgt_prev_t is not None and _orient_tgt_prev_wt is not None:
                    _dtff = _now_ff - _orient_tgt_prev_wt
                    if _dtff > 1e-3:
                        v_raw = (_orient_tgt_t_base - _orient_tgt_prev_t) / _dtff
                        _vn = float(np.linalg.norm(v_raw))
                        if _vn > ORIENT_FF_MAX:
                            v_raw = v_raw * (ORIENT_FF_MAX / _vn)
                        _orient_tgt_vel = ((1.0 - ORIENT_FF_ALPHA) * _orient_tgt_vel
                                           + ORIENT_FF_ALPHA * v_raw)
                _orient_tgt_prev_t  = _orient_tgt_t_base.copy()
                _orient_tgt_prev_wt = _now_ff
                tgt_ready = True
            else:
                # Pens too parallel -> HOLD in place (seed with the current board pose if we have no
                # target yet, so the arm never backs off just because the plane is ill-defined).
                put(frame, "ORIENT  spread the pens to define the plane", (10, 54), (0, 165, 255))
                if _orient_tgt_R_base is None:
                    if _orient_T_EE_obj is not None:
                        T_base_obj = T_base_EE @ _orient_T_EE_obj
                    else:
                        T_base_obj = T_base_cam @ make_T(bp[0], bp[1])
                    _orient_tgt_R_base = orthonormalize(T_base_obj[:3, :3])
                    _orient_tgt_t_base = T_base_obj[:3, 3].copy()
                tgt_ready = True

        if orient_frozen:
            # Manual hold: stop the arm, keep the gripper closed and keep the last (still-
            # updating) target. Press 'b' again to resume orienting the same face.
            send_cmd(np.zeros(6), GRIPPER_CLOSE)
            _reset_se3_pid()
            _orient_retreat_from = None
            _orient_prev_ang = np.zeros(3); _orient_prev_ang_t = None   # ramp rotation from rest on resume
            _tlm_phase = "frozen"
            put(frame, "ORIENT/FROZEN  press 'b' again to resume (same face)",
                (10, 54), (0, 200, 255))
        elif time.time() < _orient_button_hold_until:
            # POST-BUTTON HOLD: brief pause after a PAIR press so the user can reposition their
            # hands before the board moves to the next face. Arm held still; target keeps updating
            # (above); PID + slew reset so it ramps from rest when the hold expires.
            send_cmd(np.zeros(6), GRIPPER_CLOSE)
            _reset_se3_pid()
            _orient_prev_ang = np.zeros(3); _orient_prev_ang_t = None
            _remain = _orient_button_hold_until - time.time()
            _tlm_phase = "button_hold"
            put(frame, f"ORIENT  button pressed - reposition ({_remain:.1f}s)...",
                (10, 54), (0, 200, 255))
        elif tgt_ready:
            _orient_retreat_from = None    # pen visible -> cancel any back-off search
            _tlm_phase = "tracking"
            if _orient_pens_lost_t is not None:
                # Pens just came back (possibly after a slow-home retract). Clear the loss timer and
                # reset the servo so tracking resumes smoothly instead of lurching from the home move.
                _orient_pens_lost_t = None
                _reset_se3_pid()
                _orient_prev_ang = np.zeros(3); _orient_prev_ang_t = None
            # Board is rigidly held: the object->EE transform is read from this frame's FK
            # and (smoothed) board pose, so the servo self-corrects any grasp slip. Convert
            # the desired BOARD pose to a TIP target and drive the arm with se3_control.
            # Fixed grip (rigid object) -> constant object->EE; avoids feeding board-pose jitter into
            # the desired-EE conversion. Falls back to the live estimate only before capture.
            T_EE_obj = (_orient_T_EE_obj if _orient_T_EE_obj is not None
                        else inv_T(T_base_EE) @ (T_base_cam @ make_T(bp[0], bp[1])))
            T_base_obj_des = make_T(_orient_tgt_R_base, _orient_tgt_t_base)
            T_base_EE_des = T_base_obj_des @ inv_T(T_EE_obj)
            tip_target_T = T_base_EE_des @ T_EE_tip
            # BOUNDARY (not freeze): clamp the desired tip position into the box so the board tracks
            # the pen up to the box wall and slides along it — the arm never leaves the box, but it
            # never freezes either. (box_guard_vel below is the final velocity-level safety.)
            tip_target_T[:3, 3] = clamp_pos_to_box(tip_target_T[:3, 3])
            v, e_pos, e_rot = se3_control(tip_target_T, T_base_EE,
                                          ORIENT_KP_POS, ORIENT_KP_ROT,
                                          max_lin=ORIENT_TRACK_MAXLIN)
            _tlm_pos_err_mm = e_pos * 1000.0                # telemetry: live servo error to the pose
            _tlm_rot_err_deg = float(np.rad2deg(e_rot))
            _tlm_tgt = _orient_tgt_t_base.copy()           # telemetry: smoothed desired board centre
            v[0:3] = np.clip(v[0:3], -ORIENT_TRACK_MAXANG, ORIENT_TRACK_MAXANG)  # cap ORIENT rotation speed
            # STUCK DETECTION: while there is still error to reduce, watch the tip. If it stops making
            # progress, the 6-DOF target is momentarily unreachable -> drop rotation and follow the
            # point with TRANSLATION ONLY for a short window, then retry rotation.
            _now_st = time.time(); _tip_st = T_base_tip[:3, 3]
            _has_work = (e_pos > 0.010) or (e_rot > np.deg2rad(5.0))
            if _has_work:
                if _orient_stall_ref is None:
                    _orient_stall_ref = _tip_st.copy(); _orient_stall_rot = e_rot; _orient_stall_t = _now_st
                else:
                    # PROGRESS = the tip translated OR the orientation error shrank. Including
                    # rotation is essential: a reorient-in-place barely moves the tip, and without
                    # this it would be misread as "stuck" and rotation would be dropped, so the
                    # board could never turn to angle.
                    _moved   = float(np.linalg.norm(_tip_st - _orient_stall_ref)) > ORIENT_STUCK_EPS
                    _rotated = (_orient_stall_rot - e_rot) > ORIENT_STUCK_ROT_EPS
                    if _moved or _rotated:
                        _orient_stall_ref = _tip_st.copy(); _orient_stall_rot = e_rot; _orient_stall_t = _now_st
                    elif (_now_st - _orient_stall_t) > ORIENT_STUCK_T:
                        _orient_trans_only_until = _now_st + ORIENT_STUCK_RECOVER_T      # genuinely stuck
                        _orient_stall_ref = _tip_st.copy(); _orient_stall_rot = e_rot; _orient_stall_t = _now_st
            else:
                _orient_stall_ref = None                                                # converged, not stuck
            _trans_only = _now_st < _orient_trans_only_until
            if (not ORIENT_USE_ROTATION) or _trans_only:
                v[0:3] = 0.0                 # translation-only: follow the point, no rotation command
            # ANGULAR SLEW-RATE LIMIT: cap how fast the angular command may change so the board
            # pitches/rolls in a controllable, non-abrupt way (no sudden rotation on a target jump
            # or pen-tracking spike). Ramps from the previous command by at most accel*dt per frame.
            _now_ang = time.time()
            _dt_ang = (_now_ang - _orient_prev_ang_t) if _orient_prev_ang_t is not None else 0.0
            if ORIENT_ANG_ACCEL_MAX > 0.0 and 0.0 < _dt_ang < 0.5:
                _da_max = ORIENT_ANG_ACCEL_MAX * _dt_ang
                v[0:3] = _orient_prev_ang + np.clip(v[0:3] - _orient_prev_ang, -_da_max, _da_max)
            _orient_prev_ang = np.array(v[0:3], float); _orient_prev_ang_t = _now_ang
            # (2) FEED-FORWARD: add the estimated target-centre velocity (base frame) to the
            # linear command so the arm LEADS a moving pen target instead of lagging it, then
            # re-clamp the total linear speed so FF + feedback stays bounded.
            v[3:6] = v[3:6] + ORIENT_FF_GAIN * _orient_tgt_vel
            _lin_cap = ORIENT_TRACK_MAXLIN + ORIENT_FF_MAX
            v[3:6] = np.clip(v[3:6], -_lin_cap, _lin_cap)
            _v_pre_box = v.copy()                                          # (diag) command before the box guard
            v = box_guard_vel(v, T_base_tip[:3, 3])                       # final safety: EE can't leave the box
            send_cmd(v, GRIPPER_CLOSE)
            _tlm_cmd_lin = float(np.linalg.norm(v[3:6])) * 1000.0          # commanded speed this frame (telemetry)
            _tlm_cmd_ang = float(np.rad2deg(np.linalg.norm(v[0:3])))
            # FLUENCY timestamps for THIS face: when the robot first commands real motion
            # (MOVE_START; func_delay = time since the face was requested), and when it settles
            # at the presentation pose (READY; present_time = settle duration).
            if _fd_face is not None and not _fd_move_emitted and float(np.linalg.norm(v[3:6])) > FD_MOVE_EPS:
                _fd_move_start_t = pylsl.local_clock()
                _fd_move_emitted = True
                _fd_fd = (_fd_move_start_t - _fd_request_t) if _fd_request_t is not None else float("nan")
                if np.isfinite(_fd_fd):
                    _fd_list.append(_fd_fd)                 # collect per-face functional delay for the summary
                _csv_func_delay = _fd_fd                    # write this face's functional delay into the CSV row
                _emit_marker(f"MOVE_START:{_fd_face} trial={_trial_id} cond={_trial_cond} "
                             f"func_delay={_fd_fd:.3f}")
            if (_fd_move_emitted and not _fd_ready_emitted
                    and e_pos < FD_READY_POS and e_rot < FD_READY_ROT):
                _fd_ready_t = pylsl.local_clock()
                _fd_ready_emitted = True
                _fd_pt = (_fd_ready_t - _fd_move_start_t) if _fd_move_start_t is not None else float("nan")
                if np.isfinite(_fd_pt):
                    _pt_list.append(_fd_pt)                 # collect per-face settle time for the summary
                _csv_present_time = _fd_pt                  # write this face's settle time into the CSV row
                _emit_marker(f"READY:{_fd_face} trial={_trial_id} cond={_trial_cond} "
                             f"present_time={_fd_pt:.3f}")
            _face_txt = f"face {target_face}" if target_face is not None else "centre"
            _mode_txt = "  [TRANSLATION-ONLY: stuck, retrying rotation]" if _trans_only else ""
            put(frame, f"ORIENT/TRACK  {_face_txt} -> intersection  (board || pens {ORIENT_TILT_DEG:+.0f}deg){_mode_txt}",
                (10, 54), (0, 165, 255) if _trans_only else (0, 230, 0))
            put(frame, f"pos_err={e_pos*1000:.0f}mm  rot_err={np.rad2deg(e_rot):.1f}deg"
                       f"  pen-gap={gap_mm:.0f}mm  ff={np.linalg.norm(_orient_tgt_vel)*1000:.0f}mm/s",
                (10, 78), (0, 220, 220))
            if _dbg_frame % DEBUG_EVERY == 0:
                # desired board TILT (roll/pitch/yaw) relative to the level entry orientation, so we
                # can see whether the rotation target is sane/settling or running away.
                if _orient_ref_R is not None and _orient_tgt_R_base is not None:
                    _tr, _tp, _ty = R_to_rpy(_orient_ref_R.T @ _orient_tgt_R_base)
                    _tilt_txt = (f"  tilt[r{np.rad2deg(_tr):+.0f} p{np.rad2deg(_tp):+.0f} "
                                 f"y{np.rad2deg(_ty):+.0f}]deg")
                else:
                    _tilt_txt = ""
                # (diag) WHY isn't the arm moving? Show the commanded speed, how much the box guard
                # cut, and whether the bridge is still subscribed to z1_cmd.
                _vlin = float(np.linalg.norm(v[3:6])) * 1000.0
                _vang = float(np.rad2deg(np.linalg.norm(v[0:3])))
                _boxcut = (float(np.linalg.norm(_v_pre_box[3:6]) - np.linalg.norm(v[3:6])) * 1000.0)
                try:
                    _cons = outlet.have_consumers()
                except Exception:                       # noqa: BLE001
                    _cons = "?"
                print(f"[ORIENT/TRACK] pos_err={e_pos*1000:.1f}mm  "
                      f"rot_err={np.rad2deg(e_rot):.1f}deg  pen-gap={gap_mm:.1f}mm  "
                      f"ff={np.linalg.norm(_orient_tgt_vel)*1000:.0f}mm/s{_tilt_txt}  "
                      f"cmd[lin={_vlin:.0f}mm/s ang={_vang:.0f}deg/s box_cut={_boxcut:.0f}mm/s]  "
                      f"rot={'OFF(trans-only:stuck)' if _trans_only else 'on'}  "
                      f"z1_cmd_consumers={_cons}")
        else:
            # BOTH PENS NOT VISIBLE. For the first ORIENT_PEN_LOST_HOME_S seconds, FREEZE in place
            # (hold the last smoothed target) — a brief occlusion shouldn't move the board. If the
            # pens stay lost BEYOND that, ease the tip GENTLY back to the fixed HOME pose (FK-only,
            # no perception) at a slow speed, so the arm doesn't hang awkwardly holding the board.
            # Stays in ORIENT: the instant the pens reappear the tracking branch above resumes.
            if _orient_pens_lost_t is None:
                _orient_pens_lost_t = time.time()
            _lost_dt = time.time() - _orient_pens_lost_t
            if _lost_dt < ORIENT_PEN_LOST_HOME_S:
                # Short dropout -> hold still on the last pose (integrator cleared so resume is clean).
                _reset_se3_pid()
                send_cmd(np.zeros(6), GRIPPER_CLOSE)
                _tlm_phase = "pen_lost_hold"
                put(frame, f"ORIENT/HOLD  pens not visible — holding "
                           f"({ORIENT_PEN_LOST_HOME_S - _lost_dt:.1f}s to home)",
                    (10, 54), (0, 165, 255))
                if _dbg_frame % DEBUG_EVERY == 0:
                    print(f"[ORIENT/HOLD] pens={len(axes)}/2 — frozen, holding last pose "
                          f"({_lost_dt:.1f}s lost)")
            else:
                # Prolonged loss -> SLOW retract to the home pose with the gentle GOHOME gains but a
                # deliberately low speed cap. FK-only target (HOME_POSE_T), so it works with no markers.
                v, e_pos, e_rot = se3_control(
                    HOME_POSE_T, T_base_EE, HOME_KP_POS, HOME_KP_ROT,
                    ki_lin=HOME_KI_POS, kd_lin=HOME_KD_POS,
                    ki_ang=HOME_KI_ROT, kd_ang=HOME_KD_ROT, max_lin=ORIENT_PEN_LOST_HOME_LIN)
                v[0:3] = np.clip(v[0:3], -ORIENT_PEN_LOST_HOME_ANG, ORIENT_PEN_LOST_HOME_ANG)
                v = box_guard_vel(v, T_base_tip[:3, 3])       # stay inside the safety box
                send_cmd(v, GRIPPER_CLOSE)
                _tlm_phase = "slow_home"
                _tlm_cmd_lin = float(np.linalg.norm(v[3:6])) * 1000.0
                _tlm_cmd_ang = float(np.rad2deg(np.linalg.norm(v[0:3])))
                put(frame, f"ORIENT/HOME  pens lost {_lost_dt:.1f}s — slow retract to home "
                           f"(pos {e_pos*1000:.0f}mm)", (10, 54), (0, 200, 255))
                if _dbg_frame % DEBUG_EVERY == 0:
                    print(f"[ORIENT/HOME] pens={len(axes)}/2 lost {_lost_dt:.1f}s — slow home "
                          f"pos_err={e_pos*1000:.0f}mm rot_err={np.rad2deg(e_rot):.1f}deg")

        # --- ONE OrientTelemetry SAMPLE PER ORIENT FRAME (intersection over time, in the XDF) ---
        if orient_tlm is not None:
            try:
                orient_tlm.push_sample([
                    float(_npens), float(_seenA), float(_seenB),
                    float(_tlm_Xint[0]), float(_tlm_Xint[1]), float(_tlm_Xint[2]),
                    float(_tlm_gap),
                    float(_tlm_tgt[0]), float(_tlm_tgt[1]), float(_tlm_tgt[2]),
                    float(_tlm_pos_err_mm), float(_tlm_rot_err_deg),
                    float(_trial_id if _trial_id is not None else -1)])
            except Exception:                                 # noqa: BLE001
                pass

        # --- SAME ROW straight to the CSV (self-contained; no LabRecorder needed) ---
        if ORIENT_CSV_LOG:
            _board_T = (T_base_EE @ _orient_T_EE_obj) if _orient_T_EE_obj is not None else None
            _bp6 = _pose6(_board_T); _tp6 = _pose6(T_base_tip)
            _orient_csv_write({
                "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "t_lsl": f"{pylsl.local_clock():.6f}",
                "event": _csv_event, "participant": _trial_part, "trial": _trial_id,
                "cond": _trial_cond, "block": _trial_block,
                "n_pens": _npens, "penA_seen": int(_seenA), "penB_seen": int(_seenB),
                "Xint_x": _tlm_Xint[0], "Xint_y": _tlm_Xint[1], "Xint_z": _tlm_Xint[2],
                "gap_mm": _tlm_gap, "pen_angle_deg": _tlm_pen_angle,
                "board_x": _bp6[0], "board_y": _bp6[1], "board_z": _bp6[2],
                "board_roll": _bp6[3], "board_pitch": _bp6[4], "board_yaw": _bp6[5],
                "tip_x": _tp6[0], "tip_y": _tp6[1], "tip_z": _tp6[2],
                "tip_roll": _tp6[3], "tip_pitch": _tp6[4], "tip_yaw": _tp6[5],
                "tgt_x": _tlm_tgt[0], "tgt_y": _tlm_tgt[1], "tgt_z": _tlm_tgt[2],
                "pos_err_mm": _tlm_pos_err_mm, "rot_err_deg": _tlm_rot_err_deg,
                "func_delay_s": _csv_func_delay, "present_time_s": _csv_present_time,
                "cmd_lin_mm_s": _tlm_cmd_lin, "cmd_ang_deg_s": _tlm_cmd_ang, "phase": _tlm_phase})
            _csv_event = ""                                   # consumed; clear for the next frame

    elif state == STATE_TRACK_HANDS:
        # STUDY FINISHED: RETURN to the start ('s') pose WHILE hand-tracking. The finger-touch
        # handover is now enabled — open the gripper (release) when >= TOUCH_FINGERS_MIN fingertips
        # touch the board for HAND_CONFIRM frames, whether that happens during the return or once
        # home. Servo toward HOME with the gentle GOHOME gains; it naturally stops and holds on
        # arrival, still tracking hands, until the object is taken.
        v, _ep, _er = se3_control(HOME_POSE_T, T_base_EE, HOME_KP_POS, HOME_KP_ROT,
                                  ki_lin=HOME_KI_POS, kd_lin=HOME_KD_POS,
                                  ki_ang=HOME_KI_ROT, kd_ang=HOME_KD_ROT, max_lin=HOME_MAX_LIN)
        v[0:3] = np.clip(v[0:3], -HOME_MAX_ANG, HOME_MAX_ANG)
        send_cmd(v, GRIPPER_CLOSE)                     # drive back to start, keep gripping the object
        _orient_update_box(R_obj_cam, t_obj_cam)      # smoothed board pose in camera frame
        if _orient_box_R is not None:
            bp = (_orient_box_R, _orient_box_t)
        else:
            bp = _board_pose_cam(R_obj_cam, t_obj_cam)
        if bp is not None:
            _draw_box_from_pose(frame, bp[0], bp[1],
                                MARK_HALF_X, MARK_HALF_Y, BOARD_BOX_THICK / 2.0)
        _t_hands0 = time.perf_counter()
        hands = _detect_hands(frame_clean, frame)     # detect on pristine, draw on display
        _prof_t_hands += time.perf_counter() - _t_hands0
        touch_release = False; best_touch = 0
        for hnd in hands:
            label = hnd["label"]; pts_px = hnd["px"]; raw_info = {}
            if bp is not None:
                pts_cam, _hreproj = _hand_tip_positions_cam(hnd["world"], pts_px, label)
                if pts_cam is not None:
                    _, raw_info = _fingertips_touching(pts_cam, bp[0], bp[1])
            ema  = _dist_ema.setdefault(label, {})
            foot = _infoot_last.setdefault(label, {})
            if raw_info:
                _ema_stale[label] = 0
                for tid in FINGER_TIP_IDS:
                    d = raw_info[tid]["dist_mm"]
                    ema[tid] = (d if tid not in ema
                                else DIST_EMA_ALPHA * d + (1 - DIST_EMA_ALPHA) * ema[tid])
                    foot[tid] = raw_info[tid]["in_foot"]
            else:
                _ema_stale[label] = _ema_stale.get(label, 0) + 1
                if _ema_stale[label] > EMA_MAX_STALE:
                    ema.clear(); foot.clear()
            nt = 0
            for tid in FINGER_TIP_IDS:
                if tid in ema:
                    dd = ema[tid]; ft = foot.get(tid, False)
                    tc = bool(abs(dd) <= TOUCH_PLANE_TOL * 1000.0 and ft)
                    col = ((0, 0, 255) if tc else ((0, 220, 0) if ft else (0, 165, 255)))
                    cv2.circle(frame, pts_px[tid], 9, col, -1)
                    nt += int(tc)
            best_touch = max(best_touch, nt)
            if nt >= TOUCH_FINGERS_MIN:
                touch_release = True
        touch_release = touch_release and bp is not None
        put(frame, f"TRACK HANDS (home) — lay >=2 fingertips on the board to receive"
                   f"   [{best_touch} touching]", (10, 30),
            (0, 255, 0) if touch_release else (200, 200, 200), 0.7)
        _hand_touch_count = _hand_touch_count + 1 if touch_release else 0
        if _hand_touch_count >= HAND_CONFIRM:
            send_cmd(np.zeros(6), GRIPPER_OPEN)
            state = STATE_HANDING
            print(f"[→ HANDING]  {best_touch} fingertips on board at home "
                  f"({HAND_CONFIRM} frames) — releasing.")

    elif state == STATE_HOLD:
        send_zeros(GRIPPER_CLOSE)
        if not hold_printed:
            print("Presenting — holding for handover."); hold_printed = True

    elif state == STATE_HANDING:
        # End state: a finger touched the object -> open the gripper and release.
        # No motion command (twist zero); orientation thread already stopped.
        send_zeros(GRIPPER_OPEN)
        if not hand_open_printed:
            print("[HANDING] gripper OPEN — object released.")
            hand_open_printed = True; _handing_done_t = time.time()
        put(frame, "HANDING — gripper OPEN (released)", (10, 54), (0, 255, 0), 0.8)
        # Once the jaws have opened and the object has cleared, retract to the fixed
        # start/finish pose and park in HOLD_WAIT (NOT IDLE) so the controller stays live and
        # simply waits for the next START — re-place the object + press 'g', then START to run
        # again with no relaunch.
        if _handing_done_t is not None and time.time() - _handing_done_t > 1.0:
            _start_requested.clear(); _start_orient_at = None
            go_home(next_state=STATE_HOLD_WAIT, gripper=GRIPPER_CLOSE)

    elif state == STATE_FAULT:
        # Safe-hold after a stall/abort: keep commanding zero twist and a zero
        # gripper rate so the arm stays halted and the gripper holds. Waits for 'r'.
        send_zeros(GRIPPER_NEUTRAL)
        if not fault_printed:
            print("[FAULT] arm halted (safe-hold). Press 'r' to reset.")
            fault_printed = True
        put(frame, "FAULT — safe-hold (press r)", (10, 54), (0, 0, 255), 0.8)

    # ── FPS / profiling ───────────────────────────────────────────────────────
    # Wall time for the whole frame (from just before cap.read to here). Split into
    # camera-read vs the perception stages so a slow loop is attributable.
    _prof_t_total += time.perf_counter() - _t_frame0
    _prof_n += 1
    _now_loop = time.perf_counter()
    if _loop_t_prev is not None:
        _inst_fps = 1.0 / max(1e-6, _now_loop - _loop_t_prev)
        _fps_ema = _inst_fps if _fps_ema is None else 0.9 * _fps_ema + 0.1 * _inst_fps
    _loop_t_prev = _now_loop
    if _dbg_frame % DEBUG_EVERY == 0 and _prof_n > 0:
        _n = _prof_n
        print(f"[fps] {_prof_n / _prof_t_total:5.1f} fps  |  "
              f"read {1000*_prof_t_read/_n:5.1f}  "
              f"board {1000*_prof_t_board/_n:5.1f}  "
              f"pen {1000*_prof_t_pen/_n:5.1f}  "
              f"hands {1000*_prof_t_hands/_n:5.1f}  "
              f"total {1000*_prof_t_total/_n:5.1f} ms/frame")
        _prof_t_read = _prof_t_board = _prof_t_pen = _prof_t_hands = _prof_t_total = 0.0
        _prof_n = 0

    # ── Overlay ─────────────────────────────────────────────────────────────
    vis_ids = sorted(int(m) for m in (ids.flatten() if ids is not None else [])
                     if int(m) in MARKER_LAYOUT)
    put(frame,
        f"{state}  pos:{e_pos*1000:.1f}mm  rot:{np.rad2deg(e_rot):.1f}deg  "
        f"markers:{vis_ids}  stale:{pose_filter.stale}", (10, 28))
    if _fps_ema is not None:
        put(frame, f"{_fps_ema:4.1f} FPS", (10, 52), (0, 220, 255), scale=0.6)
    cv2.imshow("approach6dof", frame)

    # ── Keys ────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == 27:                                       # Esc
        break
    if key in (ord('s'), ord('S')) and state == STATE_IDLE:   # S: go to home pose, then wait
        go_home(next_state=STATE_IDLE, gripper=GRIPPER_CLOSE)
    # NO-APPROACH build: the Enter->grasp trigger is disabled (no detect/servo/grasp phase).
    # Place the object by hand and press 'g'; ORIENT starts on the runner's START.
    # if key == 13 and state == STATE_IDLE:  (disabled)
    if key == ord('o') and state == STATE_PRESENT and present_leveled:   # manual ORIENT fallback
        start_orient()
    if key == ord('h') and state == STATE_ORIENT:       # manual finger-touch override
        _stop_orient_thread()
        send_cmd(np.zeros(6), GRIPPER_OPEN)
        state = STATE_HANDING
        print("[→ HANDING]  manual release ('h').")
    if key == ord('b') and state == STATE_ORIENT:       # TOGGLE freeze for the SAME face
        orient_frozen = not orient_frozen               # press once -> freeze; again -> resume orienting
        print(f"[ORIENT] 'b' -> {'FROZEN (same face)' if orient_frozen else 'resumed tracking (same face)'}")
    if key == ord('b') and state == STATE_IDLE:         # 'b' at the home/'s' pose -> toggle gripper (like 'g')
        current_gripper = (GRIPPER_CLOSE if current_gripper == GRIPPER_OPEN
                           else GRIPPER_OPEN)
        send_cmd(np.zeros(6), current_gripper)
        print(f"[GRIPPER] 'b' (IDLE) -> {'closed' if current_gripper == GRIPPER_CLOSE else 'opened'}")
    if key == ord('g'):                                 # toggle gripper
        current_gripper = (GRIPPER_CLOSE if current_gripper == GRIPPER_OPEN
                           else GRIPPER_OPEN)
        send_cmd(np.zeros(6), current_gripper)
        print(f"[GRIPPER] {'closed' if current_gripper == GRIPPER_CLOSE else 'opened'}")
    if key == ord('r'):                                 # reset
        reset_all(); print("[→ IDLE] (reset)")

emergency_stop()          # halt the arm and hold the gripper on normal exit (Esc / camera fail)
cap.release()
cv2.destroyAllWindows()
