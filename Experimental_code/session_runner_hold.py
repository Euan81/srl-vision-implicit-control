#!/usr/bin/env python3
"""
session_runner_hold.py  (HOLD variant — pairs with approach6dof_hold.py)
=====================================================================
Merged experiment console = logger3_comb.py + grid_face_sequence.py in ONE
program. Reads the Arduino, drives the face-presentation grid, records the
raw button stream and per-face reaction times, and publishes the LSL markers
the robot controller and LabRecorder expect.

HOLD VARIANT: at the end of a trial this sends ONLY 'STOP' — the paired
approach6dof_hold.py controller then parks at the start pose KEEPING the object
gripped (no release, no hand-tracking) and waits for the NEXT trial's START to
re-enter ORIENT. So the object is grasped once and held across the whole block.
(This runner therefore does NOT send GOHOME_TRACK.)

WHY MERGE
  logger + grid always run together on the control machine and previously
  talked over LSL (LoggerMarkers / GridControl). That handshake was the
  fragile part (start-order races, re-resolve loops). In one process the
  serial thread advances the grid DIRECTLY (no round trip, one clock), while
  still broadcasting START/PAIR/STOP on LSL for the robot + XDF.

WHAT STILL TALKS OVER LSL
  * START:<seq>;P=;TRIAL=;COND=;BLOCK=   (LoggerMarkers)  -> robot enters ORIENT, XDF
  * PAIR:<face>                          (LoggerMarkers)  -> robot resumes from freeze, XDF
  * STOP                                 (GridControl)    -> robot parks at start, HOLDS, waits
  (GOHOME_TRACK is intentionally NOT sent in the HOLD variant.)

FIRMWARE
  Targets button_streamer_v2.ino:  line = "t_ms,B,A1..A8" @ ~200 Hz, 115200 baud.
  (t_ms is the device millis(); used for the raw log + synchronicity.)

OUTPUTS (per participant/trial, under recordings/Participant<P>/)
  * <cond>_b<block>_seq<seq>_<ts>_buttons.csv   raw button stream (CSV, high-rate)
  * <cond>_b<block>_seq<seq>_<ts>_grid.csv      per-face reaction time
  * <cond>_b<block>_seq<seq>_<ts>_events.csv    uniform event log (via trial_logger)

CONTROLS
  Enter ......... start the assigned sequence (also fires START on LSL).
  m ............. drop a timestamped NOTE marker in the event log.
  1-8 ........... (debug, no Arduino) simulate pressing that face's pair.
  f ............. toggle fullscreen (2nd monitor).
  r ............. on the DONE screen: advance to trial n+1 (new files + rotated
                  order, no restart). Mid-trial/waiting: abort back to waiting.
  q / Esc ....... quit and save.
"""

import os
import re
import csv
import sys
import time
import queue
import random
import hashlib
import datetime
import itertools
import threading
import subprocess

import numpy as np
import cv2

# ------------------------------------------------------------------------------------------------
# AUDIO FEEDBACK — distinct cues for a CORRECT vs a WRONG pair press.
# Non-blocking (fire-and-forget), never raises: audio must not stall the grid/serial loop. On
# macOS it plays two different built-in system sounds via `afplay`; elsewhere it falls back to the
# terminal bell (1 beep = correct, 2 = wrong). Swap SOUND_CORRECT / SOUND_WRONG for your own
# .aiff/.wav files, or set SOUND_ENABLED = False to mute.
# ------------------------------------------------------------------------------------------------
SOUND_ENABLED = True
SOUND_CORRECT = "/System/Library/Sounds/Glass.aiff"   # bright ding
SOUND_WRONG   = "/System/Library/Sounds/Basso.aiff"   # low buzz


def play_cue(kind):
    """kind = 'correct' | 'wrong'. Fire-and-forget audio; safe to call from any thread."""
    if not SOUND_ENABLED:
        return
    try:
        if sys.platform == "darwin":
            path = SOUND_CORRECT if kind == "correct" else SOUND_WRONG
            subprocess.Popen(["afplay", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            for _ in range(1 if kind == "correct" else 2):
                sys.stdout.write("\a"); sys.stdout.flush()
    except Exception:                                  # noqa: BLE001 — audio is best-effort
        pass

# --- optional deps: serial (no Arduino -> keyboard debug), pylsl (no LSL ->
#     robot/XDF sync off but everything else still records) ------------------
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except Exception as _e:                              # noqa: BLE001
    serial = None
    SERIAL_OK = False
    print(f"[WARN] pyserial unavailable ({_e}); running keyboard-only (debug).")

try:
    import pylsl
    LSL_OK = True
except Exception as _e:                              # noqa: BLE001
    pylsl = None
    LSL_OK = False
    print(f"[WARN] pylsl unavailable ({_e}); LSL markers disabled (robot/XDF sync off).")

try:
    from trial_logger import TrialEventLog
    TL_OK = True
except Exception as _e:                              # noqa: BLE001
    TrialEventLog = None
    TL_OK = False
    print(f"[WARN] trial_logger unavailable ({_e}); event log disabled.")

# ===========================================================================
# Configuration
# ===========================================================================
N_PAIRS              = 8
N_FACES              = 8
A_COLS               = [f"A{i+1}" for i in range(N_PAIRS)]

# Serial / firmware (button_streamer_v2.ino).
BAUD                 = 115200
# Raw line has t_ms + B + A1..A8 = 2 + N_PAIRS fields.
N_SERIAL_FIELDS      = 2 + N_PAIRS

# Counterbalancing (unchanged from logger3_comb.py).
N_SEQUENCES          = 8
COUNTERBALANCE_SEED  = 20260716

# Experiment structure.
#   trial  = one Williams face-sequence  = N_FACES (8) button-pair presses.
#   block  = TRIALS_PER_BLOCK (9) trials = 3 trials of EACH condition, INTERLEAVED as three
#            random triplets (each triplet = a random permutation of the 3 conditions) to
#            control for learning.
#   session= N_BLOCKS (3) blocks.  The robot is driven only on C2 trials (LslMarkers.robot_enabled).
N_BLOCKS             = 3
TRIALS_PER_BLOCK     = 9
N_TRIALS             = N_BLOCKS * TRIALS_PER_BLOCK   # 27

# Physical pair (Arduino A-index 1..8) -> board FACE (1..8). Edit if rewired.
PAIR_TO_FACE         = {1: 3, 2: 4, 3: 5, 4: 6, 5: 2, 6: 1, 7: 7, 8: 8}

# LSL stream names (must match the robot controller).
MARKER_STREAM_NAME   = "LoggerMarkers"
GRID_CONTROL_STREAM  = "GridControl"

# Display.
WINDOW_NAME               = "session runner"
WIN_W, WIN_H              = 900, 900
FULLSCREEN_SECOND_MONITOR = True    # auto-fullscreen on the extended display (drag+'f' fallback)
TARGET_MONITOR_INDEX      = 1
FALLBACK_MONITOR_X, FALLBACK_MONITOR_Y = 1920, 0
FALLBACK_MONITOR_W, FALLBACK_MONITOR_H = 1920, 1080
COUNTDOWN_SECONDS         = 3
PAIR_DEBOUNCE_S           = 0.60

LOG_ROOT                  = "recordings"

# Colours (BGR).
COL_BG, COL_CELL, COL_CELL_EDGE = (24, 24, 24), (70, 70, 70), (120, 120, 120)
COL_DONE, COL_HILITE, COL_HILITE_ED = (60, 110, 60), (40, 200, 255), (0, 140, 220)
COL_WRONG, COL_TEXT, COL_MUTED, COL_GO = (60, 60, 235), (235, 235, 235), (150, 150, 150), (60, 220, 90)

NUM_TO_CELL = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 2),
               5: (2, 2), 6: (2, 1), 7: (2, 0), 8: (1, 0)}
CENTER_CELL = (1, 1)


# ===========================================================================
# Williams design (balanced Latin square) — 8 standardised sequences
# ===========================================================================
def williams_sequences(n=N_FACES):
    starter, lo, hi, take_lo = [0], 1, n - 1, True
    while len(starter) < n:
        if take_lo:
            starter.append(lo); lo += 1
        else:
            starter.append(hi); hi -= 1
        take_lo = not take_lo
    return [[(starter[j] + i) % n + 1 for j in range(n)] for i in range(n)]


def _assert_williams(seqs, n=N_FACES):
    full = set(range(1, n + 1))
    for r in seqs:
        assert set(r) == full, "row not a permutation (not Latin)"
    for c in range(n):
        assert set(r[c] for r in seqs) == full, "column not a permutation"
    pair = {}
    for r in seqs:
        for a, b in zip(r, r[1:]):
            pair[(a, b)] = pair.get((a, b), 0) + 1
    off = [v for (a, b), v in pair.items() if a != b]
    assert off and min(off) == max(off) == 1, "not first-order-carryover balanced"


SEQUENCES = williams_sequences(N_FACES)
_assert_williams(SEQUENCES)


def assignment_order():
    order = list(range(1, N_SEQUENCES + 1))
    random.Random(COUNTERBALANCE_SEED).shuffle(order)
    return order


def assign_sequence(participant, trial=1):
    """A DIFFERENT (random-order) Williams face-sequence for each trial within a participant.

    The 8 Williams rows are shuffled with a PER-PARTICIPANT seed and then indexed by trial, so:
      * within a participant every trial gets a different order (no memorisation / anticipation),
      * the order is reproducible (re-running a participant/trial gives the same sequence),
      * different participants get different shuffles (counterbalanced across the sample).
    Trial 9 wraps to reuse trial 1's row (9 trials, 8 rows)."""
    s = str(participant).strip()
    p = int(s) if s.isdigit() else int(hashlib.sha1(s.encode()).hexdigest(), 16)
    ids = list(range(1, N_SEQUENCES + 1))
    random.Random(COUNTERBALANCE_SEED + p).shuffle(ids)   # per-participant random order of the rows
    return ids[(int(trial) - 1) % N_SEQUENCES]


# --- Conditions: INTERLEAVED as a LATIN SQUARE — each triplet is a DIFFERENT order -----
# Conditions are interleaved WITHIN each block to control for learning: every consecutive group
# of 3 trials is one row of a 3x3 LATIN SQUARE, so (a) each condition appears once per triplet and
# 3x per 9-trial block, and (b) the three triplets are GUARANTEED to be DIFFERENT orders (a Latin
# square's rows are distinct and position-balanced — each condition lands in each slot exactly
# once across the block). E.g. a block runs  C0 C1 C2 | C1 C2 C0 | C2 C0 C1 — never the same order
# three times. The base order is rotated per (participant, block) so blocks & participants differ,
# and the cycling direction is flipped on alternate blocks so the two 3x3 Latin squares pair up
# into a Williams-style, first-order-carryover-balanced set across the study. The robot is driven
# ONLY on the C2 trials (scattered through the block; see LslMarkers.robot_enabled).
import itertools as _it
CONDITIONS = ["C0", "C1", "C2"]
_NC = len(CONDITIONS)
_ALL_ORDERS = [list(p) for p in _it.permutations(CONDITIONS)]   # the 3! = 6 possible orders


def _pnum(participant):
    s = str(participant).strip()
    return int(s) if s.isdigit() else int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _latin_square(base, reverse=False):
    """Cyclic Latin square from `base`: _NC distinct rows, each condition once per column
    (position-balanced). `reverse` flips the cycling direction, which reverses the carryover
    pattern — pairing the two directions balances first-order carryover (Williams)."""
    n = len(base)
    if reverse:
        return [[base[(i - j) % n] for j in range(n)] for i in range(n)]
    return [[base[(i + j) % n] for j in range(n)] for i in range(n)]


def block_triplets(participant, block):
    """The three triplet-orders for a 9-trial block = the rows of a Latin square. Distinct +
    position-balanced by construction; base rotated per (participant, block); direction flipped on
    even blocks so the study-level design is Williams-balanced."""
    base = _ALL_ORDERS[(_pnum(participant) + (block - 1)) % len(_ALL_ORDERS)]
    return _latin_square(base, reverse=(block % 2 == 0))


def condition_for(participant, block, position):
    """Condition for trial `position` (1-9) in a block. Triplet = (position-1)//3 selects the Latin
    row; index = (position-1)%3 selects the slot within it."""
    triplet = (position - 1) // _NC
    idx = (position - 1) % _NC
    return block_triplets(participant, block)[triplet][idx]


def block_conditions(participant, block):
    """The full 9-condition sequence for a block (three distinct Latin-square triplets)."""
    return [condition_for(participant, block, p) for p in range(1, TRIALS_PER_BLOCK + 1)]


def _pid(participant):
    """Zero-padded participant id for tidy, sortable filenames (P01, P02, …)."""
    s = str(participant).strip()
    return f"{int(s):02d}" if s.isdigit() else s


# ===========================================================================
# LSL outlets (same streams the robot listens for)
# ===========================================================================
class LslMarkers:
    def __init__(self):
        self.mk = self.ctrl = None
        # ROBOT-ENABLED gate: the robot-directed markers (START/FACE/PAIR/FREEZE/STOP) are only
        # sent to the approach controller when this is True — set per trial to (condition == 'C2').
        # In C0/C1 (no robot) nothing is sent to the controller. The local event log still records
        # everything for every condition, so timing is not lost.
        self.robot_enabled = False
        if not LSL_OK:
            return
        try:
            self.mk = pylsl.StreamOutlet(pylsl.StreamInfo(
                MARKER_STREAM_NAME, "Markers", 1, 0, "string", "session_logger"))
            self.ctrl = pylsl.StreamOutlet(pylsl.StreamInfo(
                GRID_CONTROL_STREAM, "Markers", 1, 0.0, "string", "session_grid"))
            print(f"[LSL] outlets '{MARKER_STREAM_NAME}' + '{GRID_CONTROL_STREAM}' ready.")
        except Exception as e:                       # noqa: BLE001
            print(f"[WARN] LSL outlet init failed ({e}).")

    def _push(self, outlet, s):
        if outlet is not None:
            try:
                outlet.push_sample([s])
            except Exception:
                pass

    def start(self, seq, p, tr, cond, blk):
        if not self.robot_enabled:
            print(f"[LSL] START suppressed (cond={cond} — robot only in C2)"); return
        self._push(self.mk, f"START:{int(seq)};P={p};TRIAL={tr};COND={cond};BLOCK={blk}")
        print(f"[LSL] START:{seq} (P={p} trial={tr} cond={cond} block={blk})")

    def pair(self, face):   self.robot_enabled and self._push(self.mk, f"PAIR:{int(face)}")
    def face(self, n):      self.robot_enabled and self._push(self.mk, f"FACE:{int(n)}")   # face REQUIRED on-screen
    def freeze(self, on):   self.robot_enabled and self._push(self.mk, "FREEZE:1" if on else "FREEZE:0")
    def gohome(self):       pass   # HOLD variant: no GOHOME_TRACK / handover between trials
    def stop(self):         self.robot_enabled and self._push(self.ctrl, "STOP")


def now_clock():
    """LSL clock if available (matches robot/XDF); else wall clock."""
    return pylsl.local_clock() if LSL_OK else time.time()


# ===========================================================================
# Serial reader — raw button stream -> CSV, rising edges -> LSL + UI queue
# ===========================================================================
class ButtonReader(threading.Thread):
    """Reads button_streamer_v2 lines (t_ms,B,A1..A8), writes every sample to a
    CSV, and on each pair's rising edge (B AND A_i) publishes PAIR:<face> and
    enqueues (face, t_device_ms, t_clock) for the UI loop to consume. Runs as a
    daemon so the cv2 UI keeps the main thread (required on macOS)."""

    def __init__(self, port, csv_path, lsl, press_q, event_log=None):
        super().__init__(daemon=True)
        self.port = port
        self.csv_path = csv_path
        self.lsl = lsl
        self.press_q = press_q
        self.event_log = event_log
        self._run = threading.Event(); self._run.set()
        self._f = open(csv_path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["t_clock", "t_device_ms", "B"] + A_COLS)
        self._f.flush()                              # header safe on disk immediately
        self._n = 0                                  # rows since last flush
        self._prev = [False] * N_PAIRS
        self._lock = threading.Lock()                # guards file/log swaps vs. writes

    def stop(self):
        self._run.clear()

    def switch_output(self, csv_path, event_log=None):
        """Rotate to a NEW trial's button CSV + event log WITHOUT touching the
        serial connection or its clock. Called from the UI thread when you
        advance to the next trial; guarded so it can't tear the file out from
        under a write in progress."""
        with self._lock:
            try:
                self._f.flush(); self._f.close()
            except Exception:
                pass
            self.csv_path = csv_path
            self._f = open(csv_path, "w", newline="")
            self._w = csv.writer(self._f)
            self._w.writerow(["t_clock", "t_device_ms", "B"] + A_COLS)
            self._f.flush()
            self._n = 0
            self._prev = [False] * N_PAIRS
            self.event_log = event_log

    def inject(self, face):
        """Debug: simulate a press of `face` (keyboard path, no Arduino)."""
        self.lsl.pair(face)
        if self.event_log:
            self.event_log.event("PAIR", face=face)
        self.press_q.put((face, None, now_clock()))

    def run(self):
        if serial is None:
            return
        try:
            ser = serial.Serial(self.port, BAUD, timeout=1)
        except Exception as e:                       # noqa: BLE001
            print(f"[serial] open failed ({e}); keyboard-only.")
            return
        ser.reset_input_buffer()
        while self._run.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except Exception:                        # noqa: BLE001
                break
            if not line or "," not in line:
                continue
            parts = line.split(",")
            if len(parts) != N_SERIAL_FIELDS:
                continue
            try:
                vals = [int(p) for p in parts]
            except ValueError:
                continue
            t_dev = vals[0]; b = vals[1]; a_vals = vals[2:]
            t_clk = now_clock()
            with self._lock:                         # never write while switch_output swaps the file
                self._w.writerow([f"{t_clk:.6f}", t_dev, b] + a_vals)
                self._n += 1
                if self._n % 200 == 0:               # ~1 s at 200 Hz: crash-safe raw log
                    self._f.flush()
                for i, a in enumerate(a_vals):
                    pressed = bool(b and a)
                    face = PAIR_TO_FACE.get(i + 1, i + 1)
                    if pressed and not self._prev[i]:    # rising edge -> one event/press
                        self.lsl.pair(face)
                        if self.event_log:
                            self.event_log.event("PAIR", face=face, data={"t_device_ms": t_dev})
                        self.press_q.put((face, t_dev, t_clk))
                    self._prev[i] = pressed
        try:
            self._f.flush(); self._f.close()
        except Exception:
            pass

    def close(self):
        try:
            self._f.flush(); self._f.close()
        except Exception:
            pass


# ===========================================================================
# Drawing (from grid_face_sequence.py)
# ===========================================================================
def _cell_rect(row, col, ox, oy, cell, gap):
    x0 = ox + col * (cell + gap); y0 = oy + row * (cell + gap)
    return x0, y0, x0 + cell, y0 + cell


def _text_centered(img, text, cx, cy, scale, color, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    org = (int(cx - tw / 2), int(cy + th / 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_grid(img, active_num=None, done_nums=frozenset(), wrong_num=None):
    h, w = img.shape[:2]
    grid_span = int(min(w, h) * 0.62); gap = int(grid_span * 0.04)
    cell = (grid_span - 2 * gap) // 3
    ox = (w - (3 * cell + 2 * gap)) // 2; oy = int(h * 0.17)
    cell_to_num = {c: n for n, c in NUM_TO_CELL.items()}
    for row in range(3):
        for col in range(3):
            if (row, col) == CENTER_CELL:
                continue
            num = cell_to_num[(row, col)]
            x0, y0, x1, y1 = _cell_rect(row, col, ox, oy, cell, gap)
            if num == wrong_num:
                fill, edge, txt = COL_WRONG, COL_WRONG, (245, 245, 245)
            elif num == active_num:
                fill, edge, txt = COL_HILITE, COL_HILITE_ED, (20, 20, 20)
            elif num in done_nums:
                fill, edge, txt = COL_DONE, COL_CELL_EDGE, COL_TEXT
            else:
                fill, edge, txt = COL_CELL, COL_CELL_EDGE, COL_TEXT
            cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), edge, 2)
            _text_centered(img, str(num), (x0 + x1) // 2, (y0 + y1) // 2, 1.7, txt, 3)


def target_monitor():
    """Pick the EXTENDED display for the participant grid: prefer the first
    non-primary monitor (robust to how screeninfo orders them); fall back to
    TARGET_MONITOR_INDEX, then to the configured offset."""
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        if mons:
            # 1) first monitor NOT flagged primary = the extended screen.
            for m in mons:
                if not getattr(m, "is_primary", False):
                    print(f"[display] grid -> extended monitor at ({int(m.x)},{int(m.y)})")
                    return int(m.x), int(m.y), int(m.width), int(m.height)
            # 2) only one display (or none flagged) -> configured index.
            idx = TARGET_MONITOR_INDEX if TARGET_MONITOR_INDEX < len(mons) else len(mons) - 1
            m = mons[idx]
            return int(m.x), int(m.y), int(m.width), int(m.height)
    except Exception as e:                           # noqa: BLE001
        print(f"[display] screeninfo unavailable ({e}); using fallback offset.")
    return (FALLBACK_MONITOR_X, FALLBACK_MONITOR_Y, FALLBACK_MONITOR_W, FALLBACK_MONITOR_H)


def _primary_height():
    """Height (px) of the primary display, or None if screeninfo is unavailable."""
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        for m in mons:
            if getattr(m, "is_primary", False):
                return int(m.height)
        return int(mons[0].height) if mons else None
    except Exception:                                # noqa: BLE001
        return None


def _to_opencv_xy(gx, gy, gh):
    """Convert screeninfo coords -> the coords OpenCV's macOS moveWindow wants.

    screeninfo (this macOS build) reports display positions with a BOTTOM-LEFT,
    y-up origin, so a monitor placed ABOVE the laptop comes back with a positive
    y (e.g. y=+982). OpenCV's moveWindow, however, positions the window's
    top-left with a y that grows DOWNWARD from the top of the primary screen —
    so an 'above' monitor needs a NEGATIVE y (e.g. -1080). The conversion is
    y_opencv = primary_height - (gy + gh). Verified on the HDMI-above rig:
    screeninfo (-197, 982, 1920x1080), Hp=982 -> moveWindow(-197, -1080). If
    screeninfo isn't available we can't compute Hp, so we pass the raw coords
    through (side-by-side layouts already work with those).
    """
    Hp = _primary_height()
    if Hp is None:
        return gx, gy
    return gx, Hp - (gy + gh)


def place_window():
    """Auto-fullscreen the grid on the extended display, robust to a monitor
    arranged ABOVE the laptop.

    Two fixes over the original:
      1. Coordinate transform (_to_opencv_xy): screeninfo's y and OpenCV's y use
         opposite origins on macOS, so a monitor placed above needs a negative
         y. Without this the move landed at the primary's bottom edge and
         fullscreen snapped back to the laptop.
      2. Move/resize onto the target rect BEFORE showing anything, pumping the
         event loop between steps, so macOS commits the window to the target
         NSScreen before we call setWindowProperty(FULLSCREEN).
    """
    gx, gy, gw, gh = target_monitor()
    mx, my = _to_opencv_xy(gx, gy, gh)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    black = np.zeros((gh, gw, 3), np.uint8)
    # Commit the window to the target display across a few event-loop turns.
    for _ in range(4):
        cv2.resizeWindow(WINDOW_NAME, gw, gh)
        cv2.moveWindow(WINDOW_NAME, mx, my)          # transformed top-left of target
        cv2.imshow(WINDOW_NAME, black)
        cv2.waitKey(40)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(40)
    print(f"[display] auto-fullscreen on extended monitor: screeninfo ({gx},{gy}) "
          f"{gw}x{gh} -> moveWindow({mx},{my})")
    print("[display] if it lands on the wrong screen: press 'f', drag the window "
          "onto the HDMI monitor, press 'f' again.")
    return gw, gh


def write_grid_csv(path, meta, order, rows):
    if not rows:
        return None, None
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "trial", "condition", "block", "sequence", "order",
                    "step", "face", "shown_clock", "pressed_clock",
                    "reaction_time_s", "wrong_presses"])
        order_str = "-".join(str(x) for x in order)
        for r in rows:
            w.writerow([meta["participant"], meta["trial"], meta["condition"], meta["block"],
                        meta["sequence"], order_str, r["step"], r["face"],
                        f"{r['shown']:.4f}", f"{r['pressed']:.4f}", f"{r['rt']:.4f}", r["wrong"]])
    mean_rt = sum(r["rt"] for r in rows) / len(rows)
    print(f"[grid] {len(rows)} steps -> {path} (mean RT {mean_rt:.2f}s)")
    return path, mean_rt


# ===========================================================================
# Prompts
# ===========================================================================
def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(k in p.description for k in ("CH340", "Arduino", "USB")):
            return p.device
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    return ports[int(input("Select port number: "))].device if ports else None


# ===========================================================================
# Main
# ===========================================================================
def run():
    print("╔════════════════════════════════╗")
    print("║   Session Runner (logger+grid) ║")
    print("╚════════════════════════════════╝\n")

    # You enter participant, block (1-3) and trial-within-block (1-9); the code derives EVERYTHING
    # else. The session is 3 blocks of 9 trials; within each block the conditions are INTERLEAVED as
    # three random triplets (each triplet = a random permutation of C0/C1/C2), so every condition
    # appears 3x per block spread across its timeline (controls learning). The robot is driven ONLY
    # on C2 trials. Internally a single global trial index = (block-1)*TRIALS_PER_BLOCK + trial-in-block.
    participant = input("Participant number: ").strip()
    while True:
        raw = input(f"Block number [1-{N_BLOCKS}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= N_BLOCKS:
            block_in = int(raw); break
        print(f"   Enter a whole number 1-{N_BLOCKS}.")
    while True:
        raw = input(f"Trial within block [1-{TRIALS_PER_BLOCK}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= TRIALS_PER_BLOCK:
            trial = (block_in - 1) * TRIALS_PER_BLOCK + int(raw); break
        print(f"   Enter a whole number 1-{TRIALS_PER_BLOCK}.")

    sequence = assign_sequence(participant, trial)   # distinct Williams order per trial (set in load_trial)
    folder = os.path.join(LOG_ROOT, f"Participant{_pid(participant)}")
    os.makedirs(folder, exist_ok=True)

    lsl = LslMarkers()
    press_q = queue.Queue()
    # Serial (skipped gracefully if no Arduino -> keyboard debug still works).
    port = find_port() if SERIAL_OK else None

    # Trial-dependent state — (re)computed by load_trial() each trial so pressing
    # 'r' on the done screen rolls straight into trial n+1 with no program restart.
    block = position = 0
    condition = ""
    meta = {}
    grid_csv = buttons_csv = ""
    event_log = None
    reader = None

    def load_trial(tr, first=False):
        """Configure everything that depends on the trial number: block,
        condition (rotated Latin square), fresh timestamped output files + event
        log, and the button reader's output CSV. On the FIRST trial it creates
        and starts the reader (opens serial once for the whole session); on later
        trials it just rotates the reader's output — the serial thread and its
        clock keep running untouched."""
        nonlocal trial, block, position, condition, meta, sequence
        nonlocal grid_csv, buttons_csv, event_log, reader
        trial     = tr
        block     = (trial - 1) // TRIALS_PER_BLOCK + 1  # block (1-3)
        position  = (trial - 1) % TRIALS_PER_BLOCK + 1   # trial within THIS block (1-9)
        condition = condition_for(participant, block, position)  # interleaved: 3 of each / block
        lsl.robot_enabled = (condition == "C2")           # drive the robot ONLY on C2 trials
        sequence  = assign_sequence(participant, trial)  # distinct Williams order for THIS trial
        meta = {"participant": participant, "trial": trial, "condition": condition,
                "block": block, "sequence": sequence, "position": position}
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Filename carries participant, trial, CONDITION, block, sequence -> self-describing
        # and sortable, so you can retrieve any trial and read its condition from the name.
        stem = f"P{_pid(participant)}_T{trial}_{condition}_b{block}_seq{sequence}_{ts}"
        buttons_csv = os.path.join(folder, f"{stem}_buttons.csv")
        grid_csv    = os.path.join(folder, f"{stem}_grid.csv")

        # Close the previous trial's event log, open a fresh one for this trial.
        if event_log is not None:
            try: event_log.close()
            except Exception: pass
        event_log = TrialEventLog(meta, out_dir=folder) if TL_OK else None

        if first:
            # Always create the reader: it opens the button CSV and provides
            # keyboard inject() even with no Arduino (serial thread just exits).
            reader = ButtonReader(port, buttons_csv, lsl, press_q, event_log)
            reader.start()
        else:
            reader.switch_output(buttons_csv, event_log)   # new files, same serial/clock

        print(f"\n  Participant {participant}  ·  trial {trial}/{N_TRIALS}  ·  block {block}/{N_BLOCKS}")
        _seq = block_conditions(participant, block)        # this block's 9-trial condition order
        print(f"  Block {block} conditions (3 distinct Latin-square triplets): "
              f"{' '.join(_seq[0:3])} | {' '.join(_seq[3:6])} | {' '.join(_seq[6:9])}")
        print(f"  -> trial {position}/{TRIALS_PER_BLOCK} -> condition {condition}"
              f"  (robot {'ON (C2)' if condition == 'C2' else 'OFF'}),  sequence {sequence}")
        print(f"  files: {os.path.basename(buttons_csv)} / {os.path.basename(grid_csv)}")

    load_trial(trial, first=True)
    if input("  Correct? [Enter = yes / n = abort]: ").strip().lower() == "n":
        if reader: reader.stop(); reader.close()
        if event_log: event_log.close()
        return

    global WIN_W, WIN_H
    WIN_W, WIN_H = place_window()
    fullscreen = FULLSCREEN_SECOND_MONITOR

    # Session state.
    phase = "waiting"            # waiting -> countdown -> highlight -> done
    frozen = False               # 'b' toggles the robot freeze (broadcast to the controller via LSL)
    order, idx, done_nums = [], 0, set()
    phase_start = time.time()
    last_advance_t = 0.0
    wrong_face, wrong_t = None, 0.0
    timing_rows, wrong_counts = [], {}
    step_shown_t, done_logged, last_mean_rt = None, False, None

    def begin():
        nonlocal phase, order, idx, done_nums, phase_start, timing_rows
        nonlocal wrong_counts, step_shown_t, done_logged, last_mean_rt
        order = SEQUENCES[sequence - 1]
        idx = 0; done_nums = set(); phase = "countdown"; phase_start = time.time()
        timing_rows = []; wrong_counts = {}; step_shown_t = None
        done_logged = False; last_mean_rt = None
        lsl.start(sequence, participant, trial, condition, block)
        if event_log:
            event_log.event("START", data={"order": order})
        print(f"[SEQ] sequence {sequence}: {order}")

    def to_waiting():
        nonlocal phase, order, idx, done_nums
        phase = "waiting"; order, idx, done_nums = [], 0, set()

    def next_trial():
        """Advance to trial n+1: recompute block/condition, open fresh files and
        rotate the reader, then return to the waiting screen — no program restart.
        Bound to 'r' on the DONE screen. Caps at N_TRIALS (end of the design)."""
        if trial >= N_TRIALS:
            print(f"[session] trial {N_TRIALS}/{N_TRIALS} done — full design complete for this participant.")
            to_waiting(); return
        _next_block = (trial) // TRIALS_PER_BLOCK + 1     # block of the upcoming trial (trial+1)
        if _next_block != block:
            print(f"[session] *** BLOCK {block} COMPLETE -> next is block {_next_block}. "
                  f"Break point: rest + questionnaire, then continue. ***")
        load_trial(trial + 1)
        to_waiting()
        print(f"[session] ready for trial {trial}/{N_TRIALS} — press Enter to start.")

    while True:
        canvas = np.full((WIN_H, WIN_W, 3), COL_BG, dtype=np.uint8)
        now = time.time()

        # Drain presses from the serial thread.
        pressed_face = None
        try:
            while True:
                pressed_face, _tdev, _tclk = press_q.get_nowait()
        except queue.Empty:
            pass

        # AUTO-RESUME: a button press unfreezes the robot automatically (the PAIR marker already
        # resumes the controller), so clear our freeze state + banner without needing another 'b'.
        if pressed_face is not None and frozen:
            frozen = False
            print("[FREEZE] OFF (auto-resumed on button press)")

        if phase == "waiting":
            draw_grid(canvas, None, set())
            _text_centered(canvas, "press Enter to start the assigned sequence",
                           WIN_W // 2, int(WIN_H * 0.09), 0.85, COL_TEXT, 2)
            _text_centered(canvas, f"P{participant} · trial {trial} · {condition} · block {block}"
                           f" · seq {sequence}", WIN_W // 2, int(WIN_H * 0.92), 0.6, COL_MUTED, 2)

        elif phase == "countdown":
            draw_grid(canvas, None, done_nums)
            _text_centered(canvas, f"sequence {sequence}", WIN_W // 2,
                           int(WIN_H * 0.09), 0.8, COL_MUTED, 2)
            remaining = COUNTDOWN_SECONDS - (now - phase_start)
            if remaining > 0:
                _text_centered(canvas, str(int(np.ceil(remaining))), WIN_W // 2,
                               int(WIN_H * 0.955), 1.4, COL_GO, 3)
                _text_centered(canvas, "get ready...", WIN_W // 2, int(WIN_H * 0.90), 0.7, COL_MUTED, 2)
            else:
                _text_centered(canvas, "GO", WIN_W // 2, int(WIN_H * 0.94), 1.4, COL_GO, 3)
                if now - phase_start > COUNTDOWN_SECONDS + 0.6:
                    phase = "highlight"; last_advance_t = now; step_shown_t = now
                    lsl.face(order[idx])           # tell the robot the first required face

        elif phase == "highlight":
            active_num = order[idx]
            if step_shown_t is None:
                step_shown_t = now
            if pressed_face is not None and (now - last_advance_t) >= PAIR_DEBOUNCE_S:
                if pressed_face == active_num:
                    play_cue("correct")            # ding: correct pair
                    timing_rows.append({"step": idx + 1, "face": active_num,
                                        "shown": step_shown_t, "pressed": now,
                                        "rt": now - step_shown_t, "wrong": wrong_counts.get(idx, 0)})
                    done_nums.add(active_num); last_advance_t = now; idx += 1; step_shown_t = now
                    if idx >= N_FACES:
                        phase = "done"; phase_start = now
                    else:
                        lsl.face(order[idx])       # tell the robot the next required face
                else:
                    play_cue("wrong")              # buzz: wrong pair
                    wrong_counts[idx] = wrong_counts.get(idx, 0) + 1
                    wrong_face, wrong_t = pressed_face, now
                    if event_log:
                        event_log.event("WRONG_PRESS", face=pressed_face,
                                        data={"expected": active_num})
            flash = wrong_face if (now - wrong_t) < 0.6 else None
            draw_grid(canvas, active_num, done_nums, wrong_num=flash)
            _text_centered(canvas, f"sequence {sequence}  -  step {idx + 1} of {N_FACES}",
                           WIN_W // 2, int(WIN_H * 0.09), 0.85, COL_TEXT, 2)
            msg = (f"wrong pair ({flash}) - press pair for face {active_num}" if flash
                   else f"present face {active_num}  ->  press its button pair")
            _text_centered(canvas, msg, WIN_W // 2, int(WIN_H * 0.91), 0.8,
                           COL_WRONG if flash else COL_HILITE, 2)

        elif phase == "done":
            if not done_logged:
                _, last_mean_rt = write_grid_csv(grid_csv, meta, order, timing_rows)
                lsl.stop()                       # STOP only: the HOLD controller parks at start,
                                                 # KEEPS the object gripped, and waits for the next
                                                 # trial's START. No GOHOME_TRACK / handover here.
                if event_log:
                    event_log.event("STOP")
                print("[GridControl] STOP sent (robot holds at start, waits for next START).")
                done_logged = True
            draw_grid(canvas, None, done_nums)
            _text_centered(canvas, "TRIAL DONE — robot holding at start", WIN_W // 2,
                           int(WIN_H * 0.09), 1.0, COL_GO, 3)
            if last_mean_rt is not None:
                _text_centered(canvas, f"mean reaction time {last_mean_rt:.2f}s",
                               WIN_W // 2, int(WIN_H * 0.90), 0.6, COL_MUTED, 2)
            nxt = ("session complete" if trial >= N_TRIALS
                   else f"r = start next trial ({trial + 1}/{N_TRIALS})")
            _text_centered(canvas, nxt, WIN_W // 2,
                           int(WIN_H * 0.93), 0.65, COL_MUTED, 2)

        # BIG FREEZE banner under the grid whenever the robot is frozen.
        if frozen:
            cv2.rectangle(canvas, (0, int(WIN_H * 0.80)), (WIN_W, int(WIN_H * 0.93)), (30, 30, 120), -1)
            _text_centered(canvas, "FROZEN", WIN_W // 2, int(WIN_H * 0.865), 2.4, (90, 90, 255), 6)
            _text_centered(canvas, "press 'b' to resume", WIN_W // 2, int(WIN_H * 0.905),
                           0.7, (210, 210, 255), 2)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if key in (ord('r'), ord('R')):
            if phase == "done":
                next_trial()          # roll into trial n+1 (new files, rotated order)
            else:
                to_waiting()          # abort current trial back to the waiting screen
        if key in (ord('b'), ord('B')):                 # toggle robot FREEZE (sent to the controller via LSL)
            frozen = not frozen
            lsl.freeze(frozen)
            if event_log:
                event_log.event("FREEZE", state=phase, data={"on": int(frozen)})
            print(f"[FREEZE] {'ON (robot frozen)' if frozen else 'OFF (robot resumed)'}")
        if key in (ord('m'), ord('M')) and event_log:   # timestamped note marker
            event_log.event("NOTE", state=phase)
            print("[note] marker logged")
        if key in (13, 32) and phase == "waiting":      # Enter/Space -> start
            begin()
        if ord('1') <= key <= ord('8') and phase == "highlight" and reader:
            reader.inject(key - ord('0'))               # debug press

    if reader:
        reader.stop(); reader.close()
    if event_log:
        event_log.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
