# Experimental code

Experiment control and perception for the MSc thesis *Vision-driven
Supernumerary Robotic Limbs: Do many hands make light work?* (Imperial
College London, Bioengineering, 2026).

## Architecture

Two machines. The MacBook runs perception, intent inference and the session
console; the Linux host runs the Z1 SDK. They are linked by LSL, which also
timestamps everything for offline alignment.

| File | Role |
|---|---|
| `orient_experiment.py` | C2 controller: ArUco board and DodecaPen tracking, intersection geometry, board-presentation logic, PID servo. Publishes 6-DoF end-effector velocities over LSL |
| `session_runner_hold.py` | Session console for all three conditions: reads the Arduino button stream, logs presses and per-face reaction times, publishes START/STOP markers, drives the on-screen grid |
| `session_runner_hold_warmup.py` | Warm-up variant: back-to-back C2 trials, `WARMUP` filename tags |
| `transforms.py` | `TransformManager`: frame graph and hand-eye transform |
| `dodecapen_tracker.py` | Multi-marker pen pose from pooled faces |
| `trial_logger.py` | Uniform per-trial event log |

On the Linux host (see `../control/z1_sdk/`):

```bash
z1_simple/build/demo_z1_simple_lsl    # LSL commands -> joint velocities
z1_simple/build/demo_z1_simple_ctrl   # joint velocities -> torques
```

## Calibration artefacts

`orient_experiment.py` loads these from the working directory:

| File | Contents | Produced by |
|---|---|---|
| `camera_matrix.npy` | Camera intrinsics | Planar-target calibration (§3.2.2) |
| `dist_coeffs.npy` | Distortion coefficients | as above |
| `T_EE_cam.npz` | Hand-eye transform | Tsai-Lenz on a ChArUco board (§3.2.2) |

Values used in the study are in `calibration/`. Regenerate for a different
camera or mount.

## Requirements

Python 3.9+ on the control machine.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pylsl` is required for robot control and XDF synchronisation. Without it the
runners still log to CSV but publish no markers. Without `pyserial` the runner
falls back to keyboard-debug mode (`1`-`8` simulate face presses).

On first run, hand tracking downloads a MediaPipe model over HTTPS; if
certificate verification fails it retries with an unverified context.

## Running a session

From the repository root:

```bash
python3 Experimental_code/session_runner_hold_warmup.py   # familiarisation
python3 Experimental_code/session_runner_hold.py          # 27 trials, 3 blocks
```

The runner prompts for participant number, block and trial. Condition order is
Williams-counterbalanced; the eight Williams rows are shuffled with a
per-participant seed, so a given participant number reproduces the same order.

Controls: `Enter` start sequence · `m` timestamped note · `1`-`8` simulate
press · `f` fullscreen · `r` advance on DONE · `q`/`Esc` quit and save.

## Outputs

Written to `recordings/Participant<P>/`:

- `<cond>_b<block>_seq<seq>_<ts>_buttons.csv` - raw 200 Hz button stream
- `<cond>_b<block>_seq<seq>_<ts>_grid.csv` - per-face reaction times
- `<cond>_b<block>_seq<seq>_<ts>_events.csv` - event log

These are the input to `../analysis/run_analysis.py`.

## Hardware

Arduino firmware: `../hardware/button_streamer_v2/button_streamer_v2.ino`.
Outputs `t_ms,B,A1..A8` at ~200 Hz, 115200 baud.