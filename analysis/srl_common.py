#!/usr/bin/env python3
"""
srl_common.py
=====================================================================
Shared constants + helpers for the SRL study analysis pipeline. Kept in one
place so the synthetic generator and the analysis library agree exactly on the
design (Latin square, Williams sequences, pair<->face wiring, file naming).
"""

from __future__ import annotations
import re
import random
import hashlib

N_FACES     = 8
N_SEQUENCES = 8
CONDITIONS  = ("C0", "C1", "C2")
COUNTERBALANCE_SEED = 20260716

# Physical Arduino pair index (1..8) -> board FACE (1..8). Mirrors logger/session.
PAIR_TO_FACE = {1: 3, 2: 4, 3: 5, 4: 6, 5: 2, 6: 1, 7: 7, 8: 8}
FACE_TO_PAIR = {f: p for p, f in PAIR_TO_FACE.items()}   # face -> Arduino A-column index

# 3x3 Latin square for condition order (participant p -> row (p-1) mod 3).
LATIN_SQUARE = [
    ["C0", "C1", "C2"],
    ["C1", "C2", "C0"],
    ["C2", "C0", "C1"],
]

# Default tolerances (match trial_logger / the plan).
POS_TOL_M   = 0.015
ROT_TOL_DEG = 10.0
SYNC_THRESH_MS = 100.0        # "co-activation" threshold for synchronicity
SETTLE_ROT_DEG = 2.0          # rot_err below this = "arm settled" (functional delay)


def williams_sequences(n=N_FACES):
    """The n balanced Williams orderings (same construction as the study code)."""
    starter, lo, hi, take_lo = [0], 1, n - 1, True
    while len(starter) < n:
        if take_lo:
            starter.append(lo); lo += 1
        else:
            starter.append(hi); hi -= 1
        take_lo = not take_lo
    return [[(starter[j] + i) % n + 1 for j in range(n)] for i in range(n)]


SEQUENCES = williams_sequences(N_FACES)


def assignment_order():
    order = list(range(1, N_SEQUENCES + 1))
    random.Random(COUNTERBALANCE_SEED).shuffle(order)
    return order


def assign_sequence(participant):
    order = assignment_order()
    s = str(participant).strip()
    p = int(s) if s.isdigit() else int(hashlib.sha1(s.encode()).hexdigest(), 16)
    return order[(p - 1) % N_SEQUENCES]


def _pnum(participant):
    s = str(participant).strip()
    return int(s) if s.isdigit() else int(hashlib.sha1(s.encode()).hexdigest(), 16)


def condition_order(participant):
    """Participant's base (block-1) condition order."""
    return LATIN_SQUARE[(_pnum(participant) - 1) % len(LATIN_SQUARE)]


def block_order(participant, block):
    """This block's condition order = participant's Latin row rotated by (block-1),
    so each condition appears in every position once per participant (balanced
    within participant), and each block is balanced across participants."""
    return LATIN_SQUARE[(_pnum(participant) - 1 + (block - 1)) % len(LATIN_SQUARE)]


def trial_to_condition_block(participant, trial):
    """trial 1-9 -> (condition, block, position). Each block runs all 3 conditions
    and the order ROTATES per block (repeated Latin square): trials 1-3 = block 1,
    4-6 = block 2, 7-9 = block 3, with block 2/3 orders rotated by 1/2."""
    block = (trial - 1) // 3 + 1
    position = (trial - 1) % 3 + 1
    return block_order(participant, block)[position - 1], block, position


def pid(participant):
    s = str(participant).strip()
    return f"{int(s):02d}" if s.isdigit() else s


def norm_pid(participant):
    """Canonical participant id for joins/grouping: numeric -> unpadded string
    ('01' and '1' both -> '1'), so filename ids and questionnaire ids match."""
    s = str(participant).strip()
    return str(int(s)) if s.isdigit() else s


# Filenames written by session_runner.py / trial_logger.py:
#   P<pid>_T<trial>_<cond>_b<block>_seq<seq>_<ts>_<kind>.csv
STEM_RE = re.compile(
    r"P(?P<pid>[^_]+)_T(?P<trial>\d+)_(?P<cond>C[012])_b(?P<block>\d+)"
    r"_seq(?P<seq>\d+)_(?P<ts>\d{8}_\d{6})_(?P<kind>buttons|grid|events|robot)\.csv$")


def parse_stem(filename):
    """Return the metadata dict encoded in a per-trial filename, or None."""
    m = STEM_RE.search(str(filename))
    if not m:
        return None
    d = m.groupdict()
    d["trial"] = int(d["trial"]); d["block"] = int(d["block"]); d["seq"] = int(d["seq"])
    return d
