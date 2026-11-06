#!/usr/bin/env python
"""Freeze the settings of the second campaign's new features, by rules fixed before any test run.

  abstain (H3): the clean calibration runs (seeds 0-2, rev2_abstain_cal) log the deviation of the
    abstain statistic in standard deviations on every block while never abstaining. The threshold z
    is the smallest of 3, 4, 5, 6, 8 whose clean abstention rate over blocks 5-99 is at most 2%.
  joint backbone (D8): online DeepSIC with the joint backbone, epochs 30/100/300 x learning rate
    1e-3/5e-3 on clean runs of seeds 0-2 (rev2_joint_tune); the lowest mean clean SER wins (the rule
    used for the DeepSIC baselines). Citadel-Rx keeps its frozen settings on the new backbone.

    python scripts/freeze_rev2.py      # writes configs/revision2_frozen_abstain.json, revision2_frozen_joint.json
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = [0, 1, 2]


def load(folder):
    out = []
    for fn in glob.glob(os.path.join(ROOT, "results", folder, "*.json")):
        try:
            out.append(json.load(open(fn)))
        except Exception:
            pass
    return out


def freeze_abstain() -> None:
    recs = [r for r in load("rev2_abstain_cal") if r["seed"] in CAL]
    if sorted(r["seed"] for r in recs) != CAL:
        print("abstain: calibration runs incomplete", sorted(r["seed"] for r in recs))
        return
    dev = np.array([b["abstain_dev"] for r in recs for b in r["blocks"] if b["block"] >= 5 and "abstain_dev" in b])
    rates = {z: float(np.mean(np.abs(dev) > z)) for z in (3, 4, 5, 6, 8)}
    ok = [z for z, v in rates.items() if v <= 0.02]
    if not ok:
        print("abstain: no threshold meets the 2% rule", rates)
        return
    z = min(ok)
    out = {"rule": "smallest z in {3,4,5,6,8} with clean abstention <= 2% over blocks 5-99 of seeds 0-2",
           "clean_abstention_rate": {str(k): v for k, v in rates.items()}, "n_blocks": int(dev.size), "chosen_z": z,
           "max_abs_dev": float(np.abs(dev).max()),
           "overrides": ["citadel.abstain=true", "citadel.abstain_mode=ratio", f"citadel.abstain_z={z}"]}
    json.dump(out, open(os.path.join(ROOT, "configs", "revision2_frozen_abstain.json"), "w"), indent=1)
    print("abstain: z =", z, rates)


def freeze_joint() -> None:
    table = {}
    for r in load("rev2_joint_tune"):
        m = re.match(r"tune_joint_ep(\d+)_lr([0-9.e-]+)$", r["name"])
        if m and r["seed"] in CAL:
            table.setdefault((int(m.group(1)), float(m.group(2))), {})[r["seed"]] = float(np.mean([b["ser"] for b in r["blocks"]]))
    full = {k: float(np.mean(list(v.values()))) for k, v in table.items() if sorted(v) == CAL}
    if len(full) < 6:
        print("joint: tuning runs incomplete", len(full), "of 6 settings")
        return
    (ep, lr), ser = min(full.items(), key=lambda kv: kv[1])
    out = {"rule": "lowest mean clean SER over seeds 0-2 among epochs {30,100,300} x lr {1e-3,5e-3}",
           "table": {f"ep{k[0]}_lr{k[1]:g}": v for k, v in sorted(full.items())}, "chosen": f"ep{ep}_lr{lr:g}", "cal_clean_ser": ser,
           "overrides": [f"receiver.epochs={ep}", f"receiver.lr={lr}", "receiver.weight_decay=0.0"]}
    json.dump(out, open(os.path.join(ROOT, "configs", "revision2_frozen_joint.json"), "w"), indent=1)
    print("joint:", out["chosen"], round(ser, 4))


if __name__ == "__main__":
    freeze_abstain()
    freeze_joint()
