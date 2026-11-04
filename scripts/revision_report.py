#!/usr/bin/env python
"""Runs every revision analysis after scripts/revision_campaign.py and writes docs/REV_*.md.

Must run on the machine that ran the campaign: the paired-displacement analysis needs the
level snapshots (*.levels.npz), which are not pushed. It stores the paired curves in the JSON
records so everything else can be regenerated anywhere.

    python scripts/revision_report.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CLEAN = "results/rev_clean"
GROUPS = {
    "rev_p1": "P1 headline grid (test seeds 10-19, randomised window)",
    "rev_p2": "P2 remaining and new baselines",
    "rev_p3_adaptive": "P3 adaptive white-box sweep (B4)",
    "rev_p3_refdeg": "P3 reference degradation, joint pilot + data (B2)",
    "rev_p3_phys": "P3 physical rank-M attacker (B1)",
    "rev_p3_know": "P3 grey-box, query attacker, signal budget (B6, B8)",
    "rev_p4_paired": "P4 paired-displacement runs",
    "rev_p5_ablate": "P5 ablations on test seeds",
    "rev_p6_blocklen": "P6 block length (D1)",
    "rev_p6_arrays": "P6 arrays N > U (D4)",
    "rev_p6_qam16": "P6 16-QAM (D5)",
    "rev_p6_deploy": "P6 30-step and last-layer variants (D9)",
    "rev_p7_sens": "P7 sensitivity (E5)",
    "rev_p8_drift": "P8 optimised slow drift over 1000 blocks (B5)",
}


def run(args):
    print(">>", " ".join(args), flush=True)
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode != 0:
        print("   (failed, continuing)", flush=True)


def main():
    os.chdir(ROOT)
    for d in ("rev_p4_paired", "rev_p8_drift"):
        if os.path.isdir(os.path.join("results", d)):
            run([PY, "scripts/paired_disp.py", f"results/{d}", "--clean", CLEAN, "--out", f"docs/figures/fig_{d}"])
    if os.path.isdir("results/rev_p1"):
        run([PY, "scripts/revision_stats.py", "--attacked", "results/rev_p1", "--clean", CLEAN, "--out", "docs/REV_P1_STATS.md"])
    for d, title in GROUPS.items():
        if os.path.isdir(os.path.join("results", d)):
            run([PY, "scripts/revision_stats.py", "--by-name", "--attacked", f"results/{d}", "--clean", CLEAN,
                 "--out", f"docs/REV_{d.upper()}.md", "--title", title])
    if os.path.isdir("results/rev_p3_deny"):   # deny / abrupt runs have their own no-attack controls
        run([PY, "scripts/revision_stats.py", "--by-name", "--attacked", "results/rev_p3_deny", "--clean", "results/rev_p3_deny",
             "--out", "docs/REV_REV_P3_DENY.md", "--title", "P3 denial of adaptation and abrupt change (B3, D6)"])
    gates = [f"results/{d}" for d in ("rev_p1", "rev_p3_adaptive", "rev_p3_refdeg", "rev_p3_phys") if os.path.isdir(f"results/{d}")]
    if gates:
        run([PY, "scripts/gate_roc.py", *gates])
    print("done: docs/REV_*.md, docs/PAIRED_DISP.md and docs/figures/fig_rev_*")


if __name__ == "__main__":
    main()
