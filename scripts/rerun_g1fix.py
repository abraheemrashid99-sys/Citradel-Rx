#!/usr/bin/env python
"""Re-run the attacks that use the soft-gate surrogate, after the fix of its G1 scale.

The soft G1 term of the adaptive attacker (pilot_poisoning.Attacker._g1_soft) averaged the squared
residual over the 2N real coordinates of the received vector, half the per-antenna scale of the
receiver's G1 (gates/physics.model_residual) for complex modulation. The rows that use the soft-gate
surrogate against Citadel-Rx are re-run here with the fixed code, each attacked run with a clean twin
from the same machine, so the paired excess SER compares runs of one machine:

    p3_citadel_softgate__sir0, p3_citadel_softgate__sir10, p3_citadel_strongest__sir0   (seeds 10-14)

Every run takes its settings from the record it replaces (results/rev_p3_adaptive), so nothing but the
fixed line of attack code differs. The old records stay where they are (docs/G1_SOFT_FIX.md).

    CITADEL_MACHINE=pool13 python scripts/rerun_g1fix.py --seeds 10 11 12 --workers 4
    CITADEL_MACHINE=pool12 python scripts/rerun_g1fix.py --seeds 13 14 --workers 4
    python scripts/rerun_g1fix.py --dry
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OLD = os.path.join(ROOT, "results", "rev_p3_adaptive")
NEW = os.path.join(ROOT, "results", "rev_p3_g1fix")
ROWS = ["p3_citadel_strongest__sir0", "p3_citadel_softgate__sir0", "p3_citadel_softgate__sir10"]   # longest first


def old_record(name: str, seed: int) -> dict:
    fs = [f for f in glob.glob(os.path.join(OLD, f"{name}_s{seed}_*.json")) if not f.endswith(".levels.json")]
    if len(fs) != 1:
        raise SystemExit(f"expected one record of {name} seed {seed} in {OLD}, found {len(fs)}")
    return json.load(open(fs[0]))


def done(out_dir: str, name: str, seed: int) -> bool:
    for f in glob.glob(os.path.join(out_dir, f"{name}_s{seed}_*.json")):
        try:
            json.load(open(f))
            return True
        except Exception:
            pass
    return False


def plan(seeds, machine: str):
    jobs = []
    for name in ROWS:
        for s in seeds:
            rec = old_record(name, s)
            jobs.append(("attacked", name, s, rec["config"], NEW))
    clean_dir = os.path.join(ROOT, "results", f"rev_clean_{machine}")
    for s in seeds:   # one clean twin per seed serves all three rows (same settings apart from attack.*)
        cfg = json.loads(json.dumps(old_record(ROWS[0], s)["config"]))
        cfg["attack"]["type"] = "none"
        jobs.append(("clean", f"g1fix_twin_citadel", s, cfg, clean_dir))
    # clean twins are short: run them first on one worker so the pre-training cache is warm for the
    # attacked runs of the same seed; the long combined attacks start at once on the other workers
    clean = [j for j in jobs if j[0] == "clean"]
    att = [j for j in jobs if j[0] == "attacked"]
    return att[: len(seeds)] + clean + att[len(seeds):]


def run_one(job):
    kind, name, seed, cfgd, out_dir = job
    import torch
    torch.set_num_threads(1)
    from citadel_rx.config import Config
    from citadel_rx.eval import run_experiment
    if done(out_dir, name, seed):
        return f"{kind} {name} s{seed}: exists"
    cfg = Config.from_dict(cfgd)
    cfg.run.out_dir = out_dir
    cfg.run.name = name
    t0 = time.time()
    run_experiment(cfg, seed=seed, save=True, verbose=False)
    return f"{kind} {name} s{seed}: done in {time.time() - t0:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[10, 11, 12, 13, 14])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    machine = os.environ.get("CITADEL_MACHINE", "")
    if not machine and not a.dry:
        raise SystemExit("set CITADEL_MACHINE (the clean twins and the attacked runs must name one machine)")
    os.environ.setdefault("CITADEL_PRETRAIN_CACHE", os.path.join(ROOT, "cache", "pretrain"))
    os.environ["CITADEL_THREADS"] = "1"
    jobs = plan(a.seeds, machine or "DRY")
    for j in jobs:
        print(f"{j[0]:9s} {j[1]:34s} seed {j[2]} -> {os.path.relpath(j[4], ROOT)}")
    if a.dry:
        return
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(run_one, jobs):
            print(time.strftime("%H:%M:%S"), msg, flush=True)


if __name__ == "__main__":
    main()
