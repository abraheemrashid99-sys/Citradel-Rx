#!/usr/bin/env python
"""Parameter sweeps for the paper experiments, run as separate processes.

    python scripts/sweep.py e3 --seeds 0 1 2 --workers 4 [--dry]
    python scripts/sweep.py e2 --receivers citadel meta_deepsic --snrs 6 8 10 12
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "scripts", "run_experiment.py")

ALL_RX = ["lmmse", "online_deepsic", "meta_deepsic", "drift_deepsic", "medbn_deepsic", "titans_single", "citadel"]


def grid(exp: str, args) -> List[Dict[str, str]]:
    rx = args.receivers or ALL_RX
    jobs = []
    if exp == "e1":
        for r, lin, sir, know in itertools.product(["online_deepsic", "meta_deepsic"], [True, False],
                                                   args.sirs or [-10, -5, 0, 5, 10], ["whitebox", "transfer"]):
            jobs.append({"receiver.type": r, "receiver.pretrain": "false", "channel.linear": str(lin).lower(),
                         "attack.type": "a1", "attack.knowledge": know, "attack.sir_db": str(sir),
                         "run.name": f"e1_{r}_{'lin' if lin else 'nl'}_{know}_sir{sir}"})
    elif exp == "e2":
        for r, snr, fad in itertools.product(rx, args.snrs or [4, 6, 8, 10, 12, 14, 16], ["cosine", "jakes"]):
            jobs.append({"receiver.type": r, "channel.snr_db": str(snr), "channel.fading": fad, "attack.type": "none",
                         "run.name": f"e2_{r}_{fad}_snr{snr}"})
    elif exp == "e3":
        for r, sir, know in itertools.product(rx, args.sirs or [-10, -5, 0, 5, 10], ["whitebox", "transfer"]):
            jobs.append({"receiver.type": r, "attack.type": "a1", "attack.knowledge": know, "attack.sir_db": str(sir),
                         "run.name": f"e3_{r}_{know}_sir{sir}"})
        for r in rx:  # equal-power jamming control
            jobs.append({"receiver.type": r, "attack.type": "data_jam", "attack.sir_db": "0",
                         "run.name": f"e3_{r}_datajam_sir0"})
    elif exp == "e3snr":  # SER-vs-SNR curves with and without attack, for the Delta_dB metric
        for r, snr, atk in itertools.product(rx, args.snrs or [4, 6, 8, 10, 12, 14, 16], ["none", "a1"]):
            jobs.append({"receiver.type": r, "channel.snr_db": str(snr), "attack.type": atk, "attack.sir_db": "0",
                         "run.name": f"e3snr_{r}_{atk}_snr{snr}"})
    elif exp == "e4":
        for r, g3 in itertools.product(["citadel", "meta_deepsic", "online_deepsic"], [True, False]):
            gates = "[g1,g2,g3]" if g3 else "[g1,g2]"
            jobs.append({"receiver.type": r, "citadel.gates": gates, "attack.type": "a2",
                         "run.name": f"e4_{r}_{'g3' if g3 else 'nog3'}"})
    elif exp == "e5":
        for ck in [8, 32, 128]:
            jobs.append({"receiver.type": "citadel", "citadel.periods": f"[1,{max(2, ck // 4)},{ck}]", "attack.type": "a3",
                         "run.save_params": "true", "run.name": f"e5_citadel_CK{ck}"})
            jobs.append({"receiver.type": "citadel", "citadel.periods": f"[1,{max(2, ck // 4)},{ck}]", "attack.type": "none",
                         "run.save_params": "true", "run.name": f"e5_citadel_CK{ck}_clean"})
    elif exp == "e7":
        for gates in ["[g1]", "[g1,g2]", "[g1,g2,g3]", "[]"]:
            tag = gates.strip("[]").replace(",", "") or "nogate"
            jobs.append({"receiver.type": "citadel", "citadel.gates": gates, "attack.type": "adaptive",
                         "run.name": f"e7_citadel_{tag}_adaptive"})
    elif exp == "ablate":
        base = {"receiver.type": "citadel", "attack.type": "a1", "attack.sir_db": "0"}
        variants = {
            "K1": {"citadel.levels": "1", "citadel.periods": "[1]"},
            "K2": {"citadel.levels": "2", "citadel.periods": "[1,8]"},
            "K4": {"citadel.levels": "4", "citadel.periods": "[1,4,16,64]"},
            "misaligned": {"citadel.periods": "[1,3,7]"},
            "nogate": {"citadel.gates": "[]"},
            "g1": {"citadel.gates": "[g1]"}, "g2": {"citadel.gates": "[g2]"}, "g3": {"citadel.gates": "[g3]"},
            "last_only": {"citadel.adaptable": "last"},
            "plain_gd": {"citadel.fast_rule": "gd"},
            "adam_no_delta": {"citadel.fast_rule": "adam"},
            "fixed_eta": {"citadel.surprise_lr": "false"},
            "no_reanchor": {"citadel.reanchor_beta": "0.0"},
        }
        for tag, ov in variants.items():
            jobs.append({**base, **ov, "run.name": f"ablate_{tag}"})
    else:
        raise SystemExit(f"unknown experiment {exp}")
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--receivers", nargs="*", default=None)
    ap.add_argument("--snrs", nargs="*", type=float, default=None)
    ap.add_argument("--sirs", nargs="*", type=float, default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="extra dotted overrides applied to every job")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    cfg = args.config or os.path.join(ROOT, "configs", {
        "e1": "e1_reproduce_attack.yaml", "e2": "e2_clean_parity.yaml", "e3": "e3_a1_defence.yaml",
        "e3snr": "e3_a1_defence.yaml", "e4": "e4_a2_drift_spoof.yaml", "e5": "e5_a3_boiling_frog.yaml",
        "e7": "e7_adaptive_attacker.yaml", "ablate": "e3_a1_defence.yaml"}[args.exp])
    jobs = grid(args.exp, args)
    out_dir = f"results/{args.exp}"
    cmds = []
    for j in jobs:
        ov = [f"{k}={v}" for k, v in j.items()] + [f"run.out_dir={out_dir}"] + args.extra
        cmds.append([sys.executable, RUN, cfg, "--quiet", "--seeds", *map(str, args.seeds), "--set", *ov])
    print(f"{len(cmds)} jobs x {len(args.seeds)} seeds")
    if args.dry:
        for c in cmds:
            print(" ".join(c))
        return

    def run(c):
        r = subprocess.run(c, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(("OK  " if r.returncode == 0 else "FAIL") + " " + tail + ("" if r.returncode == 0 else "\n" + r.stderr[-2000:]), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, cmds))


if __name__ == "__main__":
    main()
