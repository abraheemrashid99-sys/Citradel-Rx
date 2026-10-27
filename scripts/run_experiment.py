#!/usr/bin/env python
"""Run one configuration (optionally several seeds).

    python scripts/run_experiment.py configs/e2_clean.yaml --set receiver.type=citadel --seeds 0 1 2
    python scripts/run_experiment.py --quick            # smoke test on the laptop
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# Small per-user MLPs do not benefit from intra-op parallelism; several runs side by side do.
torch.set_num_threads(int(os.environ.get("CITADEL_THREADS", "1")))

from citadel_rx.config import Config  # noqa: E402
from citadel_rx.eval import run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. attack.sir_db=5")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if the record already exists")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config, args.set) if args.config else Config()
    if not args.config:
        for ov in args.set:
            cfg.set(ov)
    if args.quick:
        cfg.frame.n_blocks = min(cfg.frame.n_blocks, 12)
        cfg.frame.data_bits = min(cfg.frame.data_bits, 4000)
        cfg.receiver.epochs = min(cfg.receiver.epochs, 30)
        cfg.receiver.pretrain_epochs = min(cfg.receiver.pretrain_epochs, 30)
        cfg.receiver.meta_epochs = min(cfg.receiver.meta_epochs, 20)
        cfg.citadel.fast_steps = min(cfg.citadel.fast_steps, 30)
        cfg.citadel.periods = [1, 2, 4][: cfg.citadel.levels]
        cfg.attack.pgd_steps = min(cfg.attack.pgd_steps, 3)
        cfg.attack.start_block, cfg.attack.end_block = 4, 8
        cfg.run.name = cfg.run.name + "_quick"
        cfg.run.log_every = 1
    seeds = args.seeds if args.seeds is not None else cfg.run.seeds
    for s in seeds:
        # resume support: a run is identified by (name, seed, config hash); skip finished ones
        import copy
        probe = copy.deepcopy(cfg)
        probe.channel.seed = s
        existing = os.path.join(cfg.run.out_dir, f"{cfg.run.name}_s{s}_{probe.hash()}.json")
        if os.path.exists(existing) and not args.force and not args.no_save:
            print(f"== {cfg.run.name} seed {s}: exists, skipped ({existing})")
            continue
        rec = run_experiment(cfg, seed=s, save=not args.no_save, verbose=not args.quiet)
        summ = rec["summary"]
        print(f"== {cfg.run.name} seed {s}: ser_mean={summ['ser_mean']:.4f} "
              + " ".join(f"{k}={v:.4f}" for k, v in summ.items() if k.startswith("ser_") and isinstance(v, float) and k != "ser_mean"))


if __name__ == "__main__":
    main()
