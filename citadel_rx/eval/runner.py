"""Sequential block runner: channel → (attacker) → receiver → metrics."""
from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..attacks import make_attacker
from ..channels import make_channel
from ..config import Config
from ..frames import FrameGenerator
from ..receivers import make_receiver
from .metrics import block_metrics, summarise
from .records import save_record


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretraining_blocks(cfg: Config, frames: FrameGenerator) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Reference joint pre-training data: one static-channel block of `pretrain_bits` pilot
    bits per SNR in `pretrain_snrs` (fading disabled)."""
    rc = cfg.receiver
    out = []
    pcfg = copy.deepcopy(cfg.channel)
    pcfg.fading = "static"
    rng = np.random.default_rng(cfg.channel.seed + 77)
    for snr in rc.pretrain_snrs:
        pcfg.snr_db = float(snr)
        ch = make_channel(pcfg, frames.mod.is_complex)
        for blk in range(rc.pretrain_blocks):
            st = ch.state(blk)
            bits = rng.integers(0, 2, size=(cfg.channel.n_user, rc.pretrain_bits))
            idx = frames.mod.bits_to_indices(bits).T
            y = ch.transmit(frames.mod.modulate_indices(idx), st, rng)
            out.append((y, idx))
    return out


def run_experiment(cfg: Config, seed: Optional[int] = None, save: bool = True, verbose: bool = True,
                   progress_cb=None) -> Dict[str, Any]:
    seed = cfg.channel.seed if seed is None else seed
    cfg = copy.deepcopy(cfg)
    cfg.channel.seed = seed
    set_seed(seed)
    device = pick_device(cfg.run.device)

    frames = FrameGenerator(cfg)
    channel = make_channel(cfg.channel, frames.mod.is_complex)
    receiver = make_receiver(cfg, device)
    attacker = make_attacker(cfg, device, seed)
    if hasattr(receiver, "set_channel_rho"):
        receiver.set_channel_rho(channel.rho if cfg.channel.fading == "jakes" else 1.0)

    t0 = time.time()
    if cfg.receiver.pretrain and cfg.receiver.type != "lmmse":
        receiver.pretrain(pretraining_blocks(cfg, frames))
    pretrain_time = time.time() - t0

    blocks: List[Dict[str, Any]] = []
    level_snaps: List[Dict[str, Any]] = []
    noise_var = 10 ** (-0.1 * cfg.channel.snr_db)
    for b in range(cfg.frame.n_blocks):
        st = channel.state(b)
        blk = frames.make(b)
        y_p = channel.transmit(blk.s_pilot, st)
        y_d = channel.transmit(blk.s_data, st)
        attacker.observe_clean(b, y_p, blk.idx_pilot)
        tb = time.time()
        y_p_adv, y_d_adv, astats = attacker.craft(b, y_p, blk.idx_pilot, y_d, receiver, noise_var)
        t_attack = time.time() - tb
        tb = time.time()
        out = receiver.process_block(b, y_p_adv, blk.idx_pilot, y_d_adv, aux={"framer": frames, "state": st})
        t_rx = time.time() - tb
        crc_pass = None
        if frames.framer.enabled:
            crc_pass = frames.framer.check(frames.decided_bits_to_subblocks(out.idx_hat))
        m = block_metrics(out.idx_hat, blk.idx_data, frames.mod, crc_pass)
        m.update({"block": b, "t_rx": t_rx, "t_attack": t_attack, "snr_db": cfg.channel.snr_db,
                  "h_norm": float(np.linalg.norm(st.H))})
        m.update({k: v for k, v in astats.items()})
        m.update({k: v for k, v in out.stats.items() if not isinstance(v, (np.ndarray,))})
        blocks.append(m)
        if cfg.run.save_params and hasattr(receiver, "level_snapshot"):
            level_snaps.append({"block": b, "levels": [{k: v.tolist() for k, v in receiver.level_snapshot(k).items()}
                                                      for k in range(receiver.K)]})
        if verbose and (b % cfg.run.log_every == 0 or b == cfg.frame.n_blocks - 1):
            extra = ""
            if "gamma" in m:
                extra = f" gamma={m['gamma']:.2f} r={m.get('surprise_ratio', 0):.2f}"
            print(f"[{cfg.run.name} seed={seed}] block {b:4d} ser={m['ser']:.4f} ber={m['ber']:.4f}"
                  f"{' ATT' if m.get('attacked') else '    '}{extra} t_rx={t_rx:.2f}s", flush=True)
        if progress_cb:
            progress_cb(b, m)

    window = (cfg.attack.start_block, cfg.attack.end_block) if cfg.attack.type != "none" else None
    summary = summarise(blocks, window)
    summary["pretrain_time_s"] = pretrain_time
    summary["mean_t_rx_s"] = float(np.mean([b["t_rx"] for b in blocks]))
    summary["total_time_s"] = time.time() - t0
    record = {"name": cfg.run.name, "seed": seed, "config": cfg.to_dict(), "config_hash": cfg.hash(),
              "receiver": getattr(receiver, "name", cfg.receiver.type), "summary": summary, "blocks": blocks,
              "device": str(device)}
    if level_snaps:
        record["level_snapshots"] = level_snaps
    if save:
        path = save_record(record, cfg.run.out_dir, f"{cfg.run.name}_s{seed}_{cfg.hash()}")
        record["path"] = path
        if verbose:
            print(f"saved {path}")
    return record
