#!/usr/bin/env python
"""Draw the retention certificate (Corollary 2.2) with *measured* constants on the E5 figure.

Certificate (Corollary 2.2, implemented form):
    Delta_f(W) = eps * G_adapt * min{ 1 / (1 - L),  rho_blk * (1 - p_rej) * W },  L = L_theta + L_a
    D_K(W)    <= Delta_f(W) * (1 - (1 - eta_K)^(W / C_K))  <=  eta_K * Delta_f * W / C_K
  eta_K   : level-K consolidation rate (config, level_lr[K-2] for K levels)
  rho_blk : fraction of attacked blocks (A3: 1)
  eps     : per-block l2 budget of A3 = a3_eps_small * ||Y_p||  (measured on clean pilot blocks)
  G_adapt : input-Lipschitz constant of the block adaptation map (fast level after one block of
            adaptation as a function of the pilot observation), measured by finite differences
            (theory.bound.estimate_adaptation_lipschitz; a lower bound of the true constant)
  L_theta : parameter-contraction factor of the same map; L_a its sensitivity to the slow anchor.
            L = L_theta + L_a < 1 means the paired fast trajectories forget a displacement
            geometrically (delta-rule decay + proximal pull); otherwise the min() above falls back
            to the per-window injection count, which needs no contraction.
  p_rej   : gate rejection rate of the attacked blocks in the E5 records (A3 passes: ~0)
  G       : the single-gradient input-Lipschitz constant of the proposal's statement is also
            printed for reference

The slope of D_K in W that the certificate allows is compared with the measured slope from
``results/e5`` (scripts/analyze_e5.py). Output: docs/figures/fig_e5_certificate_bound.png and
the numbers printed as a Markdown table.

    python scripts/certificate_check.py configs/e5_a3_boiling_frog.yaml
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(1)

from citadel_rx.channels import make_channel  # noqa: E402
from citadel_rx.config import Config  # noqa: E402
from citadel_rx.eval.runner import pretraining_blocks, set_seed  # noqa: E402
from citadel_rx.frames import FrameGenerator  # noqa: E402
from citadel_rx.receivers import make_receiver  # noqa: E402
from citadel_rx.theory.bound import estimate_adaptation_lipschitz, estimate_input_lipschitz  # noqa: E402


def measure_constants(cfg: Config, n_blocks: int = 20, seed: int = 0):
    """Run Citadel on clean blocks and measure G (max over blocks) and the A3 budget eps."""
    set_seed(seed)
    cfg.run.device = "cpu"
    cfg.channel.seed = seed
    frames = FrameGenerator(cfg)
    channel = make_channel(cfg.channel, frames.mod.is_complex)
    rx = make_receiver(cfg, torch.device("cpu"))
    rx.pretrain(pretraining_blocks(cfg, frames))
    Gs, epss, Ga, Lt, La = [], [], [], [], []
    for b in range(n_blocks):
        st = channel.state(b)
        blk = frames.make(b)
        y_p = channel.transmit(blk.s_pilot, st)
        epss.append(cfg.attack.a3_eps_small * float(np.linalg.norm(y_p)))
        yf, it = rx.to_feat(y_p), rx.to_idx(blk.idx_pilot)
        Gs.append(estimate_input_lipschitz(rx.core, rx.params, yf, it, rx.A, n_probe=8, radius=0.05 * float(np.linalg.norm(y_p)), seed=b))
        if b >= 2:   # after the surprise EMA is seeded, at the receiver's own learning rate
            ga, lt, la = estimate_adaptation_lipschitz(rx, yf, it, eta=rx.eta_max * 0.5, n_probe=3, seed=b)
            Ga.append(ga); Lt.append(lt); La.append(la)
        rx.process_block(b, y_p, blk.idx_pilot, channel.transmit(blk.s_data, st), aux={"framer": frames})
    return {"G_grad_max": float(np.max(Gs)), "G_grad_mean": float(np.mean(Gs)), "eps": float(np.mean(epss)),
            "G_adapt_max": float(np.max(Ga)), "G_adapt_mean": float(np.mean(Ga)),
            "L_theta_max": float(np.max(Lt)), "L_theta_mean": float(np.mean(Lt)),
            "L_a_max": float(np.max(La)), "L_a_mean": float(np.mean(La))}


def main(cfg_path: str):
    cfg = Config.from_yaml(cfg_path)
    c0 = measure_constants(cfg, n_blocks=int(os.environ.get("CERT_BLOCKS", "12")))
    eps, G_adapt = c0["eps"], c0["G_adapt_max"]
    L_tot = c0["L_theta_max"] + c0["L_a_max"]
    print(json.dumps(c0, indent=1))
    if L_tot >= 1:
        print("NOTE: L_theta + L_a >= 1: no measured contraction; the certificate uses the injection count only")
    geo_cap = 1.0 / (1.0 - L_tot) if L_tot < 1 else np.inf
    recs = defaultdict(list)
    for fn in glob.glob("results/e5/*.json"):
        r = json.load(open(fn))
        if "paired_disp" in r:
            recs[r["config"]["citadel"]["periods"][-1]].append(r)
    if not recs:
        print("no paired E5 records; run scripts/compact_e5.py first")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    rows = ["| C_K | eta_K | p_rej (measured) | Delta_f | certificate slope eta_K Delta_f / C_K | measured slope | max_W measured / certificate |",
            "|---|---|---|---|---|---|---|"]
    for ck in sorted(recs):
        rs = recs[ck]
        c = rs[0]["config"]
        a0, a1 = c["attack"]["start_block"], c["attack"]["end_block"]
        eta_K = c["citadel"]["level_lr"][len(c["citadel"]["periods"]) - 2]
        p_rej = float(np.mean([r["summary"].get("gate_reject_rate_attacked", 0.0) for r in rs]))
        DK = np.array([r["paired_disp"][-1] for r in rs]).mean(0)
        slope = np.polyfit(np.arange(a1 - a0), DK[a0:a1], 1)[0]
        W = np.arange(0, a1 - a0)
        delta_f = eps * G_adapt * np.minimum(geo_cap, 1.0 * (1 - p_rej) * W)
        bound_slope = eta_K * eps * G_adapt * min(geo_cap, (1 - p_rej) * (a1 - a0)) / ck
        cert = delta_f * (1 - (1 - eta_K) ** (W / ck))
        meas = DK[a0:a1] - DK[a0]
        ratio = float(np.max(meas[1:] / cert[1:]))
        line, = ax.plot(W, meas, label=f"measured, $C_K$={ck}")
        ax.plot(W, cert, "--", color=line.get_color(), alpha=0.7, label=f"certificate, $C_K$={ck}")
        rows.append(f"| {ck} | {eta_K:g} | {p_rej:.2f} | {float(delta_f[-1]):.3f} | {bound_slope:.4f} | {slope:.4f} | {ratio:.3f} |")
    ax.set_xlabel("attacked blocks $W$"); ax.set_ylabel("$\\|\\theta^{(K)}_{adv}-\\theta^{(K)}_{clean}\\|$")
    ax.set_yscale("log"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Cor. 2.2: measured $G_\\Phi$={G_adapt:.2f}, $L_\\theta+L_a$={L_tot:.2f}, $\\epsilon$={eps:.2f}", fontsize=9)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig("docs/figures/fig_e5_certificate_bound.png", dpi=200)
    print("\n".join(rows))
    json.dump({**c0, "rows": rows}, open("docs/certificate_constants.json", "w"), indent=1)
    print("wrote docs/figures/fig_e5_certificate_bound.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/e5_a3_boiling_frog.yaml")
