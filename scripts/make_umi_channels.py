#!/usr/bin/env python
"""3GPP TR 38.901 UMi channels with the attacker placed in the scene (reviewer audit D3).

For every seed and attacker distance, one drop of the UMi street-canyon model (Sionna PHY 2.2,
`sionna.phy.channel.tr38901.UMi`, path loss, LoS probability and shadow fading of 38.901) with
  * a base station at 10 m height with a 1 x N array (single polarisation, 38.901 element pattern);
  * U users at uniform distances in [20, 150] m and angles in [-60, 60] deg, 1.5 m high, moving at
    3 km/h in random directions; ideal slow power control: every user's channel is scaled to unit
    mean received power per antenna (the noise power then sets the SNR of the paper, 10 dB);
  * the attacker, a single-antenna transmitter at distance d at the maximum power of a user;
  * a reference user at the cell edge (150 m) at maximum power, which fixes the power-control
    target: the users arrive with the power a cell-edge user has at maximum power.
The attacker's channel is scaled by the same factor as the users' target, so its received power
relative to one power-controlled user is g_att / g_ref, and the attack budget relative to the users'
signal energy is SIR = 10 log10(U g_ref / g_att): it follows from the geometry, not from a setting.

Each block is one narrowband channel (the frequency response at the carrier) sampled every
T_block = 5 ms; frequency selectivity within a block is not modelled.

    python scripts/make_umi_channels.py --seeds 10 11 12 13 14 --dists 35 75 150
writes resources/umi/umi_N4_U4_d{d}_s{seed}.npz (H: blocks x N x U, G: blocks x N x 1, sir_db, ...)
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "resources", "umi")


def drop(seed: int, d_att: float, n_ant: int, n_user: int, n_blocks: int, t_block: float, fc: float,
         speed_kmh: float) -> dict:
    import sionna.phy
    from sionna.phy.channel.tr38901 import UMi, PanelArray
    sionna.phy.config.seed = seed
    rng = np.random.default_rng(seed + 38901)
    bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=n_ant, polarization="single", polarization_type="V",
                          antenna_pattern="38.901", carrier_frequency=fc)
    ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization="single", polarization_type="V",
                          antenna_pattern="omni", carrier_frequency=fc)
    model = UMi(carrier_frequency=fc, o2i_model="low", ut_array=ut_array, bs_array=bs_array, direction="uplink",
                enable_pathloss=True, enable_shadow_fading=True)
    # users, then the attacker, then the cell-edge reference
    r = np.concatenate([rng.uniform(20.0, 150.0, n_user), [d_att, 150.0]])
    phi = np.deg2rad(rng.uniform(-60.0, 60.0, n_user + 2))
    n_ut = n_user + 2
    ut_loc = np.stack([r * np.cos(phi), r * np.sin(phi), np.full(n_ut, 1.5)], -1)[None]
    v = speed_kmh / 3.6
    head = rng.uniform(0, 2 * np.pi, n_ut)
    vel = np.stack([v * np.cos(head), v * np.sin(head), np.zeros(n_ut)], -1)
    vel[n_user:] = 0.0                                     # attacker and reference: static
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    model.set_topology(ut_loc=t(ut_loc), bs_loc=t([[[0.0, 0.0, 10.0]]]), ut_orientations=t(np.zeros((1, n_ut, 3))),
                       bs_orientations=t(np.zeros((1, 1, 3))), ut_velocities=t(vel[None]),
                       in_state=torch.zeros((1, n_ut), dtype=torch.bool))
    a, tau = model(num_time_samples=n_blocks, sampling_frequency=1.0 / t_block)
    a = a.detach().cpu().numpy()                            # (1, 1, N, n_ut, 1, paths, T)
    Hf = a[0, 0, :, :, 0, :, :].sum(axis=2)                 # frequency response at the carrier: (N, n_ut, T)
    Hf = np.transpose(Hf, (2, 0, 1))                        # (T, N, n_ut)
    g = (np.abs(Hf) ** 2).mean(axis=(0, 1))                 # mean received power per antenna, per transmitter
    H = Hf[:, :, :n_user] / np.sqrt(g[:n_user])[None, None, :]
    G = Hf[:, :, n_user:n_user + 1] / np.sqrt(g[n_user + 1])
    sir = 10 * math.log10(n_user * g[n_user + 1] / g[n_user])
    los = None
    try:
        los = model._scenario.los.detach().cpu().numpy()[0].tolist()   # internal: LoS state per link
    except Exception:
        pass
    return {"H": H.astype(np.complex64), "G": G.astype(np.complex64), "sir_db": sir, "path_gain_db": (10 * np.log10(g)).tolist(),
            "dist": r.tolist(), "angle_deg": np.rad2deg(phi).tolist(), "los": los}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    ap.add_argument("--dists", type=float, nargs="+", default=[35.0, 75.0, 150.0])
    ap.add_argument("--n-ant", type=int, default=4)
    ap.add_argument("--n-user", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--t-block", type=float, default=0.005)
    ap.add_argument("--fc", type=float, default=3.5e9)
    ap.add_argument("--speed-kmh", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for d in args.dists:
        for s in args.seeds:
            out = drop(s, d, args.n_ant, args.n_user, args.blocks, args.t_block, args.fc, args.speed_kmh)
            fn = os.path.join(OUT, f"umi_N{args.n_ant}_U{args.n_user}_d{d:g}_s{s}.npz")
            np.savez(fn, H=out["H"], G=out["G"], sir_db=out["sir_db"],
                     meta=json.dumps({k: out[k] for k in ("path_gain_db", "dist", "angle_deg", "los")}))
            summary.append({"d_att": d, "seed": s, "sir_db": round(out["sir_db"], 2)})
            print(f"d_att {d:g} m, seed {s}: SIR {out['sir_db']:+.1f} dB, path gains (dB) "
                  + " ".join(f"{x:.0f}" for x in out["path_gain_db"]), flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
