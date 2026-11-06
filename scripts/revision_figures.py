#!/usr/bin/env python
"""Figures of the revised manuscript, from the run records (test seeds only).

  fig_rev_trace   one test seed under white-box A1 at SIR 0 dB: SER per block of the attacked run
                  (solid) and of its clean twin (dashed) for Citadel-Rx, CRC-Rollback, online DeepSIC
                  and Meta-DeepSIC; below, the gate verdicts of CRC-Rollback and Citadel-Rx.
  fig_rev_excess  excess SER per seed at the headline cell (all test seeds), one marker per seed.

    python scripts/revision_figures.py [--seed 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import revision_numbers as rn  # noqa: E402
from paper_style import C, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

C = dict(C, crc_rollback="#009E73", online_select="#E69F00")
LAB = {"citadel": "Citadel-Rx", "crc_rollback": "CRC-Rollback", "online_deepsic": "Online DeepSIC",
       "meta_deepsic": "Meta-DeepSIC", "online_select": "Online + CRC selection"}


def trace(seed: int) -> None:
    rows = {}
    for rx in ("citadel", "crc_rollback", "online_deepsic", "meta_deepsic"):
        c = rn.cell("rev_p1", f"p1_{rx}__wb_sir0", ["rev_clean"])
        if c is None:
            continue
        pr = [p for p in c["pairs"] if p[0]["seed"] == seed]
        if pr:
            rows[rx] = pr[0]
    if not rows:
        print("no records for the trace figure yet")
        return
    fig, (ax, ag) = plt.subplots(2, 1, figsize=(3.45, 3.3), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    w0, w1 = rn.window(next(iter(rows.values()))[0])
    for a_ in (ax, ag):
        a_.axvspan(w0, w1, color="0.92", zorder=0)
    for rx, (a, c) in rows.items():
        x = np.arange(len(a["blocks"]))
        ax.semilogy(x, np.maximum(rn.ser(a), 1e-4), color=C[rx], lw=1.0, label=LAB[rx])
        ax.semilogy(x, np.maximum(rn.ser(c), 1e-4), color=C[rx], lw=0.8, ls="--")
    ax.set_ylabel("SER")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, fontsize=6.5)
    for rx in ("crc_rollback", "citadel"):
        if rx in rows:
            a = rows[rx][0]
            g = [b.get("gamma") if b.get("gamma") is not None else np.nan for b in a["blocks"]]
            ag.step(np.arange(len(g)), g, where="mid", color=C[rx], lw=0.9, label=LAB[rx])
    ag.axhline(0.5, color="k", lw=0.5, ls=":")
    ag.set_ylabel(r"verdict $\gamma_b$")
    ag.set_xlabel("block")
    ag.set_ylim(-0.05, 1.05)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.85, bottom=0.13, hspace=0.12)
    save(fig, "fig_rev_trace")
    print(f"fig_rev_trace: seed {seed}, window {w0}-{w1}, receivers {list(rows)}")


def excess() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 1.9))
    xs = []
    for i, rx in enumerate(("citadel", "online_select", "crc_rollback", "online_deepsic", "meta_deepsic")):
        c = rn.cell("rev_p1", f"p1_{rx}__wb_sir0", ["rev_clean"])
        if c is None:
            continue
        y = np.array(c["seed_means"])
        ax.scatter(np.full(y.size, i) + np.linspace(-0.15, 0.15, y.size), y, s=6, color=C[rx], zorder=3)
        ax.errorbar(i, c["excess"], yerr=[[c["excess"] - c["ci"][0]], [c["ci"][1] - c["excess"]]], color="k", lw=0.8,
                    capsize=2, zorder=4)
        xs.append((i, LAB[rx]))
    if not xs:
        print("no records for the excess figure yet")
        plt.close(fig)
        return
    ax.set_xticks([i for i, _ in xs])
    ax.set_xticklabels([l for _, l in xs], rotation=15, fontsize=6.5)
    ax.set_yscale("symlog", linthresh=1e-2)
    lo = min(0.0, min(min(rn.cell("rev_p1", f"p1_{rx}__wb_sir0", ["rev_clean"])["seed_means"]) for rx in
                      ("citadel", "online_select", "crc_rollback", "online_deepsic", "meta_deepsic")))
    ax.set_ylim(lo * 1.5, 1.5)   # every seed visible (excess SER is at most 1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("excess SER")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.25)
    save(fig, "fig_rev_excess")
    print("fig_rev_excess:", [l for _, l in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()
    os.chdir(ROOT)
    trace(args.seed)
    excess()


if __name__ == "__main__":
    main()
