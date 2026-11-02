#!/usr/bin/env python
"""Manuscript figures in a uniform style (small multi-panel, labels (a)/(b)/(c), legend below).

    python scripts/fig_paper.py
Writes docs/figures/fig_run.pdf|png (representative run: SER per block, gate verdict, level
displacement), docs/figures/fig_sir.pdf|png (SER vs SIR, white-box and transfer, with seed
spread), docs/figures/fig_snr.pdf|png (SER vs SNR clean and attacked, Delta_dB construction).
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200})
C = {"online_deepsic": "#d55e00", "meta_deepsic": "#7b4f9d", "citadel": "#0072b2", "lmmse": "#777777",
     "drift_deepsic": "#009e73", "medbn_deepsic": "#cc79a7", "titans_single": "#e69f00"}
L = {"online_deepsic": "Online DeepSIC", "meta_deepsic": "Meta-DeepSIC", "citadel": "Citadel-Rx (proposed)",
     "lmmse": "LMMSE", "drift_deepsic": "Drift-detection", "medbn_deepsic": "MedBN-style",
     "titans_single": "Single-memory Titans"}
OUT = "docs/figures"


def recs(folder):
    return [json.load(open(f)) for f in glob.glob(f"{folder}/*.json")]


def fig_run():
    e3 = recs("results/e3"); e3snr = recs("results/e3snr")
    def pick(rs, rt, atk, kw, seed=0):
        for r in rs:
            c = r["config"]
            if c["receiver"]["type"] == rt and c["attack"]["type"] == atk and r["seed"] == seed and \
                    all(c[k1][k2] == v for (k1, k2), v in kw.items()):
                return r
    att = {rt: pick(e3, rt, "a1", {("attack", "knowledge"): "whitebox", ("attack", "sir_db"): 0}) for rt in ("online_deepsic", "meta_deepsic", "citadel")}
    cln = {rt: pick(e3snr, rt, "none", {("channel", "snr_db"): 10}) for rt in ("online_deepsic", "citadel")}
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.4))
    a0, a1 = 30, 70
    for a in ax:
        a.axvspan(a0, a1, color="0.9", zorder=0)
    for rt, r in att.items():
        ax[0].plot([b["block"] for b in r["blocks"]], [b["ser"] for b in r["blocks"]], color=C[rt], label=L[rt])
    for rt, r in cln.items():
        ax[0].plot([b["block"] for b in r["blocks"]], [b["ser"] for b in r["blocks"]], color=C[rt], ls="--", lw=0.9, label=f"{L[rt].split(' (')[0]}, no attack")
    ax[0].set_yscale("log"); ax[0].set_xlabel("block"); ax[0].set_ylabel("SER"); ax[0].set_title("(a) SER per block, white-box A1, SIR 0 dB", loc="left")
    r = att["citadel"]
    bl = [b["block"] for b in r["blocks"]]
    ax[1].plot(bl, [b["gamma"] for b in r["blocks"]], color=C["citadel"], label="gate verdict $\\gamma_b$")
    ax[1].plot(bl, [b["verdict_g1"] for b in r["blocks"]], color="#009e73", lw=0.8, ls=":", label="G1 residual")
    ax[1].plot(bl, [b["verdict_g2"] for b in r["blocks"]], color="#e69f00", lw=0.8, ls=":", label="G2 CRC")
    ax[1].axhline(0.5, color="k", lw=0.6, ls="--"); ax[1].set_ylim(-0.03, 1.03)
    ax[1].set_xlabel("block"); ax[1].set_title("(b) Citadel-Rx gate verdicts", loc="left")
    ax[2].plot(bl, [b["level1_disp"] for b in r["blocks"]], color=C["citadel"], lw=0.9, label="level 1 (fast)")
    ax[2].plot(bl, [b["level2_disp"] for b in r["blocks"]], color="#56b4e9", label="level 2 ($C_2=8$)")
    ax[2].plot(bl, [b["level3_disp"] for b in r["blocks"]], color="#003f5c", label="level 3 ($C_3=32$)")
    rc = cln["citadel"]
    ax[2].plot(bl, [b["level3_disp"] for b in rc["blocks"]], color="#003f5c", ls="--", lw=0.9, label="level 3, no attack")
    ax[2].set_xlabel("block"); ax[2].set_ylabel("$\\|\\theta^{(k)}_b-\\theta^{(k)}_0\\|$"); ax[2].set_title("(c) Memory-level displacement", loc="left")
    for a in ax:
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.42, wspace=0.35)
    fig.savefig(f"{OUT}/fig_run.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_run.png", bbox_inches="tight")


def fig_sir():
    e3 = recs("results/e3")
    d = defaultdict(list)
    for r in e3:
        c = r["config"]
        if c["attack"]["type"] == "a1":
            d[(c["receiver"]["type"], c["attack"]["knowledge"], c["attack"]["sir_db"])].append(r["summary"]["ser_attack"])
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    for i, know in enumerate(("whitebox", "transfer")):
        for rt in ("lmmse", "online_deepsic", "meta_deepsic", "drift_deepsic", "medbn_deepsic", "titans_single", "citadel"):
            sirs = sorted({k[2] for k in d if k[0] == rt and k[1] == know})
            if not sirs or (rt == "lmmse" and know == "whitebox"):   # LMMSE has no adaptation map to attack through
                continue
            m = [np.mean(d[(rt, know, s)]) for s in sirs]; sd = [np.std(d[(rt, know, s)]) for s in sirs]
            ax[i].errorbar(sirs, m, yerr=sd, color=C[rt], marker="o", ms=2.5, capsize=2, label=L[rt], lw=1.6 if rt == "citadel" else 1.0)
        ax[i].set_yscale("log"); ax[i].set_xlabel("SIR of the pilot perturbation (dB)")
        ax[i].set_title(f"({'ab'[i]}) {'White-box' if know == 'whitebox' else 'Transfer'} A1, SER in the attack window", loc="left")
    ax[0].set_ylabel("SER (any user wrong)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sir.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_sir.png", bbox_inches="tight")


def fig_snr():
    rs = recs("results/e3snr") + recs("results/e3rel")
    d = defaultdict(list)
    for r in rs:
        c = r["config"]
        if c["channel"]["model"] != "synthetic":
            continue
        rel = c["citadel"].get("g2_slack_mode", "absolute") == "relative"
        key = (c["receiver"]["type"] + ("_rel" if rel else ""), c["attack"]["type"], c["channel"]["snr_db"])
        d[key].append(r["summary"]["ser_attack"] if c["attack"]["type"] == "a1" else r["summary"]["ser_mean"])
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    for i, rts in enumerate((("online_deepsic", "meta_deepsic", "lmmse"), ("citadel", "citadel_rel"))):
        for rt in rts:
            for atk, ls in (("none", "--"), ("a1", "-")):
                snrs = sorted({k[2] for k in d if k[0] == rt and k[1] == atk})
                if not snrs:
                    continue
                base = rt.replace("_rel", "")
                lab = L[base].replace(" (proposed)", "") + (", relative slack" if rt.endswith("_rel") else "") + (", no attack" if atk == "none" else ", A1 SIR 0 dB")
                ax[i].plot(snrs, [np.mean(d[(rt, atk, s)]) for s in snrs], ls=ls, marker="o", ms=2.5, color=C[base],
                           alpha=0.6 if rt.endswith("_rel") else 1.0, label=lab)
        ax[i].set_yscale("log"); ax[i].set_xlabel("SNR (dB)")
        ax[i].legend(loc="lower left", frameon=False)
    ax[0].set_ylabel("SER"); ax[0].set_title("(a) Baselines: attacked SER never meets the clean curve", loc="left")
    ax[1].set_title("(b) Citadel-Rx: $\\Delta_{\\rm dB}$ is the horizontal gap", loc="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_snr.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/fig_snr.png", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_run(); fig_sir(); fig_snr()
    print("wrote", OUT)
