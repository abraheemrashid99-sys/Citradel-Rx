#!/usr/bin/env python
"""Every number of the revised manuscript, computed from the run records.

Writes paper/numbers.tex (one macro per number, used as \\val{key} in the text) and the table bodies
paper/tables/*.tex. Nothing in the paper's results is typed by hand: re-run this script after new
records arrive and recompile. Keys used in paper/main.tex but missing here are listed at the end
(they print as a red [key?] in the PDF).

    python scripts/revision_numbers.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from revision_stats import decorrelation_length, holm, moving_block_bootstrap, seed_tests, wilson  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT_NUM = os.path.join(ROOT, "paper", "numbers.tex")
OUT_TAB = os.path.join(ROOT, "paper", "tables")
V: Dict[str, str] = {}          # macro key -> text
NBOOT = 4000

RXLABEL = {"citadel": "Citadel-Rx", "online_deepsic": "Online DeepSIC", "meta_deepsic": "Meta-DeepSIC",
           "crc_rollback": "CRC-Rollback", "online_select": "Online + CRC selection", "lmmse": "LMMSE",
           "lmmse_robust": "Robust LMMSE (Huber)", "rollback_allgates": "Rollback, three gates",
           "l2sp_deepsic": "L2-SP", "ema_deepsic": "EMA teacher", "ewc_deepsic": "EWC", "dd_deepsic": "CRC-verified DD",
           "cotta_deepsic": "CoTTA-style", "eata_deepsic": "EATA-style", "drift_deepsic": "Drift detection",
           "titans_single": "Single memory (Titans)", "medbn_deepsic": "Median-of-means gradient"}


# ------------------------------------------------------------------------------------ records
_cache: Dict[str, List[dict]] = {}


def load(folder: str) -> List[dict]:
    if folder not in _cache:
        out = []
        for fn in sorted(glob.glob(os.path.join(RES, folder, "*.json"))):
            try:
                r = json.load(open(fn))
                r["_file"] = fn
                out.append(r)
            except Exception:
                pass
        _cache[folder] = out
    return _cache[folder]


def mkey(cfg: dict) -> str:
    return json.dumps({k: v for k, v in cfg.items() if k not in ("attack", "run")}, sort_keys=True)


def twins(clean_folders: List[str]) -> Dict[Tuple[str, int], dict]:
    idx = {}
    for f in clean_folders:
        for r in load(f):
            if r["config"]["attack"]["type"] == "none":
                idx[(mkey(r["config"]), r["seed"])] = r
    return idx


def window(r: dict) -> Tuple[int, int]:
    w = r["summary"].get("attack_window")
    if w:
        return int(w[0]), int(w[1])
    return int(r["config"]["attack"]["start_block"]), int(r["config"]["attack"]["end_block"])


def ser(r: dict) -> np.ndarray:
    return np.array([b["ser"] for b in r["blocks"]], dtype=float)


def rejections(r: dict, attacked: Optional[bool], skip: int = 5) -> Tuple[int, int]:
    k = n = 0
    for b in r["blocks"]:
        if b["block"] < skip or b.get("gamma") is None:
            continue
        if attacked is None or bool(b.get("attacked")) == attacked:
            k += b["gamma"] < 0.5
            n += 1
    return k, n


# ------------------------------------------------------------------------------------ one cell
def cell(folder: str, name: str, clean: List[str]) -> Optional[dict]:
    """Attacked configuration `name` in `folder`, paired with its clean twins (same seed, settings)."""
    tw = twins(clean)
    prs = []
    for r in load(folder):
        if r["name"] == name and r["config"]["attack"]["type"] != "none":
            c = tw.get((mkey(r["config"]), r["seed"]))
            if c is not None:
                prs.append((r, c))
    if not prs:
        return None
    prs.sort(key=lambda p: p[0]["seed"])
    win, post, raw, cwin = [], [], [], []
    k1 = n1 = k0 = n0 = 0
    for a, c in prs:
        w0, w1 = window(a)
        sa, sc = ser(a), ser(c)
        n = min(sa.size, sc.size)
        win.append(sa[w0:min(w1, n)] - sc[w0:min(w1, n)])
        post.append(sa[w1:min(w1 + 30, n)] - sc[w1:min(w1 + 30, n)])
        raw.append(sa[w0:min(w1, n)])
        cwin.append(sc[w0:min(w1, n)])
        a1, b1 = rejections(a, True)
        a0, b0 = rejections(c, None)
        k1, n1, k0, n0 = k1 + a1, n1 + b1, k0 + a0, n0 + b0
    L = decorrelation_length(win)
    blk = max(5, 2 * L)
    lo, hi = moving_block_bootstrap(win, blk, n_boot=NBOOT)
    pl, ph = moving_block_bootstrap(post, blk, n_boot=NBOOT) if all(p.size for p in post) else (np.nan, np.nan)
    sm = np.array([w.mean() for w in win])
    st = seed_tests(sm)
    return {"n": len(prs), "seeds": [a["seed"] for a, _ in prs], "excess": float(sm.mean()), "ci": (lo, hi),
            "post": float(np.mean([p.mean() for p in post])), "post_ci": (pl, ph), "raw": float(np.mean([x.mean() for x in raw])),
            "clean_win": float(np.mean([x.mean() for x in cwin])), "seed_means": sm.tolist(), "decor": L, "boot_block": blk,
            "rej": (k1, n1), "frej": (k0, n0), "wilcoxon_p": st["wilcoxon_p"], "min_p": st["min_attainable_p"],
            "sign_p": st["sign_p"], "pairs": prs,
            "clean_all": float(np.mean([ser(c)[5:].mean() for _, c in prs]))}


# ------------------------------------------------------------------------------------ formatting
def f_ser(x: float) -> str:
    """Plain decimals for non-negative values; math mode (true minus sign) for negative ones."""
    if x is None or not np.isfinite(x):
        return "--"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 0.1:
        t = f"{ax:.3f}"
    elif ax >= 0.001:
        t = f"{ax:.4f}"
    else:
        m, e = f"{ax:.1e}".split("e")
        t = f"{m}\\cdot10^{{{int(e)}}}"
    if x < 0:
        return f"$-{t}$"
    return f"${t}$" if "cdot" in t else t


def f_sig(x: float) -> str:
    s = f_ser(x)
    if s == "--" or x is None or not np.isfinite(x) or x <= 0:
        return s
    return "$+" + s[1:] if s.startswith("$") else f"$+{s}$"


def f_ci(ci) -> str:
    return f"[{f_sig(ci[0])}, {f_sig(ci[1])}]"


def f_rate(kn) -> str:
    k, n = kn
    return "--" if n == 0 else f"{k / n:.3f}"


def f_wil(kn) -> str:
    k, n = kn
    if n == 0:
        return "--"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def f_p(p: float) -> str:
    return "--" if p is None or not np.isfinite(p) else (f"{p:.3f}" if p >= 0.001 else "<0.001")


def put(key: str, val: str) -> None:
    V[key] = val


def put_cell(prefix: str, c: Optional[dict]) -> None:
    if c is None:
        return
    put(prefix + ".n", str(c["n"]))
    put(prefix + ".excess", f_sig(c["excess"]))
    put(prefix + ".ci", f_ci(c["ci"]))
    put(prefix + ".raw", f_ser(c["raw"]))
    put(prefix + ".cleanwin", f_ser(c["clean_win"]))
    put(prefix + ".post", f_sig(c["post"]))
    put(prefix + ".rej", f_rate(c["rej"]))
    put(prefix + ".rejk", f"{c['rej'][0]}/{c['rej'][1]}")
    put(prefix + ".frej", f_rate(c["frej"]))
    put(prefix + ".frejk", f"{c['frej'][0]}/{c['frej'][1]}")
    put(prefix + ".rejpct", f"{100 * c['rej'][0] / max(1, c['rej'][1]):.0f}")
    put(prefix + ".frejpct", f"{100 * c['frej'][0] / max(1, c['frej'][1]):.1f}")
    put(prefix + ".p", f_p(c["wilcoxon_p"]))
    put(prefix + ".minp", f_p(c["min_p"]))
    put(prefix + ".clean", f_ser(c["clean_all"]))
    put(prefix + ".decor", str(c["decor"]))


NCOLS = {"p1": 9, "p1_paired": 6, "attacks": 6, "refdeg": 6, "deny": 7, "baselines": 5, "ablations": 6, "scenarios": 7,
         "sensitivity": 7, "calibration": 7, "retention": 7}


def write_table(name: str, rows: List[str], pending: str = "runs pending") -> None:
    os.makedirs(OUT_TAB, exist_ok=True)
    body = "\n".join(rows) if rows else f"\\multicolumn{{{NCOLS.get(name, 2)}}}{{@{{}}l}}{{\\emph{{({pending})}}}}\\\\"
    open(os.path.join(OUT_TAB, name + ".tex"), "w").write(body + "\n")


# ------------------------------------------------------------------------------------ P1 headline
P1RX = ["citadel", "online_deepsic", "meta_deepsic", "crc_rollback", "online_select"]
P1CELLS = ["wb_sir-10", "wb_sir0", "wb_sir10", "tr_sir0", "jam"]
CK = {"wb_sir-10": "wbm10", "wb_sir0": "wb0", "wb_sir10": "wb10", "tr_sir0": "tr0", "jam": "jam"}


def p1() -> None:
    cells = {(rx, cl): cell("rev_p1", f"p1_{rx}__{cl}", ["rev_clean"]) for rx in P1RX for cl in P1CELLS}
    rows = []
    for rx in P1RX:
        for cl in P1CELLS:
            put_cell(f"p1.{rx}.{CK[cl]}", cells[(rx, cl)])
        c0 = cells[(rx, "wb_sir0")]
        if c0 is None:
            continue
        ex = [cells[(rx, cl)] for cl in P1CELLS]
        rows.append(" & ".join([RXLABEL[rx], f_ser(c0["clean_all"])] +
                               [(f_sig(c["excess"]) if c else "--") for c in ex[:1]] +
                               [f"{f_sig(c0['excess'])} {f_ci(c0['ci'])}"] +
                               [(f_sig(c["excess"]) if c else "--") for c in ex[2:]] +
                               [f_rate(c0["rej"]), f_rate(c0["frej"])]) + r"\\")
    write_table("p1", rows)
    # paired comparisons at the headline cell and on the clean twins (same seeds, same channel draws)
    ref = cells[("citadel", "wb_sir0")]
    if ref is None:
        return
    pr, fam_att, fam_cln = [], [], []
    for rx in P1RX[1:]:
        c = cells[(rx, "wb_sir0")]
        if c is None:
            continue
        common = sorted(set(ref["seeds"]) & set(c["seeds"]))
        if not common:
            continue
        da = np.array([np.mean(ser(a)[window(a)[0]:window(a)[1]]) for a, _ in ref["pairs"] if a["seed"] in common]) - \
            np.array([np.mean(ser(a)[window(a)[0]:window(a)[1]]) for a, _ in c["pairs"] if a["seed"] in common])
        dc = np.array([ser(cl)[5:].mean() for a, cl in ref["pairs"] if a["seed"] in common]) - \
            np.array([ser(cl)[5:].mean() for a, cl in c["pairs"] if a["seed"] in common])
        sa, sc = seed_tests(da), seed_tests(dc)
        pr.append((rx, len(common), da.mean(), dc.mean(), sa["wilcoxon_p"], sc["wilcoxon_p"],
                   (dc.mean() / np.mean([ser(cl)[5:].mean() for a, cl in c["pairs"] if a["seed"] in common]))))
    ha = holm([p[4] for p in pr]) if pr else []
    hc = holm([p[5] for p in pr]) if pr else []
    rows = []
    for (rx, n, dA, dC, pA, pC, rel), qa, qc in zip(pr, ha, hc):
        put(f"cmp.{rx}.n", str(n))
        put(f"cmp.{rx}.att", f_sig(dA))
        put(f"cmp.{rx}.cln", f_sig(dC))
        put(f"cmp.{rx}.clnrel", f"{100 * rel:+.0f}")
        put(f"cmp.{rx}.patt", f_p(qa))
        put(f"cmp.{rx}.pcln", f_p(qc))
        rows.append(f"{RXLABEL[rx]} & {n} & {f_sig(dA)} & {f_p(qa)} & {f_sig(dC)} ({100 * rel:+.0f}\\%) & {f_p(qc)}\\\\")
    write_table("p1_paired", rows)


# ------------------------------------------------------------------------------------ generic groups
def group_rows(folder: str, clean: List[str], spec: List[Tuple[str, str, str]], tab: str, prefix: str) -> None:
    """spec: (run name, label column 1, label column 2)."""
    rows = []
    for name, l1, l2 in spec:
        c = cell(folder, name, clean)
        key = prefix + "." + re.sub(r"[^A-Za-z0-9]", "", name.split("__")[0].replace(prefix, "")) + "." + \
            re.sub(r"[^A-Za-z0-9]", "", name.split("__")[-1])
        put_cell(key, c)
        if c is None:
            continue
        rows.append(f"{l1} & {l2} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])} & "
                    f"{f_rate(c['rej'])} & {f_rate(c['frej'])}\\\\")
    write_table(tab, rows)


def adaptive() -> None:
    spec = [("p3_citadel_unroll10__sir0", "Unroll 10", "Citadel-Rx"), ("p3_citadel_unroll50__sir0", "Unroll 50", "Citadel-Rx"),
            ("p3_citadel_pgd100_r5__sir0", "PGD 100, 5 restarts", "Citadel-Rx"),
            ("p3_citadel_softgate__sir0", "Soft gates (BPDA), unroll 10", "Citadel-Rx"),
            ("p3_citadel_strongest__sir0", "Combined$^{a}$", "Citadel-Rx"), ("p3_online_strongest__sir0", "Combined$^{b}$", "Online DeepSIC"),
            ("p3_citadel_unroll10__sir10", "Unroll 10, SIR 10 dB", "Citadel-Rx"),
            ("p3_citadel_softgate__sir10", "Soft gates, SIR 10 dB", "Citadel-Rx"),
            ("p3_citadel_pgd100_r5__sir10", "PGD 100 x 5, SIR 10 dB", "Citadel-Rx"),
            ("p3_citadel_M1__sir0", "Rank-1 physical", "Citadel-Rx"), ("p3_online_deepsic_M1__sir0", "Rank-1 physical", "Online DeepSIC"),
            ("p3_citadel_M2__sir0", "Rank-2 physical", "Citadel-Rx"), ("p3_online_deepsic_M2__sir0", "Rank-2 physical", "Online DeepSIC"),
            ("p3_citadel_M2_csi0.1__sir0", "Rank-2, CSI error 0.1", "Citadel-Rx"),
            ("p3_online_deepsic_M2_csi0.1__sir0", "Rank-2, CSI error 0.1", "Online DeepSIC"),
            ("p3_citadel_greybox__sir0", "Grey-box (Citadel-Rx surrogate)", "Citadel-Rx"),
            ("p3_citadel_query__sir0", "Query (NACK feedback)", "Citadel-Rx"), ("p3_online_deepsic_query__sir0", "Query (NACK feedback)", "Online DeepSIC"),
            ("p3_citadel_sigbudget__sir0", "Signal-referenced budget", "Citadel-Rx"),
            ("p3_online_deepsic_sigbudget__sir0", "Signal-referenced budget", "Online DeepSIC")]
    rows = []
    for name, l1, l2 in spec:
        folder = "rev_p3_adaptive" if any(t in name for t in ("unroll", "softgate", "strongest", "pgd")) else \
            ("rev_p3_phys" if "_M" in name else "rev_p3_know")
        c = cell(folder, name, ["rev_clean"])
        put_cell("atk." + re.sub(r"[^A-Za-z0-9]", "", name.replace("p3_", "")), c)
        if c is not None:
            rows.append(f"{l1} & {l2} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])} & {f_rate(c['rej'])}\\\\")
    write_table("attacks", rows)


def refdeg() -> None:
    rows = []
    for a in ("0.25", "0.75"):
        for ab in ("", "_abstain"):
            name = f"p3_a4_a{a}{ab}__sir0"
            c = cell("rev_p3_refdeg", name, ["rev_clean_cloud"])
            put_cell("a4." + a.replace(".", "") + ab.replace("_", ""), c)
            if c is not None:
                rows.append(f"$\\alpha={a}$ & {'abstain' if ab else '--'} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & "
                            f"{f_ser(c['raw'])} & {f_rate(c['rej'])}\\\\")
        jam = [r for r in load("rev_p3_refdeg") if r["name"] == f"p3_jamctl_a{a}__sir0"]
        tw = twins(["rev_clean_cloud"])
        ex = []
        for r in jam:
            c = tw.get((mkey(r["config"]), r["seed"]))
            if c is not None:
                w0, w1 = window(r)
                ex.append(float(np.mean(ser(r)[w0:w1] - ser(c)[w0:w1])))
        if ex:
            put("a4.jam" + a.replace(".", ""), f_sig(float(np.mean(ex))))
            rows.append(f"jamming only, $1-\\alpha={1 - float(a):g}$ of the power & -- & {len(ex)} & {f_sig(float(np.mean(ex)))} & -- & --\\\\")
    write_table("refdeg", rows)


def deny() -> None:
    """B3 / D6: abrupt change at block 50, with or without the channel-estimate spoof on blocks 40-89."""
    rows = []
    recs = load("rev_p3_deny")
    for rx in ("citadel", "citadel_streak", "online_deepsic", "lmmse"):
        def runs(tag):
            return {r["seed"]: r for r in recs if r["name"] == f"p3_{rx}_{tag}"}
        none, den, drop = runs("redraw__none"), runs("redraw__deny"), runs("snrdrop__none")
        s = sorted(set(none) & set(den))
        if not s:
            continue
        after = float(np.mean([ser(none[i])[50:].mean() for i in s]))
        ex = float(np.mean([(ser(den[i])[40:90] - ser(none[i])[40:90]).mean() for i in s]))
        post = float(np.mean([(ser(den[i])[90:] - ser(none[i])[90:]).mean() for i in s]))
        fr = [rejections(none[i], None, skip=50) for i in s]
        frk = (sum(k for k, _ in fr), sum(n for _, n in fr))
        dr = float(np.mean([ser(drop[i])[50:].mean() for i in sorted(drop)])) if drop else float("nan")
        lab = {"citadel": "Citadel-Rx", "citadel_streak": "Citadel-Rx + streak recovery", "online_deepsic": "Online DeepSIC",
               "lmmse": "LMMSE"}[rx]
        k = "deny." + rx.replace("_", "")
        put(k + ".after", f_ser(after)); put(k + ".excess", f_sig(ex)); put(k + ".post", f_sig(post))
        put(k + ".frej", f_rate(frk)); put(k + ".drop", f_ser(dr)); put(k + ".n", str(len(s)))
        rows.append(f"{lab} & {len(s)} & {f_ser(after)} & {f_rate(frk) if rx.startswith('citadel') else '--'} & {f_sig(ex)} & {f_sig(post)} & {f_ser(dr)}\\\\")
    write_table("deny", rows)


def baselines() -> None:
    names = ["lmmse", "lmmse_robust", "rollback_allgates", "l2sp_deepsic", "ema_deepsic", "ewc_deepsic", "dd_deepsic",
             "cotta_deepsic", "eata_deepsic", "drift_deepsic", "titans_single", "medbn_deepsic"]
    rows = []
    for rx in names:
        c = cell("rev_p2", f"p2_{rx}__wb_sir0", ["rev_clean"])
        put_cell("p2." + rx.replace("_", ""), c)
        if c is not None:
            rows.append(f"{RXLABEL[rx]} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])}\\\\")
    for w in ("0.001", "0.1", "1"):
        c = cell("rev_p2", f"p2_l2sp_w{w}__wb_sir0", ["rev_clean"])
        put_cell("p2.l2spw" + w.replace(".", ""), c)
        if c is not None:
            rows.append(f"L2-SP, weight {w} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])}\\\\")
    write_table("baselines", rows)


def ablations() -> None:
    lab = [("K1", "$K=1$ (gates, rollback, no slow level)"), ("K2", "$K=2$, $C=(1,8)$"), ("nogate", "No gates"),
           ("g1", "G1 only"), ("g2", "G2 only"), ("g3", "G3 only"), ("no_anchor", "No anchor ($\\lambda=0$)"),
           ("no_fallback", "No fallback"), ("no_select", "No CRC selection"), ("no_clip", "No consolidation clip")]
    rows = []
    ref = cell("rev_p1", "p1_citadel__wb_sir0", ["rev_clean"])
    if ref is not None:
        sel = [p for p in ref["pairs"] if p[0]["seed"] in (10, 11)]
        if sel:
            ex = np.mean([(ser(a)[window(a)[0]:window(a)[1]] - ser(c)[window(a)[0]:window(a)[1]]).mean() for a, c in sel])
            k1 = sum(rejections(a, True)[0] for a, _ in sel); n1 = sum(rejections(a, True)[1] for a, _ in sel)
            k0 = sum(rejections(c, None)[0] for _, c in sel); n0 = sum(rejections(c, None)[1] for _, c in sel)
            rows.append(f"Complete receiver & {len(sel)} & {f_ser(np.mean([ser(c)[5:].mean() for _, c in sel]))} & {f_sig(ex)} & "
                        f"{f_rate((k1, n1))} & {f_rate((k0, n0))}\\\\")
    for tag, l in lab:
        c = cell("rev_p5_ablate", f"p5_{tag}__wb_sir0", ["rev_clean"])
        put_cell("ab." + tag.replace("_", ""), c)
        if c is not None:
            rows.append(f"{l} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_rate(c['rej'])} & {f_rate(c['frej'])}\\\\")
    write_table("ablations", rows)


def scenarios() -> None:
    rows = []
    spec = []
    for nd in (1000, 5000):
        for npil in (32, 100):
            for rx in ("citadel", "online_deepsic"):
                spec.append(("rev_p6_blocklen", f"p6_{rx}_nd{nd}_np{npil}__sir0", f"$N_d={nd}$, $N_p={npil}$", RXLABEL[rx]))
    for rx in ("citadel", "online_deepsic", "lmmse"):
        spec.append(("rev_p6_arrays", f"p6_{rx}_4x16__sir0", "$U\\times N=4\\times16$", RXLABEL[rx]))
    for rx in ("citadel", "online_deepsic", "lmmse"):
        spec.append(("rev_p6_qam16", f"p6_{rx}_qam16__sir0", "16-QAM, 20 dB", RXLABEL[rx]))
    spec += [("rev_p6_deploy", "p6_citadel_steps30__sir0", "30 fast steps", "Citadel-Rx"),
             ("rev_p6_deploy", "p6_citadel_last_layer__sir0", "last layer adaptable", "Citadel-Rx")]
    for folder, name, l1, l2 in spec:
        c = cell(folder, name, ["rev_clean"])
        put_cell("sc." + re.sub(r"[^A-Za-z0-9]", "", name.replace("p6_", "").split("__")[0]), c)
        if c is not None:
            mt = np.mean([np.mean([b["t_rx"] for b in a["blocks"]]) for a, _ in c["pairs"]])
            rows.append(f"{l1} & {l2} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & "
                        f"{f_rate(c['rej']) if l2 == 'Citadel-Rx' else '--'} & {mt:.1f}\\\\")
    write_table("scenarios", rows)


def sensitivity() -> None:
    pars = [("g1_slack", "G1 slack $\\tau_1-1$"), ("g3_slack", "G3 slack $\\tau_3-1$"), ("tighten", "tightening $1-$floor"),
            ("soft_width", "soft-step width"), ("prox_lambda", "proximal weight $\\lambda$"), ("delta", "delta decay $\\delta$"),
            ("crc_rel", "CRC relative slack $\\rho$"), ("eta3", "level-3 rate $\\eta^{(3)}$")]
    rows = []
    for p, l in pars:
        cs = [cell("rev_p7_sens", f"p7_{p}_x{m}__wb_sir0", ["rev_clean"]) for m in ("0.5", "2")]
        if all(c is None for c in cs):
            continue
        parts = []
        for c in cs:
            parts += ["--"] * 3 if c is None else [f_ser(c["clean_all"]), f_sig(c["excess"]), f_rate(c["frej"])]
        rows.append(f"{l} & " + " & ".join(parts) + "\\\\")
    write_table("sensitivity", rows)


def tuning() -> None:
    write_table("calibration", [])
    for fn, pre in (("revision_frozen_baselines.json", "tune"), ("revision_frozen_citadel.json", "cal")):
        p = os.path.join(ROOT, "configs", fn)
        if not os.path.exists(p):
            continue
        fz = json.load(open(p))
        if pre == "tune":
            for rx in ("online_deepsic", "meta_deepsic"):
                if fz.get(rx, {}).get("overrides"):
                    d = dict(o.split("=") for o in fz[rx]["overrides"])
                    k = f"tune.{rx.replace('_', '')}"
                    put(k + ".epochs", d["receiver.epochs"]); put(k + ".lr", f"{float(d['receiver.lr']):g}")
                    put(k + ".wd", f"{float(d['receiver.weight_decay']):g}"); put(k + ".ser", f_ser(fz[rx]["cal_clean_ser"]))
        else:
            put("cal.chosen", {"rel_select_clip": "relative slack", "derived_select_clip": "derived slack",
                               "rel_select_clip_anchorstart": "relative slack with anchor start"}.get(fz["chosen"], fz["chosen"]))
            rows = []
            for tag, v in fz["candidates"].items():
                k = "cal." + tag.replace("_", "")
                put(k + ".score", f_ser(v["score"])); put(k + ".clean", f_ser(v["clean_ser"]))
                put(k + ".frej", f"{v['clean_false_rejection']:.3f}")
                nice = {"rel_select_clip": "Relative slack", "derived_select_clip": "Derived slack",
                        "rel_select_clip_anchorstart": "Relative slack, anchor start"}.get(tag, tag.replace("_", " "))
                rows.append(f"{nice} & {f_ser(v['clean_ser'])} & {f_ser(v['a1_sir0_window_ser'])} & "
                            f"{f_ser(v['cost2100_sir0_window_ser'])} & {v['clean_false_rejection']:.3f} & {f_ser(v['score'])}"
                            + (" & chosen" if tag == fz["chosen"] else " & ") + "\\\\")
            write_table("calibration", rows)


def retention() -> None:
    """P4 (A1, paired snapshots) and P8 (A3*, 400 blocks): paired displacement of the slowest level
    against the design bound 2 kappa_K n(W) and the structural bound (Theorem 1). Needs the
    paired_disp fields that scripts/paired_disp.py stores in the attacked records."""
    from paired_disp import extended_bound
    rows = []
    specs = [("rev_p4_paired", f"p4_citadel__sir{s}", f"A1, SIR {s} dB", f"ret.p4sir{s}") for s in (0, 10)] + \
        [("rev_p8_drift", f"p8_a3star_e{e}{g}__long", f"A3$^\\ast$, {int(float(e) * 100)}\\%" + (", guard" if g else ""),
          "ret.p8e" + e.replace(".", "") + g.replace("_", ""))
         for e, g in (("0.05", ""), ("0.2", ""), ("0.5", ""), ("0.5", "_guard"))]
    for folder, name, lab, key in specs:
        # P4 ran on the cloud machine, P8 on the laptop (after the cloud machine was stopped)
        c = cell(folder, name, ["rev_clean", "rev_clean_cloud"])
        put_cell(key, c)
        if c is None or not all("paired_disp" in a for a, _ in c["pairs"]):
            continue
        DK, des, ext, anc = [], [], [], []
        for a, cl in c["pairs"]:
            cfg = a["config"]["citadel"]
            K = len(a["paired_disp"])
            w0, w1 = window(a)
            n = len(a["paired_disp"][0])
            w1 = min(w1, n)
            CK = cfg["periods"][K - 1]
            ncons = int(np.floor(w1 / CK) - np.floor(w0 / CK))
            clip = cfg.get("consol_clip") or []
            DK.append(max(a["paired_disp"][K - 1][w0:w1]))
            des.append(2 * clip[K - 2] * ncons if len(clip) >= K - 1 else float("nan"))
            eta_K = cfg["level_lr"][K - 2]
            ext.append(float(extended_bound(a, cl, np.array(a["paired_disp"][K - 2]), eta_K, K, w0, w1).max()))
            anc.append(max(a["paired_anchor"][w0:w1]))
        put(key + ".DK", f"{np.mean(DK):.2f}"); put(key + ".design", f"{np.mean(des):.1f}")
        put(key + ".struct", f"{np.mean(ext):.1f}"); put(key + ".anchor", f"{np.mean(anc):.2f}")
        rows.append(f"{lab} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_rate(c['rej'])} & {np.mean(DK):.2f} & "
                    f"{np.mean(ext):.1f} & {np.mean(des):.1f}\\\\")
    write_table("retention", rows)


def campaign_size() -> None:
    n = 0
    for d in glob.glob(os.path.join(RES, "rev_*")):
        n += len(glob.glob(os.path.join(d, "*.json")))
    put("camp.records", f"{n:,}".replace(",", "\\,"))


def main() -> None:
    for name in NCOLS:   # every table file exists, so the paper always compiles
        write_table(name, [])
    p1(); adaptive(); refdeg(); deny(); baselines(); ablations(); scenarios(); sensitivity(); tuning(); retention(); campaign_size()
    lines = ["% generated by scripts/revision_numbers.py -- do not edit",
             "\\makeatletter"]
    for k in sorted(V):
        lines.append(f"\\@namedef{{val@{k}}}{{{V[k]}}}")
    lines.append("\\makeatother")
    open(OUT_NUM, "w").write("\n".join(lines) + "\n")
    tex = open(os.path.join(ROOT, "paper", "main.tex")).read() if os.path.exists(os.path.join(ROOT, "paper", "main.tex")) else ""
    used = sorted(set(re.findall(r"\\val\{([^}]+)\}", tex)))
    missing = [k for k in used if k not in V]
    print(f"{len(V)} numbers written to {os.path.relpath(OUT_NUM, ROOT)}; tables in {os.path.relpath(OUT_TAB, ROOT)}")
    print(f"used in main.tex: {len(used)}; missing (shown red in the PDF): {len(missing)}")
    for k in missing:
        print("   missing:", k)


if __name__ == "__main__":
    main()
