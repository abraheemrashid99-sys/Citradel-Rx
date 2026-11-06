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
sys.path.insert(0, ROOT)   # citadel_rx (config defaults for twin matching), also when run as a script
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
            except Exception:
                continue
            # 38.901 UMi runs made with the first drop files (5 ms blocks, replaced before any analysis)
            if r["config"]["channel"].get("model") == "umi" and (r["summary"].get("umi_drop") or {}).get("t_block") != 0.001:
                print(f"   left out {os.path.relpath(fn, RES)}: UMi drop without 1 ms blocks (re-run it)")
                continue
            if r["config"]["frame"].get("coding") == "ldpc" and not r["summary"].get("ldpc_scrambled"):
                continue   # first coded-scene version without NR scrambling (superseded by d23s_*)
            out.append(r)
        _cache[folder] = out
    return _cache[folder]


def mkey(cfg: dict) -> str:
    """Settings that a clean twin must share with its attacked run. Options added to the code after a
    record was written are filled in at their defaults, so records of different code versions match."""
    from citadel_rx.config import Config   # ROOT is on sys.path (top of this file); a failure here must not pass silently
    try:
        cfg = Config.from_dict(cfg).to_dict()
    except KeyError:   # a record with an option the code no longer has: compare as written
        pass
    return json.dumps({k: v for k, v in cfg.items() if k not in ("attack", "run")}, sort_keys=True)


MACHINE_GROUP = {"cloud": "cloud", "cloudx": "cloud", "cloudy": "cloud", "cloudz": "cloud", "windows": "windows", "helper": "helper"}


CLOUD_V1 = {"rev_p4_paired", "rev_p3_refdeg", "rev_cal_citadel", "rev_clean_cloud"}   # first campaign on the cloud
# every clean-twin folder any driver writes (scripts/revision_fast.py CLEAN_DIR), so a new machine cannot be missed
CLEAN_ALL = ["rev_clean", "rev_clean_cloud", "rev_clean_win", "rev_clean_helper", "rev_clean_local"] + [f"rev_clean_pool{i}" for i in range(1, 13)]


def machine(r: dict):
    """Which machine ran a record: second-campaign records carry it; a first-campaign record is placed
    by its folder (the cloud ran P4, the reference-degradation runs and the calibration; the MacBook
    the rest)."""
    if r.get("machine"):
        return MACHINE_GROUP.get(r["machine"], r["machine"])
    folder = os.path.basename(os.path.dirname(r.get("_file", "")))
    return "cloud" if folder in CLOUD_V1 else "mac"


def twins(clean_folders: List[str]) -> Dict[Tuple[str, int], dict]:
    """(settings, seed) -> {machine: clean record}; a run is paired only with a twin from its own machine."""
    idx: Dict[Tuple[str, int], dict] = {}
    for f in clean_folders:
        for r in load(f):
            if r["config"]["attack"]["type"] == "none":
                idx.setdefault((mkey(r["config"]), r["seed"]), {})[machine(r)] = r
    return idx


def twin_of(tw: Dict[Tuple[str, int], dict], a: dict) -> Optional[dict]:
    return tw.get((mkey(a["config"]), a["seed"]), {}).get(machine(a))


def window(r: dict) -> Tuple[int, int]:
    w = r["summary"].get("attack_window")
    if w:
        return int(w[0]), int(w[1])
    return int(r["config"]["attack"]["start_block"]), int(r["config"]["attack"]["end_block"])


def ser(r: dict, metric: str = "ser") -> np.ndarray:
    return np.array([b.get(metric, np.nan) for b in r["blocks"]], dtype=float)


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
def cell(folder: str, name: str, clean: List[str], metric: str = "ser") -> Optional[dict]:
    """Attacked configuration `name` in `folder`, paired with its clean twins (same seed, settings, machine)."""
    tw = twins(clean)
    prs = []
    for r in load(folder):
        if r["name"] == name and r["config"]["attack"]["type"] != "none":
            c = twin_of(tw, r)
            if c is not None:
                prs.append((r, c))
    if not prs:
        return None
    prs.sort(key=lambda p: p[0]["seed"])
    win, post, raw, cwin = [], [], [], []
    k1 = n1 = k0 = n0 = 0
    for a, c in prs:
        w0, w1 = window(a)
        sa, sc = ser(a, metric), ser(c, metric)
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
            "clean_all": float(np.mean([ser(c, metric)[5:].mean() for _, c in prs]))}


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
    """Interval without plus signs (a minus sign still shows): it follows a signed point estimate."""
    return f"[{f_ser(ci[0])}, {f_ser(ci[1])}]"


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
    if c["rej"][1]:
        put(prefix + ".rejci", "[{:.3f}, {:.3f}]".format(*wilson(*c["rej"])))
    if c["frej"][1]:
        put(prefix + ".frejci", "[{:.3f}, {:.3f}]".format(*wilson(*c["frej"])))
    put(prefix + ".frej", f_rate(c["frej"]))
    put(prefix + ".frejk", f"{c['frej'][0]}/{c['frej'][1]}")
    put(prefix + ".rejpct", f"{100 * c['rej'][0] / max(1, c['rej'][1]):.0f}")
    put(prefix + ".frejpct", f"{100 * c['frej'][0] / max(1, c['frej'][1]):.1f}")
    put(prefix + ".p", f_p(c["wilcoxon_p"]))
    put(prefix + ".minp", f_p(c["min_p"]))
    put(prefix + ".clean", f_ser(c["clean_all"]))
    put(prefix + ".decor", str(c["decor"]))
    put(prefix + ".smin", f_sig(min(c["seed_means"]))); put(prefix + ".smax", f_sig(max(c["seed_means"])))
    put(prefix + ".spos", str(sum(x > 0 for x in c["seed_means"])))


NCOLS = {"p1": 9, "p1_paired": 6, "attacks": 6, "refdeg": 6, "deny": 7, "baselines": 5, "ablations": 6, "scenarios": 7,
         "sensitivity": 6, "calibration": 7, "retention": 7,
         "deny2": 7, "peruser": 6, "joint": 6, "umi": 7}


def write_table(name: str, rows: List[str], pending: str = "runs pending") -> None:
    os.makedirs(OUT_TAB, exist_ok=True)
    body = "\n".join(rows) if rows else f"\\multicolumn{{{NCOLS.get(name, 2)}}}{{@{{}}l}}{{\\emph{{({pending})}}}}\\\\"
    open(os.path.join(OUT_TAB, name + ".tex"), "w").write(body + "\n")


# ------------------------------------------------------------------------------------ P1 headline
P1RX = ["citadel", "online_deepsic", "meta_deepsic", "crc_rollback", "online_select"]
P1CELLS = ["wb_sir-10", "wb_sir0", "wb_sir10", "tr_sir0", "jam"]
CK = {"wb_sir-10": "wbm10", "wb_sir0": "wb0", "wb_sir10": "wb10", "tr_sir0": "tr0", "jam": "jam"}


def p1() -> None:
    cells = {(rx, cl): cell("rev_p1", f"p1_{rx}__{cl}", CLEAN_ALL) for rx in P1RX for cl in P1CELLS}
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
    jam = [cells[(rx, "jam")] for rx in P1RX if cells[(rx, "jam")] is not None]
    if jam:
        put("p1.jam.exmin", f_sig(min(c["excess"] for c in jam))); put("p1.jam.exmax", f_sig(max(c["excess"] for c in jam)))
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
        put(f"cmp.{rx}.clnrel", f"${100 * rel:+.0f}$")
        put(f"cmp.{rx}.clnpct", f"{abs(100 * rel):.0f}")
        put(f"cmp.{rx}.patt", f_p(qa))
        put(f"cmp.{rx}.pcln", f_p(qc))
        rows.append(f"{RXLABEL[rx]} & {n} & {f_sig(dA)} & {f_p(qa)} & {f_sig(dC)} (${100 * rel:+.0f}$\\%) & {f_p(qc)}\\\\")
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
            ("b1_citadel_M2_csi0.1__sir0", "Rank-2, CSI error 0.1", "Citadel-Rx"),
            ("b1_online_deepsic_M2_csi0.1__sir0", "Rank-2, CSI error 0.1", "Online DeepSIC"),
            ("b1_citadel_M2_csi0.01__sir0", "Rank-2, CSI error 0.01", "Citadel-Rx"),
            ("b1_online_deepsic_M2_csi0.01__sir0", "Rank-2, CSI error 0.01", "Online DeepSIC"),
            ("b1_citadel_M4__sir0", "Rank-4 physical", "Citadel-Rx"), ("b1_online_deepsic_M4__sir0", "Rank-4 physical", "Online DeepSIC"),
            ("p3_citadel_greybox__sir0", "Grey-box (Citadel-Rx surrogate)", "Citadel-Rx"),
            ("p3_citadel_query__sir0", "Query (NACK feedback)", "Citadel-Rx"), ("p3_online_deepsic_query__sir0", "Query (NACK feedback)", "Online DeepSIC"),
            ("p3_citadel_sigbudget__sir0", "Signal-referenced budget", "Citadel-Rx"),
            ("p3_online_deepsic_sigbudget__sir0", "Signal-referenced budget", "Online DeepSIC")]
    rows, rng = [], {}
    for name, l1, l2 in spec:
        folder = "rev_p3_adaptive" if any(t in name for t in ("unroll", "softgate", "strongest", "pgd")) else \
            ("rev2_phys" if name.startswith("b1_") else ("rev_p3_phys" if "_M" in name else "rev_p3_know"))
        c = cell(folder, name, CLEAN_ALL)
        put_cell("atk." + re.sub(r"[^A-Za-z0-9]", "", name.replace("p3_", "")), c)
        if c is not None:
            rows.append(f"{l1} & {l2} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])} & {f_rate(c['rej'])}\\\\")
            grp = ("cit" if "citadel" in name else "onl") + ("10" if name.endswith("sir10") else "0")
            rng.setdefault(grp, []).append(c)
    for grp, cs in rng.items():   # ranges quoted in the text
        put(f"atk.{grp}.exmin", f_sig(min(c["excess"] for c in cs))); put(f"atk.{grp}.exmax", f_sig(max(c["excess"] for c in cs)))
        put(f"atk.{grp}.rejmin", f_rate(min((c["rej"] for c in cs), key=lambda kn: kn[0] / max(1, kn[1]))))
        put(f"atk.{grp}.n", str(len(cs)))
    write_table("attacks", rows)


def refdeg() -> None:
    rows = []
    for a in ("0.25", "0.75"):
        variants = [("rev_p3_refdeg", f"p3_a4_a{a}__sir0", "--", "a4." + a.replace(".", "")),
                    ("rev_p3_refdeg", f"p3_a4_a{a}_abstain__sir0", "abstain, 1st", "a4." + a.replace(".", "") + "abstain"),
                    ("rev2_abstain_v3", f"h3v_a4_a{a}__sir0", "abstain", None)]
        if a == "0.25":
            variants.append(("rev2_a4_pgd20", "b2_a4_a0.25_pgd20__sir0", "20 PGD steps", None))
        for folder, name, state, key in variants:
            c = cell(folder, name, CLEAN_ALL)
            if key:
                put_cell(key, c)
            if c is not None:
                rows.append(f"$\\alpha={a}$ & {state} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & "
                            f"{f_ser(c['raw'])} & {f_rate(c['rej'])}\\\\")
        jam = [r for r in load("rev_p3_refdeg") if r["name"] == f"p3_jamctl_a{a}__sir0"]
        tw = twins(CLEAN_ALL)
        ex = []
        for r in jam:
            c = twin_of(tw, r)
            if c is not None:
                w0, w1 = window(r)
                ex.append(float(np.mean(ser(r)[w0:w1] - ser(c)[w0:w1])))
        if ex:
            put("a4.jam" + a.replace(".", ""), f_sig(float(np.mean(ex))))
            rows.append(f"jamming & ${1 - float(a):g}$ of power & {len(ex)} & {f_sig(float(np.mean(ex)))} & -- & --\\\\")
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
        sd_ = sorted(set(drop) & set(s))   # the same seeds as the other columns
        dr = float(np.mean([ser(drop[i])[50:].mean() for i in sd_])) if sd_ else float("nan")
        lab = {"citadel": "Citadel-Rx", "citadel_streak": "+ streak recovery", "online_deepsic": "Online DeepSIC",
               "lmmse": "LMMSE"}[rx]
        k = "deny." + rx.replace("_", "")
        put(k + ".after", f_ser(after)); put(k + ".excess", f_sig(ex)); put(k + ".post", f_sig(post))
        put(k + ".frej", f_rate(frk)); put(k + ".drop", f_ser(dr)); put(k + ".n", str(len(s)))
        rows.append(f"{lab} & {len(s)} & {f_ser(after)} & {f_rate(frk) if rx.startswith('citadel') else '--'} & {f_sig(ex)} & {f_sig(post)} & {f_ser(dr)}\\\\")
        if rx.startswith("citadel"):
            rj = [rejections(den[i], True, skip=0) for i in s]
            put(k + ".rejatt", f_rate((sum(a for a, _ in rj), sum(b for _, b in rj))))
            put(k + ".rejattmin", f"{min(a / b for a, b in rj):.2f}"); put(k + ".rejattmax", f"{max(a / b for a, b in rj):.2f}")
    plain = {(r["name"].replace("p3_citadel_", ""), r["seed"]): ser(r) for r in recs if r["name"].startswith("p3_citadel_") and "streak" not in r["name"]}
    strk = {(r["name"].replace("p3_citadel_streak_", ""), r["seed"]): ser(r) for r in recs if r["name"].startswith("p3_citadel_streak_")}
    common = sorted(set(plain) & set(strk))
    if common:
        put("deny.streak.ndiff", str(sum(bool(np.any(np.abs(plain[c] - strk[c]) > 1e-12)) for c in common)))
        put("deny.streak.nruns", str(len(common)))
    srec = [r for r in recs if r["name"].startswith("p3_citadel_streak_")]
    put("deny.streak.attempts", str(sum("recovery_pass" in b for r in srec for b in r["blocks"])))
    put("deny.streak.accepted", str(sum(bool(b.get("recovered")) for r in srec for b in r["blocks"])))
    write_table("deny", rows)


def baselines() -> None:
    names = ["lmmse", "lmmse_robust", "rollback_allgates", "l2sp_deepsic", "ema_deepsic", "ewc_deepsic", "dd_deepsic",
             "cotta_deepsic", "eata_deepsic", "drift_deepsic", "titans_single", "medbn_deepsic"]
    rows = []
    for rx in names:
        c = cell("rev_p2", f"p2_{rx}__wb_sir0", CLEAN_ALL)
        put_cell("p2." + rx.replace("_", ""), c)
        if c is not None:
            rows.append(f"{RXLABEL[rx]} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])}\\\\")
    exs = [cell("rev_p2", f"p2_{rx}__wb_sir0", CLEAN_ALL) for rx in names]
    exs = [c for c in exs if c is not None]
    if exs:
        put("p2.exmin", f_sig(min(c["excess"] for c in exs))); put("p2.exmax", f_sig(max(c["excess"] for c in exs)))
        put("p2.n", str(len(exs)))
    for w in ("0.001", "0.1", "1"):
        c = cell("rev_p2", f"p2_l2sp_w{w}__wb_sir0", CLEAN_ALL)
        put_cell("p2.l2spw" + w.replace(".", ""), c)
        if c is not None:
            rows.append(f"L2-SP, weight {w} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_ser(c['raw'])}\\\\")
    write_table("baselines", rows)


def ablations() -> None:
    lab = [("K1", "$K=1$ (gates, rollback, no slow level)"), ("K2", "$K=2$, $C=(1,8)$"), ("nogate", "No gates"),
           ("g1", "G1 only"), ("g2", "G2 only"), ("g3", "G3 only"), ("no_anchor", "No anchor ($\\lambda=0$)"),
           ("no_fallback", "No fallback"), ("no_select", "No CRC selection"), ("no_clip", "No consolidation clip")]
    rows = []
    ref = cell("rev_p1", "p1_citadel__wb_sir0", CLEAN_ALL)
    if ref is not None:
        ab_seeds = {r["seed"] for r in load("rev_p5_ablate")} or {10, 11}   # the seeds the ablations were run on
        sel = [p for p in ref["pairs"] if p[0]["seed"] in ab_seeds]
        if sel:
            ex = np.mean([(ser(a)[window(a)[0]:window(a)[1]] - ser(c)[window(a)[0]:window(a)[1]]).mean() for a, c in sel])
            k1 = sum(rejections(a, True)[0] for a, _ in sel); n1 = sum(rejections(a, True)[1] for a, _ in sel)
            k0 = sum(rejections(c, None)[0] for _, c in sel); n0 = sum(rejections(c, None)[1] for _, c in sel)
            rows.append(f"Complete receiver & {len(sel)} & {f_ser(np.mean([ser(c)[5:].mean() for _, c in sel]))} & {f_sig(ex)} & "
                        f"{f_rate((k1, n1))} & {f_rate((k0, n0))}\\\\")
            put("ab.complete.clean", f_ser(np.mean([ser(c)[5:].mean() for _, c in sel]))); put("ab.complete.excess", f_sig(ex))
            put("ab.complete.rej", f_rate((k1, n1))); put("ab.complete.frej", f_rate((k0, n0)))
    for tag, l in lab:
        c = cell("rev_p5_ablate", f"p5_{tag}__wb_sir0", CLEAN_ALL)
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
    for rx in ("citadel", "online_deepsic", "lmmse"):   # SED profile stretched over the array (channel.sed_stretch)
        spec.append(("rev2_arrays_stretch", f"p6s_{rx}_4x16s__sir0", "$U\\times N=4\\times16$", RXLABEL[rx]))
    for rx in ("citadel", "online_deepsic", "lmmse"):
        spec.append(("rev2_arrays_stretch", f"d4s_{rx}_8x32s__sir0", "$U\\times N=8\\times32$", RXLABEL[rx]))
    for rx in ("citadel", "online_deepsic", "lmmse"):
        spec.append(("rev_p6_qam16", f"p6_{rx}_qam16__sir0", "16-QAM, 20 dB", RXLABEL[rx]))
    spec += [("rev_p6_deploy", "p6_citadel_steps30__sir0", "30 fast steps", "Citadel-Rx"),
             ("rev_p6_deploy", "p6_citadel_last_layer__sir0", "last layer adaptable", "Citadel-Rx")]
    bl = {}
    for folder, name, l1, l2 in spec:
        c = cell(folder, name, CLEAN_ALL)
        base = name.replace("p6s_", "").replace("d4s_", "").replace("p6_", "").replace("d4_", "").split("__")[0]
        put_cell("sc." + re.sub(r"[^A-Za-z0-9]", "", base.replace("4x16s", "4x16").replace("8x32s", "8x32")), c)
        if c is not None and folder == "rev_p6_blocklen":
            bl.setdefault("cit" if "citadel" in name else "onl", []).append(c)
        if c is not None:
            mac = [a for a, _ in c["pairs"] if machine(a) == "mac"]
            mt = np.mean([np.mean([b["t_rx"] for b in a["blocks"]]) for a in mac]) if mac else float("nan")
            rows.append(f"{l1} & {l2} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & "
                        f"{f_rate(c['rej']) if l2 == 'Citadel-Rx' else '--'} & {'--' if not np.isfinite(mt) else f'{mt:.1f}'}\\\\")
    for g, cs in bl.items():
        put(f"sc.bl.{g}exmin", f_sig(min(c["excess"] for c in cs))); put(f"sc.bl.{g}exmax", f_sig(max(c["excess"] for c in cs)))
        put(f"sc.bl.{g}rejmin", f_rate(min((c["rej"] for c in cs), key=lambda kn: kn[0] / max(1, kn[1]))))
    write_table("scenarios", rows)


def sensitivity() -> None:
    """E5: one setting at a time at x0.5, x0.75, x1.5 and x2 (seed 10; x0.5 and x2 from the first campaign)."""
    allc = []
    pars = [("g1_slack", "G1 slack $\\tau_1-1$"), ("g3_slack", "G3 slack $\\tau_3-1$"), ("tighten", "tightening $1-$floor"),
            ("soft_width", "soft-step width"), ("prox_lambda", "proximal weight $\\lambda$"), ("delta", "delta decay $\\delta$"),
            ("crc_rel", "CRC relative slack $\\rho$"), ("eta3", "level-3 rate $\\eta^{(3)}$")]
    rows = []
    for p, l in pars:
        cs = [cell("rev_p7_sens", f"p7_{p}_x{m}__wb_sir0", CLEAN_ALL) for m in ("0.5", "0.75", "1.5", "2")]
        if all(c is None for c in cs):
            continue
        fr = [c["frej"][0] / max(1, c["frej"][1]) for c in cs if c is not None]
        rows.append(f"{l} & " + " & ".join("--" if c is None else f_sig(c["excess"]) for c in cs) + f" & {max(fr):.3f}\\\\")
        allc += [(c, l) for c in cs if c is not None]
    if allc:
        put("sens.exmin", f_sig(min(c["excess"] for c, _ in allc))); put("sens.exmax", f_sig(max(c["excess"] for c, _ in allc)))
        put("sens.cleanmin", f_ser(min(c["clean_all"] for c, _ in allc))); put("sens.cleanmax", f_ser(max(c["clean_all"] for c, _ in allc)))
        cmax, lmax = max(allc, key=lambda cl: cl[0]["frej"][0] / max(1, cl[0]["frej"][1]))
        put("sens.frejmax", f_rate(cmax["frej"])); put("sens.n", str(len(allc)))
    grid = [cell("rev2_sens_grid", f"grid_crc{m1}_g1{m2}__wb_sir0", CLEAN_ALL) for m1 in ("0.5", "2") for m2 in ("0.5", "2")]
    grid = [c for c in grid if c is not None]
    if grid:   # two settings changed together (corners of CRC slack x G1 slack)
        put("sens.grid.n", str(len(grid)))
        put("sens.grid.exmin", f_sig(min(c["excess"] for c in grid))); put("sens.grid.exmax", f_sig(max(c["excess"] for c in grid)))
        put("sens.grid.frejmax", f"{max(c['frej'][0] / max(1, c['frej'][1]) for c in grid):.3f}")
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
    rows, p8 = [], []
    specs = [("rev_p4_paired", f"p4_citadel__sir{s}", f"A1, SIR {s} dB", f"ret.p4sir{s}") for s in (0, 10)] + \
        [("rev_p8_drift", f"p8_a3star_e{e}{g}__long", f"A3$^\\ast$, {int(float(e) * 100)}\\%" + (", guard" if g else ""),
          "ret.p8e" + e.replace(".", "") + g.replace("_", ""))
         for e, g in (("0.05", ""), ("0.2", ""), ("0.5", ""), ("0.5", "_guard"))] + \
        [("rev2_drift_guard", "h5_a3star_e0.05_guard__long", "A3$^\\ast$, 5\\%, guard", "ret.h5"),
         ("rev2_drift_long", "b5_a3star_e0.05__long1000", "A3$^\\ast$, 5\\%, 900 blocks", "ret.b5")]
    for folder, name, lab, key in specs:
        # P4 ran on the cloud machine, P8 on the laptop (after the cloud machine was stopped)
        c = cell(folder, name, CLEAN_ALL)
        put_cell(key, c)
        if c is not None and folder in ("rev_p8_drift", "rev2_drift_guard", "rev2_drift_long"):   # norm relative to the pilots
            e = float(c["pairs"][0][0]["config"]["attack"]["a3_eps_small"])
            put(key + ".sir", f"{-20 * np.log10(e):.0f}")
        if c is None:
            continue
        snap = [(a, cl) for a, cl in c["pairs"] if "paired_disp" in a]   # pairs whose snapshots were analysed
        if not snap:
            rows.append(f"{lab} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_rate(c['rej'])} & -- & -- & --\\\\")
            continue
        put(key + ".nsnap", str(len(snap)))
        DK, des, ext, anc, mv = [], [], [], [], []
        for a, cl in snap:
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
            if "paired_move" in a:   # movement of level K from the window start, at the window end
                u = a["paired_move"]["adv"][K - 1][w1 - 1]; v = a["paired_move"]["clean"][K - 1][w1 - 1]
                d = a["paired_disp"][K - 1][w1 - 1]
                mv.append((u, v, (u * u + v * v - d * d) / (2 * u * v) if u > 0 and v > 0 else float("nan")))
        if mv:
            put(key + ".movadv", f"{np.mean([m[0] for m in mv]):.1f}"); put(key + ".movcln", f"{np.mean([m[1] for m in mv]):.1f}")
            put(key + ".movcos", f"{np.mean([m[2] for m in mv]):.2f}")
            put(key + ".movpct", f"{100 * np.mean([m[0] / m[1] for m in mv]):.0f}")
        put(key + ".DK", f"{np.mean(DK):.2f}"); put(key + ".design", f"{np.mean(des):.1f}")
        put(key + ".struct", f"{np.mean(ext):.1f}"); put(key + ".anchor", f"{np.mean(anc):.2f}")
        nn = f"{c['n']}" if len(snap) == c["n"] else f"{c['n']} ({len(snap)})"   # (seeds with level snapshots)
        rows.append(f"{lab} & {nn} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_rate(c['rej'])} & {np.mean(DK):.2f} & "
                    f"{np.mean(ext):.1f} & {np.mean(des):.1f}\\\\")
        if folder == "rev_p8_drift":
            p8.append((c, np.mean(DK), np.mean(ext)))
    if p8:   # over the A3* settings: the structural bound relative to D_K, and the excess after the attack
        put("ret.p8.ratiomax", f"{max(e / d for _, d, e in p8):.1f}"); put("ret.p8.ratiomin", f"{min(e / d for _, d, e in p8):.1f}")
        put("ret.p8.postmax", f_sig(max(c["post"] for c, _, _ in p8)))
    write_table("retention", rows)


def extras() -> None:
    """Numbers quoted in the text that are not table cells (mechanisms behind the table entries)."""
    # why rollback is not enough (Sec. VI-B): what the reference of the CRC test does under attack,
    # when the first poisoned block is accepted, and whether the receiver recovers after the window
    for rx, tag in (("crc_rollback", "crc"), ("citadel", "cit"), ("rollback_allgates", "rag")):
        folder, nm = ("rev_p2", f"p2_{rx}__wb_sir0") if rx == "rollback_allgates" else ("rev_p1", f"p1_{rx}__wb_sir0")
        c = cell(folder, nm, CLEAN_ALL)
        if c is None:
            continue
        first0 = nacc = acc = att = 0
        postrej, refA, refC, nnA, endex = [], [], [], [], []
        for a, cl in c["pairs"]:
            w0, w1 = window(a)
            accb = [b["block"] for b in a["blocks"] if b.get("attacked") and b.get("gamma") is not None and b["gamma"] >= 0.5]
            first0 += bool(accb) and accb[0] == w0
            nacc += bool(accb)
            acc += len(accb)
            att += sum(bool(b.get("attacked")) for b in a["blocks"])
            postrej.append(np.mean([b["gamma"] < 0.5 for b in a["blocks"] if b["block"] >= w1]))
            refA.append(np.mean([b["pass_mm"] for b in a["blocks"] if b.get("attacked")]))
            refC.append(np.mean([b["pass_mm"] for b in cl["blocks"][w0:w1]]))
            nnA.append(np.mean([b["pass_nn"] for b in a["blocks"] if b.get("attacked")]))
            endex.append(float((ser(a)[-10:] - ser(cl)[-10:]).mean()))
        k = "c1." + tag
        put(k + ".first0", str(first0)); put(k + ".seedsacc", str(nacc)); put(k + ".nseeds", str(c["n"]))
        put(k + ".acc", f"{acc}/{att}"); put(k + ".postrej", f"{np.mean(postrej):.2f}")
        put(k + ".refatt", f"{np.mean(refA):.2f}"); put(k + ".refcln", f"{np.mean(refC):.2f}")
        put(k + ".nnatt", f"{np.mean(nnA):.2f}"); put(k + ".endex", f_sig(float(np.mean(endex))))
    c = cell("rev_p1", "p1_citadel__wb_sir0", CLEAN_ALL)
    if c is not None:
        att = [b for a, _ in c["pairs"] for b in a["blocks"] if b.get("attacked")]
        for g in ("g1", "g2", "g3"):   # share of the poisoned blocks each gate rejects on its own
            put(f"gate.{g}", f"{100 * np.mean([b['verdict_' + g] < 0.5 for b in att]):.0f}")
        put("gate.lmmseatt", f"{np.mean([b['pass_lmmse'] for b in att]):.2f}")
        put("gate.lmmsecln", f"{np.mean([b['pass_lmmse'] for a, cl in c['pairs'] for b in cl['blocks'][window(a)[0]:window(a)[1]]]):.2f}")

        def clipped(recs):
            k = n = 0
            for r in recs:
                for b in r["blocks"]:
                    for e in b.get("consolidations") or []:
                        n += 1
                        k += bool(e.get("clipped"))
            return k, n
        ka, na = clipped([a for a, _ in c["pairs"]])
        kc, nc = clipped([cl for _, cl in c["pairs"]])
        put("clip.att", f"{ka}/{na}"); put("clip.cln", f"{kc}/{nc}")
    # the clip against the clean consolidation increments of the chosen candidate on the calibration seeds
    fz = os.path.join(ROOT, "configs", "revision_frozen_citadel.json")
    if os.path.exists(fz):
        chosen = json.load(open(fz))["chosen"]
        for lev in (2, 3):
            inc = [e["inc_norm"] for r in load("rev_cal_citadel") if r["name"] == f"cand_{chosen}__clean"
                   for b in r["blocks"] for e in (b.get("consolidations") or []) if e["level"] == lev]
            if inc:
                put(f"clip.calmax{lev}", f"{max(inc):.1f}"); put(f"clip.calp99{lev}", f"{np.percentile(inc, 99):.1f}")
    # where the clean-SER gain comes from: online DeepSIC -> + CRC selection -> Citadel-Rx (10 seeds)
    cs = {rx: cell("rev_p1", f"p1_{rx}__wb_sir0", CLEAN_ALL) for rx in P1RX}
    if all(cs[rx] is not None for rx in ("citadel", "online_deepsic", "online_select")):
        o, sl, ci = (cs[rx]["clean_all"] for rx in ("online_deepsic", "online_select", "citadel"))
        put("sel.share", f"{100 * (o - sl) / (o - ci):.0f}")
    for rx, cc in cs.items():   # wall time per block of one single-threaded run (clean twins, MacBook)
        if cc is not None:
            put("time." + rx.replace("_", ""),
                f"{np.mean([np.mean([b['t_rx'] for b in cl['blocks']]) for _, cl in cc['pairs'] if machine(cl) == 'mac']):.2f}")
    for folder, nm, k in (("rev_p2", "p2_lmmse__wb_sir0", "lmmse"), ("rev_p6_deploy", "p6_citadel_steps30__sir0", "steps30"),
                          ("rev_p6_deploy", "p6_citadel_last_layer__sir0", "lastlayer")):
        cc = cell(folder, nm, CLEAN_ALL)
        if cc is not None:
            put("time." + k, f"{np.mean([np.mean([b['t_rx'] for b in cl['blocks']]) for _, cl in cc['pairs'] if machine(cl) == 'mac']):.2f}")
    # jamming at the same SIR covers the data symbols: energy ratio to pilot poisoning
    r0 = next(iter(load("rev_p1")), None)
    if r0 is not None:
        put("jam.energyratio", f"{r0['config']['frame']['data_bits'] // r0['config']['frame']['pilot_bits']}")
    # EATA-style filter: largest mean data entropy of the attacked receiver against its threshold
    ents = [b["data_entropy"] for r in load("rev_p2") if r["name"].startswith("p2_eata") for b in r["blocks"]
            if b.get("data_entropy") is not None]
    if ents:
        thr = 0.4 * np.log(4)
        put("eata.maxent", f"{max(ents):.2f}"); put("eata.thr", f"{thr:.2f}"); put("eata.nfilt", str(sum(e > thr for e in ents)))
    # 16-QAM: clean blocks, CRC pass fraction of the network against the reference
    cq = cell("rev_p6_qam16", "p6_citadel_qam16__sir0", CLEAN_ALL)
    if cq is not None:
        cb = [b for _, cl in cq["pairs"] for b in cl["blocks"][5:]]
        put("qam.passnn", f"{np.mean([b['pass_nn'] for b in cb]):.2f}"); put("qam.passref", f"{np.mean([b['pass_mm'] for b in cb]):.2f}")
        put("qam.frej", f"{100 * np.mean([b['gamma'] < 0.5 for b in cb]):.0f}")
    # the abstain state on clean runs: the block from which it rejects, and the reference at that point
    ab = [r for r in load("rev_clean_cloud") if r["config"]["citadel"].get("abstain")
          and r["config"]["citadel"].get("abstain_mode", "legacy") == "legacy"]   # the first version only
    if ab:
        firsts, refs0, refsl, nrej, nall = [], [], [], 0, 0
        for r in ab:
            g = [b.get("gamma") for b in r["blocks"]]
            fr = next((i for i, x in enumerate(g) if x is not None and x < 0.5), None)
            if fr is not None:
                firsts.append(fr); refs0.append(r["blocks"][0]["pass_mm"]); refsl.append(r["blocks"][fr]["pass_mm"])
            nrej += sum(x is not None and x < 0.5 for x in g); nall += len(g)
        if firsts:
            put("abs.lockmin", str(min(firsts))); put("abs.lockmax", str(max(firsts)))
            put("abs.ref0", f"{np.mean(refs0):.3f}"); put("abs.reflock", f"{np.mean(refsl):.3f}")
        put("abs.rejk", f"{nrej}/{nall}")
    # A4: the reference the gates compare against, attacked blocks against the clean twin
    for a in ("0.25", "0.75"):
        ca = cell("rev_p3_refdeg", f"p3_a4_a{a}__sir0", CLEAN_ALL)
        if ca is not None:
            put("a4." + a.replace(".", "") + ".refatt", f"{np.mean([b['pass_mm'] for x, _ in ca['pairs'] for b in x['blocks'] if b.get('attacked')]):.2f}")
            put("a4." + a.replace(".", "") + ".refcln", f"{np.mean([b['pass_mm'] for x, cl in ca['pairs'] for b in cl['blocks'][window(x)[0]:window(x)[1]]]):.2f}")
            put("a4." + a.replace(".", "") + ".nnatt", f"{np.mean([b['pass_nn'] for x, _ in ca['pairs'] for b in x['blocks'] if b.get('attacked')]):.2f}")


def g1_mismatch() -> None:
    """A8: the normalised G1 statistic on clean blocks of the tanh (nonlinear) channel, from the
    first-version runs on seeds 0-2 at SNR 4-16 dB (results/e2nl)."""
    per = {}
    for r in load("e2nl"):
        c = r["config"]
        if c["receiver"]["type"] != "citadel" or c["attack"]["type"] != "none" or c["channel"].get("linear", True):
            continue
        g = [b["gate_g1"] for b in r["blocks"] if b["block"] >= 5 and b.get("gate_g1") is not None]
        per.setdefault(int(round(c["channel"]["snr_db"])), []).extend(g)
    if 10 in per:
        g = np.array(per[10])
        put("g1nl.mean", f"{g.mean():.3f}"); put("g1nl.p99", f"{np.percentile(g, 99):.3f}")
        put("g1nl.max", f"{g.max():.3f}"); put("g1nl.n", str(g.size))
    if per:
        put("g1nl.allmax", f"{max(max(v) for v in per.values()):.3f}")
        put("g1nl.snrmin", str(min(per))); put("g1nl.snrmax", str(max(per)))


def lipschitz() -> None:
    """A6: finite-difference probes of the fast learner (scripts/certificate_check.py), largest
    L_theta + L_a per proximal weight; docs/certificate_constants.json is the default lambda = 0.5."""
    sums = {}
    p0 = os.path.join(ROOT, "docs", "certificate_constants.json")
    p1 = os.path.join(ROOT, "docs", "certificate_constants_lambda.json")
    if os.path.exists(p0):
        d = json.load(open(p0)); sums[0.5] = d["L_theta_max"] + d["L_a_max"]
    if os.path.exists(p1):
        for lam, d in json.load(open(p1)).items():
            sums[float(lam)] = d["L_theta_max"] + d["L_a_max"]
    if sums:
        put("lip.summin", f"{min(sums.values()):.2f}"); put("lip.summax", f"{max(sums.values()):.2f}")
        put("lip.lammin", f"{min(sums):g}"); put("lip.lammax", f"{max(sums):g}")


def campaign2() -> None:
    """Second campaign: experiments added to close the audit items."""
    # B2 at full strength: A4 with 20 PGD steps
    c = cell("rev2_a4_pgd20", "b2_a4_a0.25_pgd20__sir0", CLEAN_ALL)
    put_cell("b2.a4pgd20", c)
    # H3, final version: abstain on the residual of CRC-verified symbols over the pilot noise, frozen threshold
    fz = os.path.join(ROOT, "configs", "revision2_frozen_abstain_verified.json")
    if os.path.exists(fz):
        f = json.load(open(fz))
        put("h3.z", f"{f['chosen_z']:g}"); put("h3.calrate", f"{100 * f['clean_abstention_rate'][str(f['chosen_z'])]:.1f}")
        put("h3.calmax", f"{f['max_abs_dev']:.1f}"); put("h3.calblocks", str(f["n_blocks"]))
    for a in ("0.25", "0.75"):
        c = cell("rev2_abstain_v3", f"h3v_a4_a{a}__sir0", CLEAN_ALL)
        put_cell("h3.a4" + a.replace(".", ""), c)
        if c is not None:   # blocks abstained inside and outside the attack window
            ins = [bool(b.get("abstain")) for p in c["pairs"] for b in p[0]["blocks"][window(p[0])[0]:window(p[0])[1]]]
            put("h3.a4" + a.replace(".", "") + ".abst", f"{100 * np.mean(ins):.0f}")
    c = cell("rev2_abstain_v3", "h3v_a1__sir0", CLEAN_ALL)
    put_cell("h3.a1", c)
    if c is not None:
        cl = [bool(b.get("abstain")) for _, cln in c["pairs"] for b in cln["blocks"][5:]]
        put("h3.cleanabst", f"{100 * np.mean(cl):.1f}"); put("h3.cleanabstk", f"{sum(cl)}/{len(cl)}")
    for tag in ("redraw", "snrdrop"):   # legitimate changes: blocks abstained after the change, SER after it
        recs = [r for r in load("rev2_abstain_v3") if r["name"] == f"h3v_{tag}__none"]
        if recs:
            ab = [sum(bool(b.get("abstain")) for b in r["blocks"][50:]) for r in recs]
            put(f"h3.{tag}.abst", f"{np.mean(ab):.1f}"); put(f"h3.{tag}.abstmax", str(max(ab)))
            put(f"h3.{tag}.after", f_ser(float(np.mean([ser(r)[50:].mean() for r in recs])))); put(f"h3.{tag}.n", str(len(recs)))
    # B3/H4 as intended: strong poisoning (A1 at SIR 0 dB) on blocks 40-89 while the channel is redrawn at block 50
    rows, recs = [], load("rev2_deny_a1")
    lab = {"citadel": "Citadel-Rx", "citadel_streak": "+ streak recovery", "online_deepsic": "Online DeepSIC", "lmmse": "LMMSE"}
    for rx in ("citadel", "citadel_streak", "online_deepsic", "lmmse"):
        none = {r["seed"]: r for r in recs if r["name"] == f"b3_{rx}_redraw__none"}
        den = {r["seed"]: r for r in recs if r["name"] == f"b3_{rx}_redraw__a1deny"}
        sd = sorted(set(none) & set(den))
        if not sd:
            continue
        k = "b3." + rx.replace("_", "")
        win = float(np.mean([(ser(den[i])[40:90] - ser(none[i])[40:90]).mean() for i in sd]))
        post = float(np.mean([(ser(den[i])[90:] - ser(none[i])[90:]).mean() for i in sd]))
        after = float(np.mean([ser(none[i])[50:].mean() for i in sd]))
        rj = [rejections(den[i], True, skip=0) for i in sd]
        rjk = (sum(x for x, _ in rj), sum(y for _, y in rj))
        att = sum("recovery_pass" in b for i in sd for b in den[i]["blocks"])
        acc = sum(bool(b.get("recovered")) for i in sd for b in den[i]["blocks"])
        put(k + ".n", str(len(sd))); put(k + ".win", f_sig(win)); put(k + ".post", f_sig(post)); put(k + ".after", f_ser(after))
        put(k + ".rej", f_rate(rjk) if rjk[1] else "--"); put(k + ".att", str(att)); put(k + ".acc", str(acc))
        rows.append(f"{lab[rx]} & {len(sd)} & {f_rate(rjk) if rjk[1] and rx.startswith('citadel') else '--'} & {f_ser(after)} & "
                    f"{f_sig(win)} & {f_sig(post)} & {f'{att}/{acc}' if rx == 'citadel_streak' else '--'}\\\\")
    write_table("deny2", rows)
    # H7: per-user gating against a single-user attack (A2, target user 0) and against A1
    rows = []
    spec = [("h7_citadel__a2", "A2 (user 1)", "Citadel-Rx"), ("h7_citadel_peruser__a2", "A2 (user 1)", "+ per-user gating"),
            ("h7_online_deepsic__a2", "A2 (user 1)", "Online DeepSIC"), ("h7_citadel__a1", "A1", "Citadel-Rx"),
            ("h7_citadel_peruser__a1", "A1", "+ per-user gating")]
    for name, l1, l2 in spec:
        c = cell("rev2_peruser", name, CLEAN_ALL)
        key = "h7." + re.sub(r"[^A-Za-z0-9]", "", name.replace("h7_", ""))
        put_cell(key, c)
        if c is None:
            continue
        tgt = float(np.mean([np.mean([b["ser_user"][0] for b in a["blocks"][window(a)[0]:window(a)[1]]])
                             - np.mean([b["ser_user"][0] for b in cl["blocks"][window(a)[0]:window(a)[1]]]) for a, cl in c["pairs"]]))
        put(key + ".tgt", f_sig(tgt))
        if "peruser" in name:   # per-user gating: the gate reverts users, not blocks; count (block, user) reverts
            # a user is reverted on a block if the whole block is rejected (G1, G3, abstain) or its own G2 test fails
            rev = lambda b, u: (b.get("gamma") is not None and b["gamma"] < 0.5) or u in (b.get("users_reverted") or [])
            U = c["pairs"][0][0]["config"]["channel"]["n_user"]
            vt = [rev(b, 0) for a, _ in c["pairs"] for b in a["blocks"][window(a)[0]:window(a)[1]]]
            vo = [rev(b, u) for a, _ in c["pairs"] for b in a["blocks"][window(a)[0]:window(a)[1]] for u in range(1, U)]
            vc = [rev(b, u) for _, cl in c["pairs"] for b in cl["blocks"][5:] for u in range(U)]
            rj = f"{f_rate((sum(vt), len(vt)))} / {f_rate((sum(vo), len(vo)))}" if vt else "--"
            put(key + ".revtgt", f_rate((sum(vt), len(vt)))); put(key + ".revoth", f_rate((sum(vo), len(vo))))
            put(key + ".revclean", f_rate((sum(vc), len(vc))))
        else:
            rj = f_rate(c["rej"]) if "Citadel" in l2 else "--"
        rows.append(f"{l1} & {l2} & {c['n']} & {f_sig(c['excess'])} {f_ci(c['ci'])} & {f_sig(tgt)} & {rj}\\\\")
    write_table("peruser", rows)
    # D8: the second backbone (joint-symbol detector) under white-box A1 at SIR 0 dB
    fz = os.path.join(ROOT, "configs", "revision2_frozen_joint.json")
    if os.path.exists(fz):
        f = json.load(open(fz))
        m = re.match(r"ep(\d+)_lr([0-9.e-]+)$", f["chosen"])
        put("d8.tuned", f"{m.group(1)} epochs, learning rate {float(m.group(2)):g}")
        put("d8.calser", f_ser(f["cal_clean_ser"]))
    rows = []
    for rx, l in (("citadel", "Citadel-Rx"), ("online_deepsic", "Online"), ("crc_rollback", "CRC-Rollback")):
        c = cell("rev2_joint", f"d8_{rx}_joint__wb_sir0", CLEAN_ALL)
        put_cell("d8." + rx.replace("_", ""), c)
        if c is not None:
            rows.append(f"{l} & {c['n']} & {f_ser(c['clean_all'])} & {f_sig(c['excess'])} {f_ci(c['ci'])} & "
                        f"{f_rate(c['rej']) if rx != 'online_deepsic' else '--'} & {f_rate(c['frej']) if rx != 'online_deepsic' else '--'}\\\\")
    write_table("joint", rows)
    # D2/D3: NR LDPC code blocks over 38.901 UMi channels, attacker at 35, 75 and 150 m
    rows = []
    summ = os.path.join(ROOT, "resources", "umi", "summary.json")
    sirs = json.load(open(summ)) if os.path.exists(summ) else []
    for d in (35, 75, 150):
        sd = [x["sir_db"] for x in sirs if x["d_att"] == d]
        if sd:
            put(f"d23.d{d}.sirmed", f"{np.median(sd):+.0f}"); put(f"d23.d{d}.sirmin", f"{min(sd):+.0f}"); put(f"d23.d{d}.sirmax", f"{max(sd):+.0f}")
        labelled = False
        for rx, l in (("citadel", "Citadel-Rx"), ("online_deepsic", "Online"), ("crc_rollback", "CRC-Rollback"), ("lmmse", "LMMSE")):
            cb = cell("rev2_umi_ldpc_scr", f"d23s_{rx}_d{d}__phys", CLEAN_ALL, metric="bler")
            cs = cell("rev2_umi_ldpc_scr", f"d23s_{rx}_d{d}__phys", CLEAN_ALL)
            key = f"d23.{rx.replace('_', '')}.d{d}"
            put_cell(key, cs)
            put_cell(key + "bler", cb)
            if cb is None:
                continue
            first = f"{d}\\,m ({np.median(sd):+.0f}\\,dB)" if (not labelled and sd) else ""
            labelled = True
            rows.append(f"{first} & {l} & {cb['n']} & {f_ser(cb['clean_all'])} & {f_sig(cb['excess'])} {f_ci(cb['ci'])} & "
                        f"{f_sig(cs['excess'])} & {f_rate(cs['rej']) if rx in ('citadel', 'crc_rollback') else '--'}\\\\")
    write_table("umi", rows)
    # mechanism in the coded scene: how often the fast level is trusted on clean blocks, what the
    # reference (the better of LMMSE and the anchor, virtual pass fraction) does under attack, and how
    # the clean BLER of the online receivers moves over a run
    for d in (35, 75, 150):
        c = cell("rev2_umi_ldpc_scr", f"d23s_citadel_d{d}__phys", CLEAN_ALL)
        if c is None:
            continue
        k = f"d23m.d{d}"
        att = [b for a, _ in c["pairs"] for b in a["blocks"] if b.get("attacked")]
        cln = [b for a, cl in c["pairs"] for b in cl["blocks"][window(a)[0]:window(a)[1]]]
        put(k + ".refatt", f"{np.mean([b['pass_mm'] for b in att if b.get('pass_mm') is not None]):.2f}")
        put(k + ".refcln", f"{np.mean([b['pass_mm'] for b in cln if b.get('pass_mm') is not None]):.2f}")
        put(k + ".nnatt", f"{np.mean([b['pass_nn'] for b in att if b.get('pass_nn') is not None]):.2f}")
        put(k + ".nncln", f"{np.mean([b['pass_nn'] for b in cln if b.get('pass_nn') is not None]):.2f}")
    for rx in ("citadel", "online_deepsic", "crc_rollback", "lmmse"):
        tw = twins(CLEAN_ALL)
        cl = [twin_of(tw, a) for a in load("rev2_umi_ldpc_scr") if a["name"] == f"d23s_{rx}_d75__phys"]
        cl = [c for c in cl if c is not None]
        if cl:
            k = "d23m." + rx.replace("_", "")
            put(k + ".blerfirst", f_ser(float(np.mean([np.mean([b["bler"] for b in c["blocks"][:20]]) for c in cl]))))
            put(k + ".blerlast", f_ser(float(np.mean([np.mean([b["bler"] for b in c["blocks"][-20:]]) for c in cl]))))
            put(k + ".serfirst", f_ser(float(np.mean([np.mean([b["ser"] for b in c["blocks"][:20]]) for c in cl]))))
            put(k + ".serlast", f_ser(float(np.mean([np.mean([b["ser"] for b in c["blocks"][-20:]]) for c in cl]))))
            put(k + ".ncl", str(len(cl)))
            if rx == "citadel":
                fr = [rejections(c, None, skip=5) for c in cl]
                put(k + ".frej", f_rate((sum(a for a, _ in fr), sum(b for _, b in fr))))
    # D4: the 8 x 32 array (rows added to the scenarios table by scenarios())
    # D7: 2000-block clean horizon
    for rx in ("citadel", "online_deepsic"):
        recs = [r for r in load("rev2_long_clean") if r["name"] == f"d7_{rx}__clean2000"]
        if recs:
            k = "d7." + rx.replace("_", "")
            s_ = [ser(r) for r in recs]
            put(k + ".n", str(len(recs))); put(k + ".ser", f_ser(float(np.mean([x[5:].mean() for x in s_]))))
            put(k + ".first", f_ser(float(np.mean([x[5:500].mean() for x in s_]))))
            put(k + ".last", f_ser(float(np.mean([x[-500:].mean() for x in s_]))))
            if rx == "citadel":
                fr = [rejections(r, None, skip=5) for r in recs]
                put(k + ".frej", f_rate((sum(a for a, _ in fr), sum(b for _, b in fr))))
                put(k + ".frejk", f"{sum(a for a, _ in fr)}/{sum(b for _, b in fr)}")
                late = [rejections({"blocks": r["blocks"][-500:]}, None, skip=0) for r in recs]
                put(k + ".frejlast", f_rate((sum(a for a, _ in late), sum(b for _, b in late))))
    # A6: stronger Lipschitz estimates (hill-climbing input gain, per channel family)
    fn = os.path.join(ROOT, "docs", "lipschitz_strong.json")
    if os.path.exists(fn):
        res = json.load(open(fn))["results"]
        for r in res:
            k = "lip2." + r["family"].replace("cost2100", "cost")
            put(k + ".sum", f"{r['L_sum_max']:.2f}"); put(k + ".lt", f"{r['L_theta']['max']:.2f}"); put(k + ".la", f"{r['L_a']['max']:.2f}")
            put(k + ".g", f"{r['G_Phi']['max']:.2f}")
        put("lip2.summin", f"{min(r['L_sum_max'] for r in res):.2f}"); put("lip2.summax", f"{max(r['L_sum_max'] for r in res):.2f}")
    # D9: multiply-accumulates per block (scripts/flops.py)
    fn = os.path.join(ROOT, "docs", "flops.json")
    if os.path.exists(fn):
        fl = json.load(open(fn))["4x4_qpsk"]
        for rx, r in fl.items():
            put("flops." + rx.replace("_", ""), f"{r['total'] / 1e9:.1f}")
        put("flops.ratio", f"{fl['citadel']['total'] / fl['online_deepsic']['total']:.1f}")
        put("flops.trainshare", f"{100 * fl['citadel']['train'] / fl['citadel']['total']:.0f}")


def campaign_size() -> None:
    n = 0
    for d in glob.glob(os.path.join(RES, "rev_*")):
        n += len(glob.glob(os.path.join(d, "*.json")))
    put("camp.records", f"{n:,}".replace(",", "\\,"))


def main() -> None:
    for name in NCOLS:   # every table file exists, so the paper always compiles
        write_table(name, [])
    p1(); adaptive(); refdeg(); deny(); baselines(); ablations(); scenarios(); sensitivity(); tuning(); retention(); extras(); g1_mismatch(); lipschitz(); campaign2()
    campaign_size()
    lines = ["% generated by scripts/revision_numbers.py -- do not edit",
             "\\makeatletter"]
    for k in sorted(V):
        lines.append(f"\\@namedef{{val@{k}}}{{{V[k]}}}")
    lines.append("\\makeatother")
    open(OUT_NUM, "w").write("\n".join(lines) + "\n")
    tex = open(os.path.join(ROOT, "paper", "main.tex")).read() if os.path.exists(os.path.join(ROOT, "paper", "main.tex")) else ""
    tex = "\n".join(re.split(r"(?<!\\)%", l)[0] for l in tex.splitlines())   # comments; an escaped \% is text
    used = sorted(set(re.findall(r"\\val\{([^}]+)\}", tex)))
    missing = [k for k in used if k not in V]
    print(f"{len(V)} numbers written to {os.path.relpath(OUT_NUM, ROOT)}; tables in {os.path.relpath(OUT_TAB, ROOT)}")
    print(f"used in main.tex: {len(used)}; missing (shown red in the PDF): {len(missing)}")
    for k in missing:
        print("   missing:", k)


if __name__ == "__main__":
    main()
