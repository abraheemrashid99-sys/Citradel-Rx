#!/usr/bin/env python
"""Put the level snapshots and the pretrained-weight caches of the revision back into a clone.

The files are too large for the main branch (GitHub accepts at most 100 MB per file), so they live on
the branch `snapshots` of this repository, the large ones cut into parts of 90 MB. MANIFEST.json (the
cloud machine) and manifests/<machine>.json (every other machine) on that branch list every file with
its size, its sha256, its parts and the machine that wrote it. This script downloads the parts it needs,
joins them, checks the sha256 of every file and writes it to its place. A file that is already in place
with the right sha256 is left alone; a file whose sha256 does not match is an error and is not written.

    python scripts/restore_snapshots.py                  # in a clone of main: every level snapshot
    python scripts/restore_snapshots.py --list           # only list what is on the branch
    python scripts/restore_snapshots.py --machine pool4  # only the snapshots of one machine
    python scripts/restore_snapshots.py --only pretrain_cache --machine cloud
    python restore_snapshots.py --to /path/to/Citradel-Rx    # in a checkout of the snapshots branch

What is restored, by kind:
  snapshot        results/<folder>/<record>.levels.npz, next to its JSON record (the default)
  pretrain_cache  cache/pretrain/<key>.pt; every machine made its own, so name the machine (--machine)
  machine_file    other files a machine held that are not on main (figures and tables it drew from its
                  own snapshots, logs); written to results_archive/<machine>/, which the analysis never reads

Nothing in the paper needs these files (see docs/SNAPSHOTS.md): with the snapshots in place, `make paper`
gives the same numbers, because scripts/paired_disp.py recomputes the stored displacements from them.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BRANCH = "snapshots"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _rm(func, path, _exc):          # Windows: git marks its object files read-only
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _inside(rel: str) -> bool:
    """A relative path that stays inside the folder it is joined to (a manifest must never write elsewhere)."""
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and "\\" not in rel and ":" not in parts[0] and all(p not in ("", ".", "..") for p in parts)


def load_manifests(src: str) -> list[dict]:
    out = []
    for p in [os.path.join(src, "MANIFEST.json")] + sorted(glob.glob(os.path.join(src, "manifests", "*.json"))):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out += json.load(f)
    for m in out:
        if not _inside(m["path"]) or not all(_inside(q) for q in m["parts"]):
            sys.exit(f"refusing a manifest entry with a path outside the repository: {m['path']!r}")
    return out


def select(manifest: list[dict], only: str | None, machine: str | None) -> list[dict]:
    kind = only or "snapshot"
    sel = [m for m in manifest if m["kind"] == kind and machine in (None, m.get("machine"))]
    if kind == "pretrain_cache" and machine is None:
        sys.exit("every machine made its own caches for the same folder: name one, e.g. --only pretrain_cache --machine cloud")
    seen, out = {}, []
    for m in sel:                    # two machines never hold the same record; if they ever do, keep the first
        if m["path"] in seen:
            if seen[m["path"]] != m["sha256"]:
                print(f"warning: {m['path']} is on the branch twice with different content; kept the copy of the first machine")
            continue
        seen[m["path"]] = m["sha256"]
        out.append(m)
    return out


def run(*cmd, cwd=None, data=None) -> bool:
    return subprocess.run(cmd, cwd=cwd, input=data).returncode == 0


def download(url: str, tmp: str, want) -> None:
    """Fetch the branch into tmp. First only the manifests, then only the parts that `want` selects; if this
    git cannot do a partial clone, the whole branch."""
    partial = run("git", "clone", "-q", "--depth", "1", "--single-branch", "--branch", BRANCH, "--filter=blob:none",
                  "--no-checkout", url, tmp)
    if partial:
        pats = "/MANIFEST.json\n/manifests/*.json\n"
        partial = (run("git", "sparse-checkout", "set", "--no-cone", "--stdin", cwd=tmp, data=pats.encode())
                   and run("git", "checkout", "-q", BRANCH, cwd=tmp))
    if partial:
        parts = [p for m in want(load_manifests(tmp)) for p in m["parts"]]
        esc = lambda p: "/" + "".join("\\" + c if c in "*?[\\!#" else c for c in p)
        pats += "".join(esc(p) + "\n" for p in parts)
        if parts:
            print(f"downloading {len(parts)} files of the branch '{BRANCH}' ...", flush=True)
        if not run("git", "sparse-checkout", "set", "--no-cone", "--stdin", cwd=tmp, data=pats.encode()):
            partial = False
    if not partial:
        shutil.rmtree(tmp, onerror=_rm)
        print(f"downloading the whole branch '{BRANCH}' (several GB) ...", flush=True)
        if not run("git", "clone", "-q", "--depth", "1", "--single-branch", "--branch", BRANCH, url, tmp):
            sys.exit("download failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None, help="a checkout of the snapshots branch (default: download it)")
    ap.add_argument("--to", dest="dst", default=None, help="the clone to restore into (default: this repository)")
    ap.add_argument("--only", choices=["snapshot", "pretrain_cache", "machine_file"], default=None,
                    help="the kind of file to restore (default: snapshot)")
    ap.add_argument("--machine", default=None, help="only the files written by this machine (cloud, pool1, ...)")
    ap.add_argument("--list", action="store_true", help="list the files of the branch and stop")
    args = ap.parse_args()

    on_branch = os.path.exists(os.path.join(HERE, "MANIFEST.json"))
    src = args.src or (HERE if on_branch else None)
    dst = args.dst or (None if on_branch else os.path.dirname(HERE))
    if dst is None and not args.list:
        sys.exit("give the clone to restore into: --to /path/to/Citradel-Rx")
    want = (lambda man: [m for m in man if args.only in (None, m["kind"]) and args.machine in (None, m.get("machine"))]) \
        if args.list else (lambda man: select(man, args.only, args.machine))
    tmp = None
    try:
        if src is None:
            url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=dst or HERE, capture_output=True, text=True).stdout.strip()
            if not url:
                sys.exit("no git remote 'origin' here; clone the snapshots branch yourself and pass --from")
            tmp = tempfile.mkdtemp(prefix="citadel_snapshots_")
            download(url, tmp, (lambda man: []) if args.list else want)
            src = tmp
        todo = want(load_manifests(src))
        if args.list:
            for m in todo:
                print(f"{m['bytes'] / 2 ** 20:8.1f} MB  {m['kind']:14s} {m.get('machine', ''):8s} {m['path']}")
            print(f"{len(todo)} files, {sum(m['bytes'] for m in todo) / 2 ** 20:.0f} MB")
            return
        if not todo:
            sys.exit("nothing on the branch matches")
        written = kept = 0
        for m in todo:
            out = os.path.join(dst, *m["path"].split("/"))
            if os.path.exists(out) and os.path.getsize(out) == m["bytes"] and sha256(out) == m["sha256"]:
                kept += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            part = out + ".restoring"
            h = hashlib.sha256()
            with open(part, "wb") as w:
                for p in m["parts"]:
                    with open(os.path.join(src, *p.split("/")), "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 22), b""):
                            h.update(chunk)
                            w.write(chunk)
            if h.hexdigest() != m["sha256"] or os.path.getsize(part) != m["bytes"]:
                os.remove(part)
                sys.exit(f"sha256 mismatch for {m['path']}: the download is damaged, nothing written for this file")
            os.replace(part, out)
            written += 1
            print(f"restored {m['path']} ({m['bytes'] / 2 ** 20:.0f} MB)", flush=True)
        print(f"{written} files restored, {kept} already in place, all sha256 verified")
    finally:
        if tmp:
            shutil.rmtree(tmp, onerror=_rm)


if __name__ == "__main__":
    main()
