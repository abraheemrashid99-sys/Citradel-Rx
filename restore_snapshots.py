#!/usr/bin/env python
"""Put the level snapshots and the pretrained-weight caches of the revision back into a clone.

The files are too large for the main branch (GitHub accepts at most 100 MB per file), so they live on
the branch `snapshots` of this repository, the large ones cut into parts of 90 MB. MANIFEST.json on that
branch lists every file with its size, its sha256 and its parts. This script joins the parts, checks
the sha256 of every file and writes it to its place (results/<folder>/<record>.levels.npz and
cache/pretrain/<key>.pt). A file that is already in place with the right sha256 is left alone; a file
whose sha256 does not match is an error and is not written.

    python scripts/restore_snapshots.py                 # in a clone of main: downloads the branch (1.4 GB)
    python scripts/restore_snapshots.py --list          # only list what is on the branch
    python scripts/restore_snapshots.py --only snapshot # or: --only pretrain_cache
    python restore_snapshots.py --to /path/to/Citradel-Rx   # in a checkout of the snapshots branch

Nothing in the paper needs these files (see docs/SNAPSHOTS.md): with them in place, `make paper` gives
the same numbers, because scripts/paired_disp.py recomputes the stored displacements from them.
"""
from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None, help="a checkout of the snapshots branch (default: download it)")
    ap.add_argument("--to", dest="dst", default=None, help="the clone to restore into (default: this repository)")
    ap.add_argument("--only", choices=["snapshot", "pretrain_cache"], default=None)
    ap.add_argument("--list", action="store_true", help="list the files and stop")
    args = ap.parse_args()

    on_branch = os.path.exists(os.path.join(HERE, "MANIFEST.json"))
    src = args.src or (HERE if on_branch else None)
    dst = args.dst or (None if on_branch else os.path.dirname(HERE))
    if dst is None and not args.list:
        sys.exit("give the clone to restore into: --to /path/to/Citradel-Rx")
    tmp = None
    try:
        if src is None:
            url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=dst, capture_output=True, text=True).stdout.strip()
            if not url:
                sys.exit("no git remote 'origin' here; clone the snapshots branch yourself and pass --from")
            tmp = tempfile.mkdtemp(prefix="citadel_snapshots_")
            print(f"downloading the branch '{BRANCH}' of {url} (about 1.4 GB) ...", flush=True)
            r = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--branch", BRANCH, url, tmp])
            if r.returncode != 0:
                sys.exit("download failed")
            src = tmp
        manifest = json.load(open(os.path.join(src, "MANIFEST.json")))
        todo = [m for m in manifest if args.only in (None, m["kind"])]
        if args.list:
            for m in todo:
                print(f"{m['bytes'] / 2 ** 20:8.1f} MB  {m['kind']:14s} {m.get('machine', ''):8s} {m['path']}")
            print(f"{len(todo)} files, {sum(m['bytes'] for m in todo) / 2 ** 20:.0f} MB")
            return
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
