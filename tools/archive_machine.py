#!/usr/bin/env python3
"""Add the files a machine holds that are not on GitHub to the `snapshots` branch.

A machine that ran part of a campaign keeps files that never reach the main branch: the level
snapshots (`*.levels.npz`, ignored by .gitignore and too large for GitHub as single files), the
pretrained-weight cache (`cache/`), and whatever else is untracked, ignored or modified in its clone
(figures and tables it generated from its own snapshots, console logs, partial records). This script
stores all of them on the `snapshots` branch under `machines/<machine>/`, the large ones cut into parts
of 90 MB, writes `manifests/<machine>.json` (size, sha256 and parts of every file) and pushes. It never
changes the clone it reads from, and it does not download the files that are already on the branch.

One command, from anywhere (python3 or python):

    python3 archive_machine.py --machine pool4 --repo /path/to/Citradel-Rx          # add --dry to only list

In a clone of main the same script is scripts/archive_machine_snapshots.py and --repo defaults to that
clone. On a machine that has only the clone, fetch the script from the branch without downloading the
branch:

    git init -q snaparch && cd snaparch \
      && git remote add origin https://github.com/abraheemrashid99-sys/Citradel-Rx \
      && git fetch -q --depth 1 --filter=blob:none origin snapshots \
      && git show FETCH_HEAD:tools/archive_machine.py > archive_machine.py

The script works in a staging directory of its own (a small git repository next to the clone, or the
current directory if that already is one with the remote `origin`), which it removes when everything is
verified on the branch.

How it works: every file (or part) is written as a git blob with `git hash-object`; the tree of the
branch tip is read into a private index (only tree objects are needed, the blobs of the other machines
stay on GitHub), the new entries are added, and the resulting commit is pushed on top of the tip. If
another machine pushed in between, the push is rejected; the script then fetches the new tip, rebuilds
the commit on it (the paths of two machines never overlap) and pushes again. Files are pushed in groups
of at most 400 MB; the manifest goes in with the last group, so a manifest on the branch means that all
of its files are there. At the end the script reads the branch back and checks every blob.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

BRANCH = "snapshots"
PART = 90 * 2 ** 20            # GitHub refuses files over 100 MB
GROUP = 400 * 2 ** 20          # bytes per push
SKIP = ["*/__pycache__/*", "__pycache__/*", "*.pyc", ".pytest_cache/*", "*/.pytest_cache/*", ".venv/*", "*.egg-info/*",
        "paper/main.aux", "paper/main.log", "paper/main.out", "paper/build.log", ".DS_Store", "*/.DS_Store",
        "*.restoring", ".git/*"]
FAST = ["-c", "core.compression=0", "-c", "core.looseCompression=0", "-c", "pack.window=0", "-c", "gc.auto=0",
        "-c", "http.postBuffer=1048576000"]


WORK = None                    # the staging repository; set in main()


def git(*args, cwd=None, data=None, check=True, env=None, quiet=True):
    r = subprocess.run(["git", *FAST, *args], cwd=cwd or WORK, input=data, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE if quiet else None)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args[:3])} ... failed:\n{(r.stderr or b'').decode(errors='replace')}")
    return r


def rmtree(path: str) -> None:
    """Remove a directory tree; git marks its object files read-only, which stops a plain rmtree on Windows."""
    def fix(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=fix)
    else:
        shutil.rmtree(path, onerror=fix)


def skip(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in SKIP)


def inventory(repo: str) -> list[str]:
    """Every file of the clone that is not exactly what its HEAD commit holds: untracked, ignored or modified."""
    out = set()
    for args in (["ls-files", "-z", "--others"], ["ls-files", "-z", "--modified"]):
        for rel in git(*args, cwd=repo).stdout.decode().split("\0"):
            if rel and not skip(rel) and os.path.isfile(os.path.join(repo, rel)) and not os.path.islink(os.path.join(repo, rel)):
                out.add(rel)
    return sorted(out)


def kind_and_path(rel: str, machine: str) -> tuple[str, str]:
    if rel.endswith(".levels.npz"):
        return "snapshot", rel                               # goes back next to its JSON record
    if rel.startswith("cache/"):
        return "pretrain_cache", rel                         # goes back into cache/ (only on request, per machine)
    return "machine_file", f"results_archive/{machine}/{rel}"   # never read by the analysis


def fetch_tip() -> str:
    for attempt in range(5):
        r = git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", BRANCH, check=False)
        if r.returncode == 0:
            return git("rev-parse", "FETCH_HEAD").stdout.decode().strip()
        time.sleep(2 * 2 ** attempt)
    sys.exit("cannot fetch the snapshots branch:\n" + r.stderr.decode(errors="replace"))


def commit_on(tip: str, entries: dict[str, str], message: str) -> str | None:
    """A commit whose tree is the tree of `tip` plus `entries` (stored path -> blob); None if nothing changes."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"))
        git("read-tree", tip, env=env)
        info = "".join(f"100644 {sha}\t{path}\0" for path, sha in sorted(entries.items()))
        git("update-index", "-z", "--index-info", data=info.encode(), env=env)
        tree = git("write-tree", "--missing-ok", env=env).stdout.decode().strip()
    if tree == git("rev-parse", f"{tip}^{{tree}}").stdout.decode().strip():
        return None
    if git("config", "--bool", "commit.gpgsign", check=False).stdout.decode().strip() == "true":
        r = git("commit-tree", tree, "-p", tip, "-S", "-m", message, check=False)   # sign as `git commit` would here
        if r.returncode == 0:
            return r.stdout.decode().strip()
        print("   signing the commit failed, committing unsigned", flush=True)
    return git("commit-tree", tree, "-p", tip, "-m", message).stdout.decode().strip()


def push_group(entries: dict[str, str], message: str) -> str:
    """Put `entries` on the branch, whatever the other machines push meanwhile. Returns the new tip."""
    last = ""
    for attempt in range(12):
        tip = fetch_tip()
        commit = commit_on(tip, entries, message)
        if commit is None:
            return tip                                       # already there (a re-run)
        r = git("push", "-q", "origin", f"{commit}:refs/heads/{BRANCH}", check=False)
        if r.returncode == 0:
            return commit
        last = r.stderr.decode(errors="replace")
        print(f"   push not accepted (attempt {attempt + 1}): {last.strip().splitlines()[-1] if last.strip() else '?'}; retrying", flush=True)
        time.sleep(min(60, 3 * 2 ** attempt))
    sys.exit("push failed 12 times:\n" + last)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, help="the name of this machine, e.g. pool4, helper, windows")
    here = os.path.dirname(os.path.abspath(__file__))
    default_repo = os.path.dirname(here) if os.path.isdir(os.path.join(os.path.dirname(here), "citadel_rx")) else None
    ap.add_argument("--repo", default=default_repo, required=default_repo is None,
                    help="the clone of Citradel-Rx on this machine (read only)")
    ap.add_argument("--workdir", default=None, help="staging directory (default: next to the clone)")
    ap.add_argument("--dry", action="store_true", help="list what would be stored and stop")
    ap.add_argument("--trailer", default="", help="text appended to the commit messages (attribution lines)")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    if not os.path.exists(os.path.join(repo, ".git")):
        sys.exit(f"{repo} is not a git clone")
    if not all(c.isalnum() or c in "-_" for c in args.machine) or args.machine == "cloud":
        sys.exit("--machine: letters, digits, - and _ only (and not 'cloud', the coordinating machine)")

    global WORK
    inside = lambda d: os.path.abspath(d) == repo or os.path.abspath(d).startswith(repo + os.sep)
    cwd = os.getcwd()
    if args.workdir:
        WORK = os.path.abspath(args.workdir)
    elif (os.path.isdir(os.path.join(cwd, ".git")) and not inside(cwd)
          and subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd, capture_output=True).returncode == 0):
        WORK = cwd                                           # an existing staging repository (see the docstring)
    else:
        WORK = os.path.join(os.path.dirname(repo), f".snaparch_{args.machine}")
    ours = WORK == os.path.join(os.path.dirname(repo), f".snaparch_{args.machine}")   # removed once all is verified
    if inside(WORK):
        sys.exit("the staging directory must be outside the clone")
    if not git("branch", "-r", "--contains", "HEAD", cwd=repo, check=False).stdout.strip():
        print("WARNING: the clone has commits that are on no remote branch; push them, this script only stores "
              "files that are untracked, ignored or modified", flush=True)

    files = inventory(repo)
    total = sum(os.path.getsize(os.path.join(repo, f)) for f in files)
    print(f"{args.machine}: {len(files)} files not on GitHub, {total / 2 ** 20:.0f} MB "
          f"(HEAD {git('rev-parse', '--short', 'HEAD', cwd=repo).stdout.decode().strip()})", flush=True)
    if args.dry or not files:
        for f in files:
            print(f"  {os.path.getsize(os.path.join(repo, f)):>11d}  {kind_and_path(f, args.machine)[0]:14s} {f}")
        return

    if not os.path.isdir(os.path.join(WORK, ".git")):
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo, capture_output=True, text=True).stdout.strip()
        if not url:
            sys.exit(f"{repo} has no remote 'origin'")
        os.makedirs(WORK, exist_ok=True)
        git("init", "-q", ".")
        git("remote", "add", "origin", url)

    manifest, blobs = [], []                                 # blobs: (stored path, blob sha, bytes)
    for rel in files:
        src = os.path.join(repo, rel)
        size = os.path.getsize(src)
        kind, dest = kind_and_path(rel, args.machine)
        stored = f"machines/{args.machine}/{rel}"
        h, parts, read = hashlib.sha256(), [], 0
        with open(src, "rb") as f:
            for i in range(max(1, -(-size // PART))):        # one blob for a small (or empty) file, else 90 MB parts
                data = f.read(PART)
                h.update(data)
                read += len(data)
                p = stored if size <= PART else f"{stored}.part{i:02d}"
                sha = git("hash-object", "-w", "--stdin", data=data).stdout.decode().strip()
                parts.append(p)
                blobs.append((p, sha, len(data)))
            if read != size or f.read(1):
                sys.exit(f"{rel} changed while it was read; nothing pushed")
        manifest.append({"path": dest, "kind": kind, "machine": args.machine, "source": rel, "bytes": size,
                         "sha256": h.hexdigest(), "parts": parts})
    mpath = f"manifests/{args.machine}.json"
    msha = git("hash-object", "-w", "--stdin", data=(json.dumps(manifest, indent=1) + "\n").encode()).stdout.decode().strip()

    groups, cur, cur_bytes = [], {}, 0
    for p, sha, n in sorted(blobs, key=lambda b: (b[2] > 2 ** 20, b[0])):   # small files first
        if cur and cur_bytes + n > GROUP:
            groups.append(cur)
            cur, cur_bytes = {}, 0
        cur[p] = sha
        cur_bytes += n
    cur[mpath] = msha
    groups.append(cur)

    trailer = ("\n\n" + args.trailer.replace("\\n", "\n")) if args.trailer else ""
    for g, entries in enumerate(groups):
        mb = sum(n for p, _, n in blobs if p in entries) / 2 ** 20
        print(f"pushing group {g + 1}/{len(groups)}: {len(entries)} files, {mb:.0f} MB ...", flush=True)
        push_group(entries, f"{args.machine}: files not on main, group {g + 1} of {len(groups)}\n\n"
                            f"Level snapshots, pretrained-weight cache and other untracked or ignored files of the machine\n"
                            f"{args.machine}; manifests/{args.machine}.json lists them with size, sha256 and parts.{trailer}")

    tip = fetch_tip()
    want = dict((p, sha) for p, sha, _ in blobs)
    want[mpath] = msha
    listing = git("ls-tree", "-r", "-z", tip, "--", f"machines/{args.machine}", mpath).stdout.decode().split("\0")
    have = {line.split("\t", 1)[1]: line.split()[2] for line in listing if line}
    missing = [p for p, sha in want.items() if have.get(p) != sha]
    if missing:
        sys.exit(f"NOT VERIFIED: {len(missing)} files are not on the branch as expected, e.g. {missing[:3]}")
    kinds = {}
    for m in manifest:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    if ours:
        rmtree(WORK)
    print(f"VERIFIED {args.machine}: {len(manifest)} files ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}), "
          f"{total / 2 ** 20:.0f} MB in {len(blobs)} blobs on {BRANCH} at {tip[:12]}")


if __name__ == "__main__":
    main()
