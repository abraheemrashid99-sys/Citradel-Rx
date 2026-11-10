# Citadel-Rx: level snapshots and pretrained-weight caches

This branch is an archive. It holds the files of the revision that are too large for the `main` branch
of this repository; it shares no history with `main` and contains no code apart from the restore script.

| what | files | size |
|---|---|---|
| level snapshots `results/<folder>/<record>.levels.npz` (parameter trajectory of every level, one float16 snapshot per block, written by runs with `run.save_params=true`) | 14 | 1.3 GB |
| pretrained starting weights `cache/pretrain/<key>.pt` (written by the first run of each configuration, reused by the later ones) | 54 | 99 MB |

GitHub accepts at most 100 MB per file, so the files above 90 MB are cut into parts
(`<file>.part00`, `<file>.part01`, ...). `MANIFEST.json` lists every file with its size, its sha256 and
its parts, and the machine that wrote it.

## Putting the files back

In a clone of `main`:

    python scripts/restore_snapshots.py

or, in a checkout of this branch:

    python restore_snapshots.py --to /path/to/Citradel-Rx

The script joins the parts, checks the sha256 of every file and writes it next to its JSON record. By
hand, a file is its parts in order: `cat x.levels.npz.part* > x.levels.npz`, then compare
`sha256sum x.levels.npz` with `MANIFEST.json`.

## Do I need this?

No, not to reproduce the paper. Every value the paper takes from a snapshot (the paired displacement of
every level and of the anchor) is stored in the JSON record of the run on `main`, and `make paper`
rebuilds every number, table and figure from those records. The snapshots are kept so that the stored
displacements can be recomputed from the raw parameter trajectories (`scripts/paired_disp.py`), and the
caches so that a re-run starts from the same pretrained weights without retraining them. See
`docs/SNAPSHOTS.md` on `main`.

To clone the repository without downloading this branch:

    git clone --single-branch --branch main https://github.com/abraheemrashid99-sys/Citradel-Rx
