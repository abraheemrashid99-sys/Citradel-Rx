# Citadel-Rx: level snapshots, pretrained-weight caches and other files kept off `main`

This branch is an archive. It holds the files of the revision that are too large for the `main` branch
of this repository, or that only one machine of the campaign held. It shares no history with `main`
and contains no code apart from the two scripts below.

| kind | what | where it goes back |
|---|---|---|
| `snapshot` | level snapshots `<record>.levels.npz`: the parameter trajectory of every level, one float16 snapshot per block, written by runs with `run.save_params=true` | `results/<folder>/`, next to the JSON record |
| `pretrain_cache` | pretrained starting weights `cache/pretrain/<key>.pt`, written by the first run of each configuration on a machine and reused by its later runs | `cache/pretrain/` (per machine) |
| `machine_file` | anything else a machine held that is not on `main`: figures and tables it drew from its own snapshots, console logs | `results_archive/<machine>/`, which the analysis never reads |

## Layout

- `MANIFEST.json` and `results/`, `cache/`: the files of the cloud machine that coordinated the campaign.
- `manifests/<machine>.json` and `machines/<machine>/`: the files of every other machine (pool1, pool2, ...),
  stored under the path they had in that machine's clone.
- GitHub accepts at most 100 MB per file, so every file above 90 MB is cut into parts
  (`<file>.part00`, `<file>.part01`, ...).
- Each manifest lists its files with the kind, the machine, the size, the sha256 and the parts. A manifest
  is pushed after its files, so a manifest on the branch means all of its files are there.

## Putting the files back

In a clone of `main` (downloads only the parts it needs):

    python scripts/restore_snapshots.py                  # every level snapshot, of every machine
    python scripts/restore_snapshots.py --list           # what is on the branch
    python scripts/restore_snapshots.py --machine pool4  # the snapshots of one machine
    python scripts/restore_snapshots.py --only pretrain_cache --machine cloud

or, in a checkout of this branch: `python restore_snapshots.py --to /path/to/Citradel-Rx`.

The script joins the parts, checks the sha256 of every file and writes it to its place. By hand, a file
is its parts in order: `cat x.levels.npz.part* > x.levels.npz`, then compare `sha256sum x.levels.npz`
with the manifest.

## Adding the files of another machine

`tools/archive_machine.py` stores everything a machine's clone holds that is not on GitHub (untracked,
ignored or modified files) under `machines/<machine>/` and pushes it, without downloading what is
already here and without changing the clone. Its docstring has the three commands to run.

## Do I need this?

No, not to reproduce the paper. Every value the paper takes from a snapshot (the paired displacement of
every level and of the anchor) is stored in the JSON record of the run on `main`, and `make paper`
rebuilds every number, table and figure from those records. The snapshots are kept so that the stored
displacements can be recomputed from the raw parameter trajectories (`scripts/paired_disp.py`), and the
caches so that a re-run on the same machine type starts from the same pretrained weights. See
`docs/SNAPSHOTS.md` on `main`.

To clone the repository without downloading this branch:

    git clone --single-branch --branch main https://github.com/abraheemrashid99-sys/Citradel-Rx
