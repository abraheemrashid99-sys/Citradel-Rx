#!/usr/bin/env bash
# One click on the MacBook. In the VS Code terminal, inside the repository folder:
#
#     bash scripts/run_laptop.sh
#
# It (1) updates the code from GitHub, (2) creates the Python environment .venv (first time only),
# (3) runs the tests, (4) runs the laptop part of the revision campaign with the Mac kept awake,
# pushing results to GitHub every 45 minutes, and (5) pushes at the end. Every run is resumable:
# after any interruption (crash, Ctrl-C, reboot) just run the same command again.
#
# Options:  WORKERS=8  (parallel runs; default 8 of the 10 cores)
#           TIER=3     (1 = only what the revised claims need, 2 = + remaining items, 3 = + sensitivity)
set -uo pipefail
cd "$(dirname "$0")/.."
BRANCH="claude/fervent-cray-ah5drn"
WORKERS="${WORKERS:-8}"
TIER="${TIER:-3}"
mkdir -p logs
LOG="logs/laptop_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== Citadel-Rx revision, laptop part   $(date)"
echo "   machine: $(uname -sm), $(sysctl -n hw.ncpu 2>/dev/null || nproc) cores"

# ---- 1. code -------------------------------------------------------------------------------
if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
  git fetch origin "$BRANCH" && git checkout "$BRANCH" || { echo "!! cannot switch to branch $BRANCH"; exit 1; }
fi
git config user.email >/dev/null || git config user.email "laptop@citadel-rx.invalid"
git config user.name  >/dev/null || git config user.name  "Citadel-Rx laptop"
git pull --no-rebase --no-edit origin "$BRANCH" || echo "   (git pull failed: continuing with the local copy)"
echo "   code version: $(git rev-parse --short HEAD)"

# ---- 2. disk and Python environment ---------------------------------------------------------
free_mb=$(df -m . | awk 'NR==2 {print $4}')
echo "   free disk: ${free_mb} MB"
if [ ! -x .venv/bin/python ]; then
  if [ "$free_mb" -lt 1500 ]; then
    echo "!! about 1.5 GB of free disk is needed for the Python environment. Free space with:"
    echo "     conda clean -a -y ; pip cache purge ; rm -rf ~/Library/Caches/pip   (and empty the Bin)"
    exit 1
  fi
  PY=""
  for c in python3.12 python3.11 python3.13 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, platform; sys.exit(0 if sys.version_info >= (3, 10) and platform.machine() in ("arm64", "aarch64", "x86_64") else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "!! no Python >= 3.10 found. With conda:  conda create -y -n py311 python=3.11 && conda activate py311  then re-run."
    exit 1
  fi
  if [ "$(uname -s)" = "Darwin" ] && [ "$("$PY" -c 'import platform; print(platform.machine())')" != "arm64" ]; then
    echo "!! $PY is an Intel (x86_64) Python running under Rosetta; PyTorch needs a native arm64 Python."
    echo "   Install Miniforge (arm64) or use /usr/bin/python3 if it is 3.10+, then re-run."
    exit 1
  fi
  echo "   creating .venv with $("$PY" --version 2>&1)"
  "$PY" -m venv .venv || exit 1
  .venv/bin/python -m pip install -q --no-cache-dir --upgrade pip
  .venv/bin/python -m pip install --no-cache-dir -r requirements.txt || { echo "!! pip install failed"; exit 1; }
fi
PYV=.venv/bin/python
"$PYV" -c "import torch, numpy, scipy, yaml; print('   torch', torch.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)" \
  || { echo "!! the environment is broken: delete the .venv folder and re-run"; exit 1; }

# ---- 3. tests --------------------------------------------------------------------------------
echo "== tests (a few minutes)"
"$PYV" -m pytest tests -q -x || { echo "!! tests failed: stop here and send this log ($LOG)"; exit 1; }

# ---- 4. campaign -----------------------------------------------------------------------------
echo "== campaign: $WORKERS parallel runs, tiers up to $TIER (keep the Mac on power, lid open)"
AWAKE=""
command -v caffeinate >/dev/null 2>&1 && AWAKE="caffeinate -ims"
$AWAKE "$PYV" scripts/revision_fast.py --shard laptop --workers "$WORKERS" --tier "$TIER"

# ---- 5. final push ---------------------------------------------------------------------------
"$PYV" scripts/revision_fast.py --shard laptop --push-only
echo "== finished $(date). Results are on GitHub (branch $BRANCH)."
