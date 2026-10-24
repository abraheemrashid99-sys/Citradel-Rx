# Citadel-Rx

**Continuum-Memory Test-Time-Learning Receivers with Certified Robustness to Pilot Poisoning**

Online and meta-learned neural receivers learn from pilots at test time, which
turns the pilot stream into a training set that any transmitter in range can
poison (Wu et al. 2024). Citadel-Rx splits the receiver's adaptable parameters
into memory levels updated at periods aligned with the channel's coherence
structure; transfer to slower levels passes a *physics-consistency gate*
(model-based residual, CRC-verified decisions, channel-prior likelihood), so the
slow levels become certifiably hard to poison and re-anchor the fast level after
an attack stops.

This repository is the complete, reproducible code for the paper: the receiver,
every baseline, every attack, the experiment protocol (E1–E8), the numerical
evaluation of the theory, and the scripts that regenerate every number in the
manuscript from JSON run records.

Read `docs/RESEARCH_NOTES.md` first: it records what the literature and the
reference code actually contain, what is unverified, and why each design
decision was made.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU is enough for the synthetic / COST2100 experiments
pip install -e .
pytest -q                                # ~1 minute on a laptop CPU
```

GPU (for E6 Sionna PUSCH and the E4/E8 DeepMIMO runs): install a CUDA build of
PyTorch, then `pip install -e .[sionna]`.

## Run

```bash
# smoke test (seconds)
python scripts/run_experiment.py --quick --set receiver.type=citadel attack.type=a1

# one configuration, three seeds
python scripts/run_experiment.py configs/e3_a1_defence.yaml --seeds 0 1 2

# override anything with dotted keys
python scripts/run_experiment.py configs/e3_a1_defence.yaml --set receiver.type=meta_deepsic attack.sir_db=5

# full sweeps for a paper experiment (writes results/<exp>/*.json)
python scripts/sweep.py e3 --seeds 0 1 2 --workers 4

# regenerate every table / figure from the run records
python scripts/paper_numbers.py results/
```

## What is implemented

| Component | File | Notes |
|---|---|---|
| DeepSIC backbone (functional) | `citadel_rx/receivers/deepsic.py` | architecture + hyper-parameters identical to the TWC 2023 reference code; users stacked so every operation is a pure function of a parameter dict |
| Online DeepSIC, Meta-DeepSIC (MAML), drift-detection receiver, MedBN-style robust-gradient receiver, LMMSE | `receivers/online.py`, `meta.py`, `drift.py`, `robust.py`, `lmmse.py` | baselines |
| Citadel-Rx and single-memory Titans ablation | `receivers/citadel.py` | K levels, DGD fast rule, surprise-driven rate, gated consolidation, re-anchoring |
| Gates G1 / G2 / G3 with self-generated thresholds | `gates/physics.py` | |
| A1 (white-box + transfer), A2, A3, gate-aware adaptive attacker, equal-power jamming control | `attacks/pilot_poisoning.py` | PGD through an unrolled differentiable adaptation map |
| Channels: SED cosine fading (reference), Jakes AR(1), static, PA nonlinearity, COST2100 | `channels/base.py` | COST2100 traces are the public files from the reference repository |
| CRC sub-block framing | `coding/crc.py` | 3GPP polynomials |
| Metrics (SER/BER/BLER, Δ_dB, recovery time, gate ROC) and JSON run records | `eval/` | |
| Theory: adversary term, retention certificate, measured Lipschitz constant | `theory/bound.py` | |

## Experiment map (proposal §2.10)

| Exp | Config | Status on laptop CPU |
|---|---|---|
| E1 reproduce attack | `configs/e1_reproduce_attack.yaml` | runs (minutes per seed) |
| E2 clean parity | `configs/e2_clean_parity.yaml` | runs |
| E3 A1 defence vs SIR | `configs/e3_a1_defence.yaml` | runs |
| E4 A2 drift spoofing | `configs/e4_a2_drift_spoof.yaml` | runs (DeepMIMO variant: GPU) |
| E5 A3 boiling frog + certificate | `configs/e5_a3_boiling_frog.yaml` | runs (long) |
| E6 Sionna PUSCH + LDPC | `configs/e6_sionna_pusch.yaml` | GPU only |
| E7 adaptive attacker | `configs/e7_adaptive_attacker.yaml` | runs |
| E8 channel-prediction variant | planned | GPU |

## Reproducibility

Every run writes `results/<exp>/<name>_s<seed>_<confighash>.json` with the full
config, git commit, per-block metrics and gate statistics. `scripts/paper_numbers.py`
rebuilds all tables and figures from those files and nothing else.
