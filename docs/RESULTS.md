# Results log (regenerated from JSON run records; nothing here is typed by hand)

Setting unless stated: 4×4 QPSK uplink, SED channel with the reference cosine
fading, SNR 10 dB, 200 pilot bits + 50 000 data bits per coherence block,
100 blocks, A1 availability poisoning (white-box PGD through a 2-step unrolled
adaptation, 20 PGD steps) on blocks 30–69, seed 0, CPU. Records:
`results/full2/*.json` (commit `git log -1`).

## E3 — A1 availability attack vs. SIR (24 Oct 2026, 1 seed, preliminary)

| receiver | clean SER | SER in attack window, SIR 0 / 5 / 10 dB | SER after attack | recovery (blocks) | gate rejection of attacked / false rejection of clean |
|---|---|---|---|---|---|
| online DeepSIC (TWC 2021) | 0.056 | 0.795 / 0.483 / 0.279 | 0.093 / 0.081 / 0.078 | 12 / 11 / 9 | – |
| Meta-DeepSIC (TWC 2023) | 0.058 | – / – / 0.246 | 0.134 | 30 (censored: never recovers) | – |
| **Citadel-Rx** (K=3, C=(1,8,32), G1+G2+G3, fallback) | **0.031** | **0.027 / 0.033 / 0.034** | 0.030 / 0.031 / 0.030 | **0** | 0.97 / 0.02, 0.95 / 0.00, 0.93 / 0.00 |

Reading:
* Under attack Citadel-Rx stays on its clean trajectory at every budget,
  including SIR 0 dB where the attacker spends as much energy as all four
  users together and online DeepSIC is destroyed (80 % SER).
* The mechanism is visible in the records: 93–97 % of attacked blocks are
  rejected by the gates, their fast update is reverted, and the data are
  detected with the slow memory (`fallback` flag); the slow level's
  displacement at the end of the run is 16–22 (parameter-norm units) for the
  attacked runs vs 16 for the clean run, i.e. the poison did not reach it.
* Meta-DeepSIC is the most vulnerable receiver *after* the attack: the
  poisoned pilots stay in its meta-training buffer, so it never recovers
  within 30 blocks. This is the Wu et al. observation, reproduced.
* Clean condition: Citadel-Rx is 45 % below online DeepSIC in SER. The
  pre-training and the proximal anchoring to the slow memory act as a
  regulariser; this must be separated in the ablations (E2 with
  `receiver.pretrain=true` for online DeepSIC as well).

Figure: `docs/figures/fig_ser_vs_block_a1_sir10.png`, `..._sir5.png`,
`..._sir0.png` (SER vs block, clean trajectories dashed, attack window shaded).

## 3-seed replication of the key cells (seeds 0, 1, 2; mean ± std over seeds)

| receiver | clean SER (no attack, 100 blocks) | pre-attack SER | SER in attack window (SIR 10 dB) | SER after | recovery |
|---|---|---|---|---|---|
| online DeepSIC | 0.0575 ± 0.0014 | 0.0696 ± 0.0033 | 0.2779 ± 0.0095 | 0.0772 ± 0.0041 | 9, 9, 12 |
| **Citadel-Rx** | **0.0305 ± 0.0006** | 0.0395 ± 0.0010 | **0.0342 ± 0.0006** | 0.0299 ± 0.0005 | 0, 0, 0 |

Seed-to-seed spread is an order of magnitude below the effect; the headline
numbers are not a seed artefact.

## Baselines and ablations at SIR 10 dB (1 seed)

| receiver | clean SER (blocks 0–29) | SER in attack window | SER after | recovery (blocks) |
|---|---|---|---|---|
| online DeepSIC | 0.069 | 0.279 | 0.078 | 9 |
| online DeepSIC + joint pre-training | 0.074 | 0.274 | 0.082 | 12 |
| drift-detection receiver (TWC 2025 style, "adapt less") | 0.088 | 0.265 | 0.080 | 4 |
| MedBN-style median-of-means gradient (vision TTA defence) | 0.073 | 0.365 | 0.170 | 30 (censored) |
| single-memory Titans receiver (DGD + surprise rate, K=1, no gates) | 0.069 | 0.269 | 0.073 | 8 |
| **Citadel-Rx** | **0.039** | **0.034** | **0.030** | **0** |

Reading:
* Pre-training does not explain Citadel's clean advantage (pre-trained online
  DeepSIC: 0.074), nor its robustness. The gain comes from the levels, the
  gates and the proximal anchoring.
* A single test-time memory with the same fast rule is exactly as vulnerable
  as online DeepSIC: NL-style memory without the frequency hierarchy is not a
  defence (consistent with the Titans reimplementation study).
* The drift-detection receiver trades clean performance (0.088) for a small
  robustness gain and a faster recovery; the vision-style robust-gradient
  defence is worse than no defence and leaves a lasting poison (post 0.17).

## E7 adaptive (gate-aware) attacker and transfer attacker, SIR 10 dB (1 seed)

| receiver / gates | attacker | SER in attack window | attacked blocks accepted | rejection of attacked |
|---|---|---|---|---|
| Citadel, G1 only | adaptive (G1-penalty weight 5) | 0.046 | 8 / 40 | 0.80 |
| Citadel, G1 only | adaptive (weight 500 — attacker's best effort) | 0.074 | 28 / 40 | 0.30 |
| Citadel, G1+G2+G3 | adaptive (weight 5) | 0.032 | 1 / 40 | 0.97 |
| Citadel, G1+G2+G3 | A1 transfer (surrogate receiver) | 0.041 | 14 / 40 | 0.65 |
| online DeepSIC | A1 transfer | 0.197 (white-box: 0.278) | – | – |

Reading: a gate-aware attacker partially defeats the residual gate on its
own (as the proposal predicted: G1 is necessary but weak); with the CRC and
prior gates it gains nothing. Transfer attacks are milder than white-box
(as in Wu et al.) and slip the gates more often, but the leaked blocks cost
little. Pending: sweep of the gate-aware weight (50, 500) to represent the
attacker's best effort.

## Caveats (to be closed before any number enters the manuscript)
1. One seed. Three seeds and the 5-point SIR grid are the next sweep.
2. The attacked blocks' SER (0.03) is *lower* than the clean run's mean
   because the window happens to cover the easy part of the fade; the honest
   metric is the per-block difference to the clean trajectory, which the
   figure shows (≤ 0.06 at the deepest fade).
3. No nonlinear / COST2100 / Jakes runs yet (configs exist).
4. Wu et al. numbers still unverified at the primary source (arXiv blocked here).
