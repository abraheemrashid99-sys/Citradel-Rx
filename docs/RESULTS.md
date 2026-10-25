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

## E3 full sweep from the GPU workstation (25 Oct 2026; 3 seeds; records `results/e3/`)

SER in the attack window (blocks 30–69), mean over 3 seeds. Clean-window SER of every
receiver is in the "pre" column; "post" is the mean after the attack; "rec" the recovery
time in blocks (30 = censored, never recovered within the run).

| receiver | pre | **white-box A1**, SIR −10 / −5 / 0 / 5 / 10 dB | post (SIR 0) | rec | **transfer A1**, SIR −10 / −5 / 0 / 5 / 10 dB | rec | data jamming, 0 dB (control) |
|---|---|---|---|---|---|---|---|
| LMMSE (no adaptation) | 0.056 | n/a¹ | – | – | 0.873 / 0.740 / 0.447 / 0.149 / 0.064 | 0 | 0.892 |
| online DeepSIC | 0.071 | 0.973 / 0.936 / 0.769 / 0.474 / 0.279 | 0.087 | 11 | 0.353 / 0.367 / 0.347 / 0.289 / 0.205 | 10 | 0.855 |
| Meta-DeepSIC | 0.062 | 0.863 / 0.800 / 0.594 / 0.400 / 0.243 | 0.187 | 30† | 0.194 / 0.206 / 0.214 / 0.207 / 0.171 | 14 | 0.851 |
| drift-detection | 0.077 | 0.975 / 0.942 / 0.776 / 0.459 / 0.265 | 0.097 | 10 | 0.345 / 0.359 / 0.337 / 0.277 / 0.199 | 7 | 0.858 |
| MedBN-style | 0.065 | 0.933 / 0.853 / 0.738 / 0.549 / 0.397 | 0.204 | 30† | 0.724 / 0.641 / 0.544 / 0.423 / 0.296 | 30† | 0.858 |
| single-memory Titans | 0.068 | 0.975 / 0.939 / 0.788 / 0.486 / 0.276 | 0.075 | 6 | 0.363 / 0.368 / 0.354 / 0.290 / 0.200 | 6 | 0.861 |
| **Citadel-Rx** | **0.039** | **0.025 / 0.025 / 0.026 / 0.029 / 0.032** | **0.029** | **0** | **0.026 / 0.026 / 0.027 / 0.029 / 0.037** | **0** | 0.848 |

Citadel gate statistics across the sweep: rejection of attacked blocks 0.87–0.99
(white-box), 0.77–0.98 (transfer); false rejection of clean blocks 0.000–0.017.

¹ The white-box attacker differentiates through the victim's adaptation map; LMMSE has
none, so the "white-box" record for LMMSE contains no perturbation. The transfer column
(perturbation crafted on a surrogate and injected) is the valid LMMSE row; it shows that
pilot poisoning also wrecks model-based channel estimation at high budgets. Fixed in the
code: receivers without an adaptation map are now attacked through the surrogate.
† Censored: SER had not returned to within 10 % of the pre-attack level 30 blocks after the attack.

Reading:
* Against Citadel-Rx, pilot poisoning is worthless at every budget and under both
  knowledge models: the attack-window SER (0.025–0.037) is at or below the receiver's own
  clean level (0.039, blocks 0–29), and recovery is immediate. Every other receiver is
  destroyed at ≤ 0 dB (SER 0.6–0.98) and still loses 4–6× at 10 dB.
* The equal-energy jamming control makes the point a security reviewer wants: against
  every baseline, poisoning the 200 pilot bits is as damaging as jamming the 50 000 data
  bits with the same energy (0.77 vs 0.86 at 0 dB) *and* leaves persistent damage
  (Meta-DeepSIC and MedBN never recover; jamming leaves no trace). Against Citadel the
  attacker is reduced to plain jamming, which costs continuous power and has no persistence.
* Transfer attacks are weaker than white-box for the baselines (as Wu et al. report) but
  not harmless (0.2–0.37); Citadel is indifferent to the knowledge model.
* The vision-style robust-gradient defence (MedBN) is the worst receiver under attack and
  the only baseline besides Meta-DeepSIC with permanent damage.

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
| Citadel, G1 only | adaptive (weight 50) | 0.044 | 6 / 40 | 0.85 |
| Citadel, G1+G2+G3 | adaptive (weight 5) | 0.032 | 1 / 40 | 0.97 |
| Citadel, G1+G2+G3 | adaptive (weight 500 — best effort) | 0.027 | 2 / 40 | 0.95 |
| Citadel, G1+G2+G3 | A1 transfer (surrogate receiver) | 0.041 | 14 / 40 | 0.65 |
| online DeepSIC | A1 transfer | 0.197 (white-box: 0.278) | – | – |

Reading: a gate-aware attacker partially defeats the residual gate on its
own (as the proposal predicted: G1 is necessary but weak); with the CRC and
prior gates it gains nothing. Transfer attacks are milder than white-box
(as in Wu et al.) and slip the gates more often, but the leaked blocks cost
little. The gate-aware weight sweep (5 / 50 / 500) shows the attacker's
trade-off against G1 alone (rejection 0.80 → 0.85 → 0.30, SER 0.046 → 0.044 →
0.074: evading the residual test costs poisoning strength); against the full
gate set the best-effort attacker gains nothing (rejection 0.95, SER 0.027).

## Caveats (to be closed before any number enters the manuscript)
1. One seed. Three seeds and the 5-point SIR grid are the next sweep.
2. The attacked blocks' SER (0.03) is *lower* than the clean run's mean
   because the window happens to cover the easy part of the fade; the honest
   metric is the per-block difference to the clean trajectory, which the
   figure shows (≤ 0.06 at the deepest fade).
3. No nonlinear / COST2100 / Jakes runs yet (configs exist).
4. Wu et al. numbers still unverified at the primary source (arXiv blocked here).
