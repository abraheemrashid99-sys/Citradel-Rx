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

## E2 clean parity → clean *gain* (26 Oct; 3 seeds; `results/e2/`; full table: `docs/TABLES.md`)

Clean SER at 10 dB, no attack, 100 blocks: cosine fading — LMMSE 0.054, online 0.059,
Meta 0.053, drift 0.065, MedBN 0.064, single-memory Titans 0.053, **Citadel 0.030**;
Jakes f_D T_b = 0.05 — LMMSE 0.0095, online 0.011, Meta 0.010, **Citadel 0.0053**. Citadel
is the best receiver at every SNR from 4 to 12 dB on both fading models (at 14–16 dB all
receivers are at the 1e-4 floor of a 100-block run and differences are noise). Since
pre-trained online DeepSIC is *not* better than online DeepSIC (0.074 vs 0.069 pre-attack
in the SIR-10 runs), the gain comes from the memory hierarchy: the slow levels accumulate
channel knowledge across blocks and the proximal anchor turns it into a prior for the
per-block fit from 100 pilot symbols. The proposal's parity target (≤ 0.2 dB cost) is
replaced by a ~2 dB clean gain, which must be presented as a second contribution and
ablated (see below).

## E1 reproduction of the attack (3 seeds; `results/e1/`)

White-box A1 destroys online DeepSIC and Meta-DeepSIC at ≤ 0 dB SIR on both the linear
and the tanh-PA channel (0.78–0.97) and still costs 4–5× at 10 dB; Meta-DeepSIC never
recovers (30 blocks censored, post-attack SER 0.18–0.20 vs 0.06–0.07 clean). The transfer
attacker is much weaker on the linear channel (0.26–0.30) but nearly as strong as white-box
on the nonlinear channel (0.67–0.77): the surrogate transfers better when the receiver's
nonlinearity is what the attack exploits. (The Wu et al. "3.09 dB at 15 dB" figure uses a
per-symbol ε budget; the Δ_dB comparison comes from E3snr.)

## Ablations under white-box A1 at SIR 0 dB (3 seeds; `results/ablate/`)

| variant | clean (pre) | attack window | recovery | rejection | false rej. |
|---|---|---|---|---|---|
| full Citadel (K=3, C=1/8/32, G1+G2+G3, DGD, surprise η, proximal anchor, fallback) | 0.039 | **0.026** | 0 | 0.97 | 0.006 |
| K=1 (single memory, gates active, no slow level to fall back on) | 0.068 | 0.788 | 6.3 | 0.98 | 0.222 |
| K=2 (C=1/8) | 0.038 | 0.050 | 0 | 0.97 | 0.006 |
| K=4 (C=1/4/16/64) | 0.038 | 0.029 | 0 | 0.97 | 0.011 |
| periods misaligned with coherence (C=1/3/7) | 0.037 | 0.031 | 0 | 0.95 | 0.000 |
| no gates | 0.039 | 0.540 | 0 | 0.00 | – |
| G1 only | 0.039 | 0.031 | 0 | 0.97 | 0.000 |
| G2 only | 0.039 | 0.028 | 0 | 0.97 | 0.000 |
| G3 only | 0.039 | 0.507 | 0 | 0.08 | 0.000 |
| adaptable = last layer only | 0.052 | 0.029 | 0 | 0.98 | 0.011 |
| plain GD fast rule | 0.066 | 0.048 | 0 | 0.90 | 0.072 |
| Adam without the delta-rule decay | 0.040 | 0.030 | 0 | 0.97 | 0.006 |
| fixed η (no surprise) | 0.040 | 0.027 | 0 | 0.97 | 0.011 |
| **no anchoring** (λ = 0, β = 0) | **0.058** | 0.031 | 0 | 0.99 | 0.178 |
| jump re-anchoring (Eq. 72, β = 0.25) instead of proximal | 0.058 | 0.031 | 0 | 0.97 | 0.150 |
| **no fallback** (rejected block still detected by the fast level) | 0.040 | **0.245** | 0 | 0.97 | 0.011 |
| keep (not revert) the fast update of a rejected block | 0.040 | 0.025 | 0 | 1.00 | 0.017 |

Reading:
* **The proximal anchor is the clean gain.** Without it clean SER goes 0.039 → 0.058
  (online DeepSIC: 0.069), and the jump re-anchoring of the CMS paper is no better than no
  anchoring at all; both also raise clean false rejections to 15–18 % because the fast
  level drifts away from the memory it is compared against. Continuous proximal pull
  toward the slow projection is what turns the memory hierarchy into a prior.
* **The fallback is the attack-window protection.** Reverting a rejected update alone
  leaves the data of that block to a poisoned fast level (0.245); detecting it with the
  slow projection brings it to 0.026. Whether the rejected update is reverted or kept
  makes no difference once the fallback is active (0.025 vs 0.026).
* **The slow memory is the mechanism.** With a single level the gates still fire (0.98)
  but there is nothing uncorrupted to fall back on, and the receiver collapses like online
  DeepSIC; two levels already give 0.050 and three or four give the clean level.
* G1 or G2 alone defeats the non-adaptive A1 (E7 shows G1 alone does not survive a
  gate-aware attacker); G3 alone is useless against A1, as the proposal predicted — its
  role is A2/A3.
* Honest negatives: the delta-rule decay and the surprise-driven rate each buy only
  2–3 % clean SER; period alignment with the coherence structure does not matter on this
  channel; plain GD costs clean performance and false rejections.
* (The `no_reanchor` row of the first GPU sweep set the jump weight while the anchoring
  mode is proximal and is discarded; the four corrected variants above were run here.)

## E4 — A2 targeted drift spoofing, Jakes f_D T_b = 0.05, SIR 0 dB (3 seeds; `results/e4/`)

Online DeepSIC and Meta-DeepSIC lose all users (window SER 0.44 / 0.35; user-0 SER 0.16 /
0.11, others 0.16 / 0.13 — with λ = 1 the attack is not very selective), Citadel stays at
its clean level (0.010) with 82 % rejection. G3 on/off makes no measurable difference:
G1/G2 catch this attack anyway. G3's value has to be shown on A3 (E5) or reported as
not needed in these settings.

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
