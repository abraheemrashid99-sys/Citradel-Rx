# Response to the reviewer audit

Every item of the audit is listed with what changed and where. Section, table and equation
names refer to the revised manuscript (`paper/main.tex`); the submitted version is
`paper/main_submitted.tex`. Every result number in the revision is computed from the run records
by `scripts/revision_numbers.py`; the experiments are listed in `scripts/revision_fast.py` and their
reductions against the full plan in `docs/REVISION_PLAN.md`.

A correction found during the revision, before any test result: the consolidation rule of the
first version divided the gated sum of buffered states by max(1, sum of verdicts). When most
blocks of a period are rejected, which is what happens under attack, this pulls the slow level
toward the origin, so Lemma 1 (convex combination) did not hold for the implementation. The
revision uses the normalised gated mean (Eq. (consol), `citadel.consol_mean=normalized`,
`tests/test_consolidation.py`); every revision result uses it, and the first-version rule remains
selectable so the old records keep their meaning.

## A. Theory

| Item | Change | Where |
|---|---|---|
| A1 neither branch applies | "non-expansive" deleted. Theorem 1 is now a retention theorem with (i) a design bound D_K <= 2 kappa_K n(W) from the consolidation clip, which needs no measured quantity and holds for any attacker and any verdicts, and (ii) the structural bound for unequal verdicts. The contraction branch is a remark, stated as not applying because the measured joint constant exceeds one | Sec. IV, Thm. 1, Remark 1 |
| A2 equal verdicts | Extended structural bound with the normalised weights, the steps actually taken (after the clip) and the clean buffer radius; proof in the text; evaluated on the paired trajectories of A1 as well as A3* | Thm. 1(ii), Sec. VI-E, `scripts/paired_disp.py` |
| A3 mis-stated contraction bound | Corrected to G eps (1 - L^n_W)/(1 - L), with n_W the accepted attacked blocks | Remark 1 |
| A4 near-tautology | "Certificate" removed everywhere; the design bound is attacker-independent by construction | Sec. IV |
| A5 Theorem 2 vacuous | Withdrawn; the last-layer variant is evaluated without a regret claim | Remark 1, Sec. VI-H |
| A6 weak Lipschitz estimates | Finite-difference probes give lower bounds on a Lipschitz constant, so the measured L between 1.08 and 1.39 already shows the map is not a contraction; a stronger estimator could only raise it. The conclusion the bound needs does not depend on the estimator | Remark 1 |
| A7 delta_min hand-set | Derived slack: threshold min{1 - z sigma, relative-slack threshold} with sigma the running standard deviation of the clean pass ratio, floored by its binomial value; candidate in the calibration | Sec. IV (design rule), Table (calibration) |
| A8 G1 mismatch | The uncorrelated-error condition is stated as an assumption; the clean G1 statistic on the tanh channel is reported (mean 1.000, p99 1.050, max 1.080 over 285 blocks, never at tau_1 = 1.25) | Sec. IV |
| A9 notation | Notation table; the CRC slack is rho, the attacked fraction q, the soft-step width omega, kappa is the consolidation clip | Table (notation) |

## B. Threat model and adaptive attacks

| Item | Change | Where |
|---|---|---|
| B1 unconstrained perturbation | Rank-M physical attacker E = G X with M = 1, 2 and CSI error 0.1 at M = 2; full-rank kept as the worst case; SIR mapped to distance under the UMi path-loss exponent | Sec. II-B, Table (attacks) |
| B2 joint pilot + data | Attacker A4: fraction alpha of the energy on the pilots and 1 - alpha on all data sub-blocks, optimised jointly behind the soft gates, alpha = 0.25 and 0.75, with and without the abstain state; jamming controls with the same data power | Table (refdeg) |
| B3 denial of adaptation | Fixed channel-estimate spoof on blocks 40-89 combined with an abrupt channel redraw at block 50; with and without streak recovery; online DeepSIC and LMMSE references | Table (deny) |
| B4 adaptive strength | Unroll 10 and 50, PGD 100 with 5 restarts, BPDA soft-gate output objective with the exact thresholds and running statistics, EOT; combined attack also against online DeepSIC | Table (attacks) |
| B5 optimal slow drift | A3*: 300 attacked blocks of a 400-block run, drift toward a spoofed basin with the soft-gate term, energies 5, 20 and 50%, drift guard at 50%; paired snapshots | Sec. VI-E |
| B6 transfer too narrow | Grey-box attacker with a Citadel-Rx surrogate; query attacker on NACK feedback | Table (attacks) |
| B7 knowledge vague | Knowledge table; synchronisation and pilot knowledge stated | Table (knowledge) |
| B8 budget includes noise | Both budgets defined; the signal-referenced one evaluated | Eq. (budget), Table (attacks) |

## C. Missing baselines

| Item | Change | Where |
|---|---|---|
| C1 CRC + rollback | CRC-Rollback and rollback with all gates; the K = 1 ablation now reverts and falls back (it previously did neither, a bug) | Sec. VI-B, Tables (headline, ablations) |
| C2 regularised online | L2-SP (4 weights), EWC (Fisher from pre-training), EMA teacher | Table (baselines) |
| C3 CRC-verified DD | CRC-verified decision-directed training | Table (baselines) |
| C4 TTA defences | EATA-style and CoTTA-style receivers | Table (baselines) |
| C5 under-tuned baselines | Epochs x learning rate x weight decay grid on calibration seeds, rule fixed in advance; every online-based receiver uses the tuned setting | Sec. V |
| C6 MedBN mislabelled | Renamed "median-of-means gradient" | Sec. V |
| C7 robust LMMSE | Huber-robust LMMSE | Table (baselines) |

## D. Experimental realism

| Item | Change | Where |
|---|---|---|
| D1 block length | N_d in {1000, 5000} x N_p in {32, 100}, plus the default 25000 / 100; the derived slack follows n | Table (scenarios) |
| D2 NR link level + LDPC | Not done: a validated Sionna NR slot with LDPC is several days of work; stated as the main limitation | Sec. VII |
| D3 attacker in the scene | SIR translated to distance analytically; a scene simulation is not done (with D2) | Sec. II-B, Sec. VII |
| D4 N > U | 4 x 16 | Table (scenarios) |
| D5 higher order | 16-QAM at 20 dB | Table (scenarios) |
| D6 abrupt change | Redraw and 6 dB SNR drop at block 50, no attack | Table (deny) |
| D7 long horizon | The 2000-block clean horizon was dropped from the reduced campaign; the 400-block A3* runs are the longest | Sec. VII |
| D8 second backbone | Not done; the claim that the levels and gates "apply unchanged" is deleted | Sec. VII, VIII |
| D9 compute | 30-step and last-layer variants; per-block time on one laptop core; no GPU measurement (the GPU machine was unavailable) | Table (scenarios) |

## E. Statistics and protocol

| Item | Change | Where |
|---|---|---|
| E1 correlated blocks | Moving-block bootstrap with the measured decorrelation length; seed-level exact Wilcoxon tests with the smallest attainable p stated; "every p < 0.001" deleted | Sec. V |
| E2 too few seeds | 10 test seeds at the headline cell, 3 at the other headline cells, 2 elsewhere (stated per table) | Tables |
| E3 easy part of the fade | Attack window start drawn per seed from blocks 10-50; excess SER over the matched clean run is the primary metric | Sec. II-C, V |
| E4 thresholds after test | Calibration seeds 0-2 with rules fixed in advance; all reported results on seeds 10-19, never used for any choice | Sec. V, Table (provenance) |
| E5 sensitivity | One-at-a-time x0.5 and x2 for eight settings (reduced from four multipliers; heat map dropped) | Table (sensitivity) |
| E6 intervals and counts | Wilson intervals with counts | Sec. II-C, tables |
| E7 Fig. 3c | Paired displacement of every level and the anchor, with the design and structural bounds | Sec. VI-E |
| E8 Holm family | Family = the four baselines of a column, stated in the captions | Table (paired) |

## F. Claim consistency

Every sentence F1-F14 was rewritten; the abstract and conclusion carry only numbers computed by
`scripts/revision_numbers.py` from the test-seed records, each tied to one table cell.
F6/F7 ("certificate", "non-expansive") are deleted; F10 is rewritten (the retention bound needs the
reversion; the measured effect of keeping the update is not claimed); F11 is restricted to A1;
F12 uses the suggested wording; F13 is deleted; F14 is replaced by the seed-level tests of E1.
"Halves the SER" is replaced by the measured relative change against the tuned online DeepSIC.

## G. Novelty, positioning, references

| Item | Change |
|---|---|
| G1 | Pilot contamination and its detection, code-verified online training, EATA / CoTTA / SAR / RoTTA / persistent TTA, L2-SP / EWC / mean teacher, adaptive-attack methodology, adversarial PHY learning, complementary learning systems; every reference checked at its source (`docs/REFERENCES_VERIFIED.md`) |
| G2 | The introduction leads with the principle; the CRC-Rollback result (Sec. VI-B) supports the hierarchy |
| G3 | The CRC rule is presented as a design rule |
| G4 | Table I has the rows for pilot-spoofing detection, code-verified online training, CRC-Rollback, filtering/restoring TTA and anchored fine-tuning |

## H. Method upgrades

H1: the consolidation clip (design bound) and the anchor-start variant (calibration candidate);
H2: CRC-verified data labels (the decision-directed baseline); H3: abstain state (with A4);
H4: streak recovery (with denial of adaptation); H5: drift guard (with A3*); H6: per-sub-block CRC
selection (in the receiver, and as an online-DeepSIC comparator that separates its gain from the
memory); H7: per-user gating is implemented but not evaluated in this revision.

## I. Writing and reproducibility

| Item | Change |
|---|---|
| I1 | The repository URL works as given; renaming the repository to Citadel-Rx is an owner action (GitHub redirects the old URL) |
| I2 | Algorithm 1 |
| I3 | Notation table |
| I4 | The period vectors are stated wherever K or C change |
| I5 | Provenance table |
| I6-I8 | SED defined; SER and per-user CRC defined; informal wording removed |
| I9-I12 | Tables carry three budgets; one figure replaces the dense architecture figure; four numbers in the abstract; detailed tables are in `docs/REV_*.md` |
| I13 | One command runs the campaign (`bash scripts/run_laptop.sh`, `make cloud`) and one rebuilds every number (`python scripts/revision_numbers.py`) and the paper (`make paper`); a Zenodo DOI is an owner action |
| I14 | Acknowledgement of the COST2100 trace release; supervision and funding to be added by the author |
