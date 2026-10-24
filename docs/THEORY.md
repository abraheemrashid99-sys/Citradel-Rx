# Theory targets and proof routes (working document)

Everything here is a *target with a proof route*, to be upgraded to a proof
during weeks 7–8 of the plan. The notation follows the proposal (Appendix A).
Where the code measures a constant, the measuring function is named.

## Setting

* Blocks b = 1..T. Per-block detection loss ℓ_b(θ), convex and L-smooth in
  the adaptable parameters A (true for the last-layer-adaptable variant with a
  log-linear output; for full-MLP adaptation the statements hold for the
  linearised receiver around the pre-trained point, which is what we will
  state and what the LMMSE-regularised variant guarantees). Gradients bounded
  by G_θ; domain diameter D.
* θ*_b = per-block minimiser; path length P_T = Σ_b ‖θ*_{b+1} − θ*_b‖. For
  AR(1)/Jakes block correlation ρ_D the optimum moves O(√(1−ρ_D)) per block
  (first-order perturbation of the minimiser of a strongly convex loss whose
  data distribution is parametrised by H_b), so P_T = O(T √(1−ρ_D)).
* Adversary: a fraction ρ_blk of blocks carry ‖E_b‖_F ≤ ε. The gradient of
  ℓ_b in the *input* is G-Lipschitz (measured by
  `theory.bound.estimate_input_lipschitz` as a lower bound on the true G), so
  a corrupted block changes the observed gradient by at most ε G.
* Gate: rejects a corrupted block with probability ≥ p_k^rej when it would
  transfer into level k; accepts a clean block with probability ≥ 1 − α_k.
* Timescales: C_1 = 1 < C_2 < … < C_K, η^(k) C_k ≤ c < 1.

## Theorem 2.1 (corruption-robust dynamic regret) — route

Regret_T = Σ_b ℓ_b(θ_b) − ℓ_b(θ*_b)
         ≤ O(√(T(1+P_T)))                                  [clean dynamic regret]
         + Σ_k O(ρ_blk ε G (1 − p_k^rej) T / C_k)            [adversary, per level]
         + Σ_{k≥2} O(α_k D T / C_k).                         [false rejections]

1. **Clean term.** The fast level is online gradient descent with step
   η_b = Θ(√((1+P_T)/T)) against a moving comparator; Zinkevich (2003) /
   Mokhtari et al. (2016) give the √(T(1+P_T)) rate. The surprise-driven rate
   η_b = η_max · 2r_b/(1+r_b) is bounded within [0, 2η_max], so it changes
   constants only (this is why v1 uses a deterministic, bounded rate).
2. **Adversary term, level 1.** A corrupted block perturbs the gradient by at
   most εG; standard corrupted-OCO bookkeeping (Wang & Chaudhuri 2018; the
   additive-C structure of corruption-robust OCO) adds η εG per corrupted
   block, i.e. ρ_blk εG T / C_1 with C_1 = 1 — the fully exposed regime of Wu
   et al.
3. **Adversary term, level k ≥ 2.** Level k is only touched at ⌊T/C_k⌋
   consolidation instants. At each, the consolidated increment is the gated
   average of level-(k−1) increments; a corrupted contribution survives only
   if the gate failed, probability ≤ 1 − p_k^rej, and its magnitude is
   bounded by η^(k) ρ_blk ε G (the gate average divides by the number of
   accepted increments, so one corrupted block among C_k contributes at most
   its own share). Summing: O(ρ_blk ε G (1 − p_k^rej) T / C_k).
4. **Coupling between levels.** Re-anchoring pulls the fast iterate toward the
   slow projection with weight β every C_2 blocks. Treat the fast level as the
   inner (quasi-equilibrated) variable and the slow levels as the outer
   variable in a two-timescale decomposition (as in the Nested-MARL
   convergence proof): over one slow period the fast iterate's average
   regret against its own moving comparator is bounded by step 1 plus the
   displacement of the anchor, which is bounded by step 3. The condition
   η^(k) C_k ≤ c < 1 makes each level a contraction over its own period so
   the decomposition is well defined.
5. **False rejections.** A rejected clean block removes its (bounded) honest
   increment from the consolidation; the lost progress per rejection is at
   most η^(k) G_θ ≤ D, probability α_k, at T/C_k instants.

Open point for the proof: the gate verdict γ_b is a soft weight in [0,1],
not a Bernoulli; the bookkeeping goes through with E[γ_b] in place of
1 − p^rej (the certificate below then holds in expectation, or with high
probability via Hoeffding over the C_k blocks in a period).

## Corollary 2.2 (retention certificate) — route

Within any window of W blocks, level K receives ⌊W/C_K⌋ consolidations, each
moving it by at most η^(K) × (gated mean of increments), and the adversary's
share of the gated mean is ≤ ρ_blk ε G (1 − p_K^rej) by step 3. Hence

   ‖θ^(K)_adv − θ^(K)_clean‖ ≤ η^(K) ρ_blk ε G (1 − p_K^rej) W / C_K.

The code evaluates the right-hand side with a *measured* G and compares it
with the paired adv/clean level-K snapshots (`run.save_params`) in E5. The
empirical slope in 1/C_K is the testable prediction.

## Proposition 2.3 (CRC gate) — route

Sub-block of L symbols with an r-bit CRC; post-adaptation symbol error
probability p_1 under A1 (p_0 clean). A sub-block passes iff all L symbols
are correct, or at least one is wrong and the CRC collides (probability
2^{−r} for a uniformly random syndrome). So

   P(pass) = (1−p_1)^L + 2^{−r}(1 − (1−p_1)^L)  ≤  (1−p_1)^L + 2^{−r}.

With n_sub sub-blocks per block the pass *fraction* concentrates around this
value (Hoeffding), and the gate G2 rejects when the fraction falls below τ_2
times the clean EMA ≈ (1−p_0)^L. An availability attacker that succeeds in
raising p_1 is therefore rejected with probability → 1 exponentially in n_sub;
one that leaves p_1 ≈ p_0 causes no availability harm by definition. This is
checked numerically by `tests/test_basics.py::test_crc_pass_rate_theory_matches_simulation`.

For A3 (boiling frog) the attacker keeps p_1 ≈ p_0 by construction and the
CRC gate is uninformative; the defence is the 1/C_K factor (Corollary 2.2)
plus G3 (innovation test against the channel prior).

## What must be measured, not assumed

| Constant | Where measured |
|---|---|
| G (input-Lipschitz of the gradient) | `theory.bound.estimate_input_lipschitz` |
| p_k^rej, α_k | gate rejection / false-rejection rates in run records (`gate_reject_rate_attacked`, `gate_false_reject_rate_clean`) |
| ρ_D, P_T proxy | `h_norm` trajectory in run records; Jakes ρ from `channel.rho` |
| η^(K), C_K | config |
