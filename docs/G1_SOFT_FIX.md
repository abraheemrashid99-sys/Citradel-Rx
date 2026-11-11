# Soft G1 term of the adaptive attacker: scale fix and re-run

## What was wrong

The soft-gate attacker (B4, `attack.soft_gate=true`) replaces the receiver's hard gate verdicts
with smooth ones, so that its objective can follow the gates. Its G1 term is computed by
`Attacker._g1_soft` (`citadel_rx/attacks/pilot_poisoning.py`): the residual of the linear model
with the posterior-mean symbols of the victim's detector, divided by the LMMSE residual
(`physics.model_residual`) and by the receiver's clean running statistic.

`_g1_soft` averaged the squared residual over the 2N real coordinates of the stacked received
vector `[Re y, Im y]`. The receiver's G1 (`physics.model_residual`) averages |.|^2 over the N
complex antennas. For complex modulation (QPSK, all soft-gate runs) the attacker's statistic was
therefore half of the receiver's statistic at the same symbols. Its soft G1 verdict sat near one
and the G1 term hardly shaped the objective. The attack was weaker on G1 than the paper described
("models G1 and G2").

Found while tracing every equation of the ten-page version to the code (math trace, finding P1).

## Fix

`_g1_soft` now sums the squared residual over the real and imaginary parts and divides by N, the
scale of `physics.model_residual`. For real modulation the value is unchanged.
`tests/test_attack_g1_scale.py` checks that with one-hot decisions the two residuals agree (QPSK
and BPSK).

## Affected rows and re-run

Only runs with `soft_gate=true` against a receiver that has gates are affected:

| row | old records | new records |
|---|---|---|
| `p3_citadel_softgate__sir0` (soft gates, unroll 10, SIR 0 dB) | `results/rev_p3_adaptive` | `results/rev_p3_g1fix` |
| `p3_citadel_softgate__sir10` (soft gates, SIR 10 dB) | `results/rev_p3_adaptive` | `results/rev_p3_g1fix` |
| `p3_citadel_strongest__sir0` (combined attack, SIR 0 dB) | `results/rev_p3_adaptive` | `results/rev_p3_g1fix` |

`p3_online_strongest__sir0` is not affected: online DeepSIC has no gates, so the soft-gate term
is constant there (`_soft_gamma` returns one).

`scripts/rerun_g1fix.py` re-runs the three rows on seeds 10-14 with the settings read from the
records they replace, so the fixed line is the only difference. Each attacked run is paired with a
clean twin from the same machine (seeds 10-12 on machine `pool13`, seeds 13-14 on `pool12`;
twins in `results/rev_clean_pool13` and `results/rev_clean_pool12`). `scripts/revision_numbers.py`
reads the three rows from `results/rev_p3_g1fix`; the old records are kept unchanged in
`results/rev_p3_adaptive` for comparison.

    CITADEL_MACHINE=pool13 python scripts/rerun_g1fix.py --seeds 10 11 12 --workers 4
    CITADEL_MACHINE=pool12 python scripts/rerun_g1fix.py --seeds 13 14 --workers 4
