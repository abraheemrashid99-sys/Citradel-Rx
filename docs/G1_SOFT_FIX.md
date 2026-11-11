# Soft G1 term of the adaptive attacker: scale fix and re-run

## What was wrong

The soft-gate surrogate replaces the receiver's hard gate verdicts with smooth ones, so that the
attacker's objective can follow the gates. Its G1 term is computed by `Attacker._g1_soft`
(`citadel_rx/attacks/pilot_poisoning.py`): the residual of the linear model with the posterior-mean
symbols of the victim's detector, divided by the LMMSE residual (`physics.model_residual`) and by the
receiver's clean running statistic.

`_g1_soft` averaged the squared residual over the 2N real coordinates of the stacked received vector
`[Re y, Im y]`. The receiver's G1 (`physics.model_residual`) averages |.|^2 over the N complex
antennas. For complex modulation (QPSK in every affected run) the attacker's statistic was therefore
half of the receiver's statistic at the same symbols. Its soft G1 verdict sat near one and the G1 term
hardly shaped the objective. These attacks were weaker on G1 than the paper described.

Found while tracing every equation of the ten-page version to the code (math trace, finding P1).

## Fix

`_g1_soft` now sums the squared residual over the real and imaginary parts and divides by N, the scale
of `physics.model_residual`. For real modulation the value is unchanged.
`tests/test_attack_g1_scale.py` checks that the two residuals agree for one-hot decisions (QPSK and
BPSK).

## Which records are affected

The soft-gate objective is used (`use_soft` in `Attacker.craft`) by every attack with
`attack.soft_gate=true` **and by every A4 and A3\* attack**, whenever the victim has gates. Records of
these attacks against Citadel-Rx, by folder:

| folder | experiment | records | re-run into |
|---|---|---|---|
| `rev_p3_adaptive` | soft gates at SIR 0 and 10 dB, combined attack at SIR 0 dB | 15 | `rev_p3_adaptive_g1fix` |
| `rev_p3_refdeg` | A4, alpha 0.25 and 0.75, with and without the first abstain state | 14 | `rev_p3_refdeg_g1fix` |
| `rev2_abstain_v3` | A4 against the abstain state, seeds 10-19 | 20 | `rev2_abstain_v3_g1fix` |
| `rev2_a4_pgd20` | A4 with 20 PGD steps | 5 | `rev2_a4_pgd20_g1fix` |
| `rev_p8_drift` | A3\*, 5, 20 and 50 % of the pilot norm, 400 blocks | 8 | `rev_p8_drift_g1fix` |
| `rev2_drift_guard` | A3\*, 5 %, with the drift guard | 2 | `rev2_drift_guard_g1fix` |
| `rev2_drift_long` | A3\*, 5 %, 1000 blocks | 2 | `rev2_drift_long_g1fix` |

Not affected: attacks without the soft-gate objective (A1, A2, unrolled and PGD variants, physical,
grey-box, query, denial), and soft-gate attacks on receivers without gates (`p3_online_strongest`:
`_soft_gamma` returns one there). `rev2_abstain` (`h3_a4_a0.25_ratio`, the abandoned first abstain
statistic) is affected but read by no script; it is not re-run.

## Re-run

`scripts/rerun_g1fix.py` finds the affected records itself and re-runs each one with the settings read
from its own record, so the fixed line is the only difference. The old records stay where they are.

Pairing with clean twins:

* `softgate` and `a3star`: the clean twin of every attacked run is run on the same machine
  (`results/rev_clean_<machine>`; the A3\* twins save level snapshots, and `scripts/paired_disp.py`
  writes the paired displacements into the attacked records on that machine).
* `a4`: the attacked run carries `pairs_as`, the machine that ran the record it replaces, and is paired
  with that machine's stored clean twin. This rests on a check: every cloud container gives
  bit-identical runs. The clean twins of seeds 10-14 re-run on `pool12` and `pool13` equal those of
  `cloud` and `pool1`-`pool5` block for block (per-block SER), while the MacBook's differ. Each machine
  that uses `pairs_as` first repeats a stored cloud clean run and writes the comparison to
  `results/g1fix_probe/<machine>.json`; `scripts/revision_numbers.py` honours `pairs_as` only if that
  probe found identical runs.

`scripts/revision_numbers.py` (`with_g1fix`) switches an experiment to its re-runs only when every one
of its seeds has been re-run, so no table cell mixes the two attack versions.

| machine | job | command |
|---|---|---|
| pool13 | soft gates, seeds 10-12 | `rerun_g1fix.py --family softgate --seeds 10 11 12` |
| pool12 | soft gates, seeds 13-14 | `rerun_g1fix.py --family softgate --seeds 13 14` |
| pool14-pool17 | A4, four balanced shards | `rerun_g1fix.py --family a4 --shard k/4` |
| pool18, pool19 | A3\*, seeds 10 and 11 | `rerun_g1fix.py --family a3star --seeds 10` (11) |

(The soft-gate runs of pool12 and pool13 were made with the first version of the script, which wrote
them to `results/rev_p3_g1fix`; the folder was renamed afterwards.)
