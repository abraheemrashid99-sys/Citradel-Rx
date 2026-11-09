# Level snapshots (`*.levels.npz`)

Runs with `run.save_params=true` (paired displacement: P4, A3*/P8, H5, B5 and their clean twins) also write
the parameter trajectory of every level, one float16 snapshot per block. These files are large (37-375 MB
each; GitHub accepts at most 100 MB per file) and are not in the repository (`*.npz` is in `.gitignore`).
Nothing in the paper needs them: `scripts/paired_disp.py` stores everything it derives from them (the paired
displacement of every level, of the anchor and the movement of each run, fields `paired_disp`,
`paired_anchor`, `paired_move`, `paired_with`) in the attacked run's JSON record, and the tables and
figures are computed from those fields. A snapshot is deterministic given the record's configuration and
seed: re-running that configuration (`scripts/run_experiment.py ... --set run.save_params=true`) on the
same kind of machine writes it again.

Snapshot files that existed at the end of the revision on the cloud machine (sha256 for an archive, e.g.
Zenodo); those of the pool machines (P4 seeds 13-14, B5) were analysed there and their displacements are in
the records:

| file | MB | sha256 |
|---|---|---|
| `results/rev2_drift_guard/h5_a3star_e0.05_guard__long_s10_228773785711.levels.npz` | 150 | `5bc3135f2049b8235b7bc9a07e1e35ac14e0b2b84b2b397b4b7b520514158750` |
| `results/rev2_drift_guard/h5_a3star_e0.05_guard__long_s11_9e49237c4cf0.levels.npz` | 150 | `c9c7c1fe753fdc87fccca54825148af578a11901ef155029c926d5e5c5a57577` |
| `results/rev_clean_cloud/clean_265be639e5_s10_e8d895ab47e0.levels.npz` | 150 | `37b45e5584b14ee264b4bc300e11a4aa67692f6b410b053cc0e638bbd1c9485b` |
| `results/rev_clean_cloud/clean_265be639e5_s11_832eeef7a101.levels.npz` | 150 | `e9ccc5e023121c0d6b4513ee6f50e2478bdca10cd333cec7863038de0128ae65` |
| `results/rev_clean_cloud/clean_3b3e8e4941_s10_964cda537d56.levels.npz` | 37 | `efedf0007674410eaebe2617826b3fc7bc229fe3391826c4affb08dae59a900e` |
| `results/rev_clean_cloud/clean_3b3e8e4941_s11_d9a0ef005fd2.levels.npz` | 37 | `fe1e407329168965fc5d5162ec908fb13bc474e5aced7500104c1964eea236f8` |
| `results/rev_clean_cloud/clean_3b3e8e4941_s12_98d1f8348318.levels.npz` | 37 | `c0a01dd50394e7e6f7c695fe7e9c751c4128b2a8d0773296afa91b647a1f7fca` |
| `results/rev_clean_cloud/clean_a14ebcbf4c_s11_0af73751a678.levels.npz` | 375 | `b7a4ff6a8ad4e7e0c0db2051a61a7188f19531973240c886fafc6b56b32451c5` |
| `results/rev_p4_paired/p4_citadel__sir0_s10_73b2823d82c9.levels.npz` | 37 | `8341917c83db801e532eb548ccb27aed6e59a63a2671b1ea90c95a5ffda1fb29` |
| `results/rev_p4_paired/p4_citadel__sir0_s11_19e14c40eb21.levels.npz` | 37 | `8dd5f28417e617d5615d4c7ce75c4877c21dffb923acf4d4ba4b268d8231dc7c` |
| `results/rev_p4_paired/p4_citadel__sir0_s12_4f46f97a6a83.levels.npz` | 37 | `8a963608c428751de7ef0ef734ae48e73a8235b3bf251a7611725d7922fcaa25` |
| `results/rev_p4_paired/p4_citadel__sir10_s10_5f0413fbf2f4.levels.npz` | 37 | `5ea8eaa0186033ad203e80e1dea3489752f01c386d8210afa06ea40b73e08323` |
| `results/rev_p4_paired/p4_citadel__sir10_s11_794929b771cd.levels.npz` | 37 | `9007f9125c5ea3c06528bc0c75e3e782a2fd90a13eccd5d3c753cf595dabe432` |
| `results/rev_p4_paired/p4_citadel__sir10_s12_bab27f7112d4.levels.npz` | 37 | `f233e27b25c4e3b326fdd74f85b0cc4d716e8bd33692280c66a7980ad113845f` |
