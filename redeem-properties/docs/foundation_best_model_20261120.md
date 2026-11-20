# ReDeeM unified model: current best checkpoint (rtharm, F2-H)

Chosen 2026-11-20 after the F2-I and F2-I2 arms. It is trained from scratch, with no teacher distillation.
The weights are not in Git. They are on Codon:

`/nfs/research/juan/singj/foundation_model/models/redeem_unified_rtharm_f2h_20261116/`
(2.9 GB; `README.md` sha256 891b20f6ac23fa66dbb45c4c95030c65bd53a7202348b1ade8c076aaaf649845,
`SHA256SUMS` sha256 ca68064641cbb73556e65f73712e8151e9597b3b2595f273270922c245238055, 147 files.)

Recommended checkpoint: `seedN/model/epoch_060` (seed0 has the best Val RT).

## How to reproduce

- Source: commit `f563ae37faf3` (trainer). Launcher: `scripts/redeem_f2e_budget_20261107.sbatch` at `684686a`.
- Container: `redeem-foundation_f2f-resume-f563ae37faf3.sif`, sha256 `ac7c37ede7a2673bbd5302857b7da62d38939b70006b220106ef11f7490c18b2`.
- Data: prepared run YAML `run_v0260_root20261111.yaml`, sha256 `ca642a6f7fe53d5a2faa3830b3eeea75c430bfb2521acf20210968d35de276c7`.
- Recipe: `research_curriculum`, 60 epochs, LR 2e-5, warmup 500, cosine to 0.1, grad clip 1.0, `model.rt_objective: Harmonized`.
  All other options are at their defaults (`mobility_batch_size` unset, `specialists.mobility_length_mass` false,
  `specialists.ms2_fragment_tokens` false).
- Launcher environment: `F2E_EPOCHS=60 F2F_ARM_TAG=rtharm F2H_RT_OBJECTIVE=Harmonized`, seeds
  20261006/default, 20261007/1, 20261008/2.

## Results (F2-A Validation scorecard, epoch 60, 3-seed mean)

| Task | v0.52 | rtharm (this model) | ccsx (F2-I) | ccslm (F2-I2) |
|---|---|---|---|---|
| RT MAE | 3.319 | **3.230** | 3.362 | 3.277 |
| CCS MAE | 11.24 | 10.63 | **9.88** | 10.67 |
| MS2 cosine | 0.909 | **0.921** | 0.918 | 0.921 |

Noise floors (MDE, from 3 seeds): RT 0.58, CCS 2.28, MS2 0.124. Every rtharm seed beats v0.52 on all 3 tasks,
but the RT and CCS margins are below the MDE.

## Options added on this branch (all opt-in; the defaults reproduce earlier runs)

| Option | Commit | Result |
|---|---|---|
| `model.rt_objective: Harmonized` | 31528be | Used in this model. RT +0.18 (all pairs), per-source scale error removed. |
| `training.resume_from_epoch` | f563ae3 | Exact data order and LR. AdamW moments reset. |
| `specialists.ms2_fragment_tokens` | 7c964c6 | No rule-level gain. |
| `training.mobility_batch_size` | 585fe9d | Batch 32: CCS +0.75, but RT -0.13 and MS2 -0.003. |
| `specialists.mobility_length_mass` | 585fe9d | No effect alone (CCS -0.04). |
| `init_seed`, `save_epoch_checkpoints`, `term_gradient_probe_every` | ac867c1, b652ee6 | Tooling. |
