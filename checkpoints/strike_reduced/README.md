# Strike task (reduced arena) — reference models

These models produced the decision experiments in `results/strike_reduced/`. Seed 0 of every stage. The arena and
method are described in `STRIKE_HANDOFF.md`.

| folder | what | trained with | key numbers |
|---|---|---|---|
| `phase1_td3_a0p2_seed0/` | Phase-I safe-arrival backup: hover LQR + learned residual (TD3, anchor 0.2), static chaser balls, T = 1.5 s, 5M steps | `docker_batch/configs/strike.json` | in C_N on test scenes: general 0.840, blocked 0.365, chaser 0.452, shell 1.000 (hover LQR: 0.556 / 0.001 / 0.079 / 1.000) |
| `residual_striker_s0/` | residual striker (pursuit 6 m/s + du, guidance features, 135 inputs), trained **without** the filter, 10M steps | `docker_batch/configs/strike_residual_guid.json` | warm start of the PS2 striker; baseline "no PS2" |
| `ps2_striker_s0/` | PS2 striker: the residual striker trained 3M steps **through** the filter (one escape) | `docker_batch/configs/strike_residual_ps2.json` | with the deployed filter: 128/128 strikes, 0 contacts (15 s) |

Files:
- `configs.json`: the run's task configuration (the evaluation reads its `strike` / `env` / `actor` blocks);
- `summary.json`: training summary;
- `strike_backup_policy_actor.pkl`: Phase-I actor;
- `best_weights.pkl`: striker actor, critics and temperature, at the best training evaluation.

Evaluate (from the repository root; GPU, or `JAX_PLATFORMS=cpu`):

```bash
python scripts/eval_strike_filter.py --backup_run checkpoints/strike_reduced/phase1_td3_a0p2_seed0 \
  --task docker_batch/configs/strike_final_ps2.json --chaser_model guidance \
  --guidance '{"committed_prediction": true, "escape_variants": [[2.0, 60], [2.0, -60], [3.5, 0], [3.5, 60], [3.5, -60]]}' \
  --nominal striker --striker_run checkpoints/strike_reduced/ps2_striker_s0 --modes "none learned" \
  --episodes 128 --scene_seed 9001 --n_show 0 --gif 2 --gif_pick close_chaser --out outputs/strike_eval_reference
```

`--nominal striker --striker_run checkpoints/strike_reduced/residual_striker_s0` gives the "no PS2" baseline, and
`--filter_type bcbf` gives the minimal-correction QP with the same safeguard.
