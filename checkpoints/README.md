# PS2-RL checkpoint bundle

This directory holds the pretrained artifacts needed to reproduce the paper's
Phase-2 runs and the reset-library / warmstart pipelines, plus representative
**deployed Phase-1 SA policies and Phase-2 PS2 policies** you can evaluate/deploy without a 5–14 h retrain.
git-LFS / external hosting is unnecessary), so the commands below resolve out of the box
after cloning.

## Contents

Three groups: `deployed_sa/` (Phase-1 safe-arrival policies), `deployed_ps2/`
(Phase-2 PS2 policies), and `quadrotor_vanilla/` (the vanilla-tracker warm-start +
its derived reset library + source traces).

| Path | What it is | Consumed by |
|---|---|---|
| `deployed_sa/unicycle_sa/best_weights.pkl` | Unicycle Phase-1 safe-arrival backup actor | `train_phase2 --system unicycle` (learned backup), `evaluate_phase1` |
| `deployed_sa/quadrotor_sa/best_weights.pkl` | Quadrotor Phase-1 safe-arrival backup actor | `train_phase2 --system quadrotor` (learned backup), `evaluate_phase1` |
| `deployed_sa/quadrotor_sa/configs.json` | Run config for the quad SA actor (`actor` net arch + `backup_env`/`reset_library`; CBF params at `backup_env.cbf_cfg`) | loaded alongside `deployed_sa/quadrotor_sa/best_weights.pkl` |
| `quadrotor_vanilla/quadrotor_vanilla_weights.pkl` | Vanilla-tracker SAC weights used to warm-start quad Phase-2 | `train_phase2 --system quadrotor` |
| `quadrotor_vanilla/reset_library.pkl` | Quad Phase-1 reset library: the pools/splits/nominal/held-out reset **states** plus the minimal base-set/safe-set config needed to sample them | `train_phase1 --system quadrotor` (reference; rebuilt from the traces) |
| `quadrotor_vanilla/vanilla_traces/trace_seed0_runIdx{NN}.npz` | 20 vanilla-eval episode traces (`NN` ∈ 00–19) | reset-library construction |
| `deployed_ps2/<family>/{best_weights.pkl, configs.json}` | Deployed Phase-2 PS2 policies (4 families) — a trained safe controller you can evaluate/deploy directly | `evaluate_phase2 --experiment checkpoints/deployed_ps2/<family>` |

The *Consumed by* column names the release entrypoints
(`train_phase1` / `train_phase2` / `evaluate_phase1` / `evaluate_phase2` with `--system …`).

The **powerloop reference trajectory** is not stored here — it ships in the package at
`ps2rl/envs/assets/quadrotor_powerloop_reference.npz` (byte-identical to the research
source).

### Reset-library trace selection

The Phase-1 quadrotor reset library is built from the **20** shipped traces
(`seed_0`, `runIdx 00–19`), in numeric `runIdx` order. The `trace_seed0_runIdx{NN}.npz`
naming makes this selection explicit and reproducible; `train_phase1 --system quadrotor`
rebuilds the library from them and reproduces the shipped `reset_library.pkl`.

### Deployed Phase-2 PS2 policies

`deployed_ps2/` ships one representative **deployed** PS2 policy per config family, so
you can evaluate/deploy the safe controller without retraining Phase 2. Each family is a
minimal run directory (`configs.json` + the checkpoint), so the release evaluator loads
it directly:

| Family (`deployed_ps2/…`) | System | Backup policy | Checkpoint file |
|---|---|---|---|
| `unicycle_ps2_analytic`  | unicycle  | analytic (LQR)        | `best_weights.pkl` |
| `unicycle_ps2_learned`   | unicycle  | learned (Phase-1 SA)  | `best_weights.pkl` |
| `quadrotor_ps2_analytic` | quadrotor | analytic (cascaded PID + LQR) | `best_weights.pkl` |
| `quadrotor_ps2_learned`  | quadrotor | learned (Phase-1 SA)  | `best_weights.pkl` |

Every family's checkpoint is `best_weights.pkl` — the checkpoint the release
Phase-2 evaluator selects (`--weight_preference best_only`). For the quadrotor this is
the return-selected best (the source run saved it as `best_weights_return.pkl`; it is
renamed to `best_weights.pkl` on staging so both systems are uniform). The `configs.json`
are the runs' training configs **converted to the release schema** (value-preserving:
the base controller K and base set are unchanged; only the on-disk schema differs), with
the learned arms' backup path and the warm-start path rewired to the staged checkpoints.
Run the evaluator from the repo root so those relative paths resolve, e.g.:

```bash
python scripts/evaluate_phase2.py --system quadrotor \
  --outputs_dir checkpoints/deployed_ps2 \
  --experiment  checkpoints/deployed_ps2/quadrotor_ps2_learned \
  --weight_preference best_only

python scripts/evaluate_phase2.py --system unicycle \
  --outputs_dir checkpoints/deployed_ps2 \
  --experiment  checkpoints/deployed_ps2/unicycle_ps2_learned \
  --weight_preference best_only
```

