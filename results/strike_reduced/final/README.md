# Strike task (reduced arena) — the report's run

The 3-seed run behind the conference report (2026 공군사관학교 미래 항공우주 학술대회). It was made with
`docker_batch/run_strike_final.sh` and the task files in `docker_batch/configs/strike_final_{p1,residual,ps2}.json` as
they are in this branch.

- Arena: 40 × 20 m, 3 tall obstacles, 1 free chaser (radius 0.4 m, 2 m/s, 2 m/s²), a drifting target, 15 s episodes.
- Test set: 1000 scenes (scene seed 9001), the same for every configuration.
- Evaluated:
  - seed 0: every run;
  - seed 1: pursuit, residual and PS2 with the final filter, and PS2 with one escape;
  - seed 2: models only.
- `run/` has the same layout as `outputs/strike_final`, so it works as `ROOT` for every script.

## Headline (15 s, the report's Table 1)

| configuration (report name) | episodes | strike % | contact % (count) |
|---|---|---|---|
| 잔차 정책 (residual striker, no filter) | 2 × 1000 | 74.0 ± 0.9 | 26.05 (521) |
| 잔차 + 최종 필터 (residual striker, final filter) | 2 × 1000 | 94.4 ± 0.6 | 0.00 (0) |
| PS2 + 단일 회피 필터 (PS2 striker, one escape) | 1 × 1000 | 95.7 | 1.70 (17) |
| PS2 + 최종 필터 (PS2 striker, final filter) | 2 × 1000 | 97.6 ± 0.8 | 0.05 (1) |

- **Paired tests** (same scenes, exact McNemar):
  - final filter vs one escape (PS2, seed 0): strikes 30 vs 3 discordant, p = 1.4·10⁻⁶; contacts 0 vs 16, p = 3.1·10⁻⁵;
  - PS2 vs residual (final filter, 2 seeds): 89 vs 25, p < 10⁻⁶.
- **Phase I** (3 seeds, safe arrival within 1.5 s): learned 70.1 ± 0.3 % weighted, hover LQR 43.8 %.
- **The one contact** of PS2 + final filter is s0 scene 845. There, all 35 candidates fail because every backup plan crosses the wall. Phase I was trained without walls.

All numbers: `paper/numbers.md` (Clopper–Pearson intervals, counts, paired tests).

## Contents

| path | content |
|---|---|
| `run/p1/td3_a0p2_seed{0,1,2}/` | Phase-I backups: `strike_backup_policy_actor.pkl` (what the filter loads), certificate, configs, training history, summary |
| `run/p1/compare/`, `run/p1/logs/` | Phase I vs hover LQR (tables, figures), training logs |
| `run/residual/striker_rs0p1_s{0,1,2}/` | residual strikers (no filter): `best_weights.pkl`, configs, history, summary |
| `run/ps2/striker_rs0p1_s{0,1,2}/` | PS2 strikers (trained through the final filter): `best_weights.pkl`, configs, history, summary |
| `run/eval/s{0,1}/<run>/` | evaluations: `strike_filter_table.{md,json}`, per-episode records `strike_filter_episodes.json` (outcome code = index in `outcomes`), logs |
| `run/report/` | `scripts/report_strike.py` output: results, ablation, 10 s horizon and Phase-I tables (md + tex), figures |
| `paper/tables/` | the report's tables 1 (10 s / 15 s), 2 (filters), 3 (ablations), 4 (Phase I), 5 (paired tests); md / csv / tsv (HWP) / tex |
| `paper/figures/` | outcomes, strike time, chaser clearance, filter activity, training curves, ablation, Phase I (png + pdf) |
| `paper/numbers.md`, `paper/episode_picks.json` | numbers quoted in the text; representative scenes |
| `paper/traj4/` | trajectory figures of the four compared configurations in the report's terms (`grid_main`: 827, 576, 845; `grid_all`: seven scenes; per-claim pairs; `snapshots/` of the filter's check in scene 845); `replay_vs_gpu_eval.txt` |

Left out to keep the branch small: `final_weights.pkl`, the Phase-I `best_weights.pkl` (full training state; the actor
is kept), the scene-pool caches (rebuilt on first use) and the failure GIFs.

## Regenerate

```bash
# tables, figures and numbers from these records (seconds, CPU)
ROOT=results/strike_reduced/final/run OUT=results/strike_reduced/final/paper STEPS=tables bash docker_batch/run_strike_paper.sh
# report tables
python scripts/report_strike.py --root results/strike_reduced/final/run --seeds 0 1 2
# trajectory figures (replays of the chosen scenes; GPU, EXACT=1 = exactly the evaluated episodes)
ROOT=results/strike_reduced/final/run STEPS="episodes traj4" METHODS="residual ps2_one ps2" RES_MODES="none learned" \
  EPISODES="827 69 601 271 576 972 845" TRAJ_GRID="827 576 845" TRAJ_SINGLE="827 576 845" \
  TRAJ_PAIRS="filter:res,res_f:827,69:clear,hc backup:one,ours:827,601:hc,variant backup:one,ours:576:dist,variant phase2:res_f,ours:271,972:dist,eta" \
  TRAJ_SNAP=845 bash docker_batch/run_strike_paper.sh
```

The figures in `paper/traj4/` were replayed on a CPU with `EXACT=0`. In all seven scenes, all four configurations end
as in the GPU evaluation, within 0.2 s (`paper/traj4/replay_vs_gpu_eval.txt`).
