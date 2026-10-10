# Outside-cone task under a disturbance (cone obstacle + UE-bCBF)

The drone must reach a target cylinder under the tip of an apex-down funnel without entering the
funnel or flying below a minimum altitude, under a bounded translational disturbance. The filter is
the UE-bCBF of the `landing-ue` / `runway-bird-ue` branches with the cone-obstacle sets; Phase II
follows the UE-landing two-stage recipe.

## Task (`docker_batch/configs/cone_obstacle_ue.json`)

| item | value |
|---|---|
| funnel | tip 1.2 m, half-angle 45 deg, tip rounding eps 0.25 m: `h_cone = sqrt(r^2 + eps^2) - eps - tan(theta) (z - tip) >= 0` |
| floor | `z >= ground + 0.5 m` |
| target | cylinder r 0.25 m on the axis, top at the tip; success = first touch |
| base set | 9-D hover LQR ellipsoid at (-2, 0, 1.7), c_B = 12 (c_U 17.88, c_chart 23.18, c_Lyap 15.67, geometric 21.54) |
| actuator box | the UE-landing box (`docker_batch/configs/landing_cone45_r0p3.json`): thrust 0-39.24 m/s^2 (4 g), body rates +-18 rad/s on all axes |
| disturbance | delta_d 0.5 m/s^2, delta_v = 2 pi 0.05 delta_d = 0.157 m/s^3, observer lambda 20, e_bar 0.02, tube scale 1.2 |
| filter | alpha 10 (funnel), alpha_ground 20 (floor), base_alpha 2, horizon 2 s (N = 100), float64 QP |
| reference | L3-1-f: (-3, 0, 2.2) -> level to above the axis -> drop to (0, 0, 0.9), 5.6 s, peak 2.2 m/s |

c_B = 12 (not 2): the end of every backup rollout must lie in the tightened base set
{V <= c_B - m_base(s_N)}; at T = 2 s, m_base = 4.55 with the LQR's contraction (8.36 worst case),
so c_B = 2 would leave an empty tightened set.

## Phase I: LQR alone vs safe arrival on the shrunk / plain sets (CPU, ~2 h each)

```bash
python scripts/train_phase1_cone_obstacle_ue.py --tighten true  --run_name sa_shrunk_seed0 \
    --total_steps 3000000 --hidden_size 128 --start_steps 0 --actor_start_update 20000 --bc_weight 0.5 \
    --contraction_weight 0.01 --eval_every 250000 --seed 0
python scripts/train_phase1_cone_obstacle_ue.py --tighten false --run_name sa_plain_seed0  (same flags)
python scripts/compare_phase1_cone_obstacle.py \
    --run sa_shrunk=outputs/cone_obstacle_phase1/sa_shrunk_seed0 --run sa_plain=outputs/cone_obstacle_phase1/sa_plain_seed0
```

Both policies are trained without a disturbance (d_hat = 0) and used blind to d_hat. The comparison
scores LQR / SA-shrunk / SA-plain with the criterion the Phase-II filter applies (tightened sets,
frozen |d_hat| <= delta_d + e_bar) on the design-region test sets and on a reference tube, and writes
`outputs/cone_obstacle_phase1/compare/{compare.json,chosen_backup.json,compare.png}`. Rule: LQR unless
the best SA policy is >= 2 points better (`--lqr_margin`); the shrunk one only if it beats the plain
one by >= 5 points (`--shrunk_margin`).

## Phase II on the GPU (two stages, evaluation, figures, GIFs)

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_cone_obstacle_ue.sh \
    > ~/ws_shared/PS2-RL/outputs/cone_obstacle_ue.out 2>&1"
docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/cone_obstacle_ue.out"
```

Stages (`STAGES="V P E F"`): V vanilla tracker (no CIL, 3M steps), P PS2 through the UE-bCBF CIL
warm-started from V (1M steps, UE-landing stage-2 settings, backup from `chosen_backup.json`), E
evaluation (128 episodes, same starts and disturbance draws), F figures:
`outputs/cone_obstacle_phase2_ue/cone_ue_progress.png` (tracker without CIL / tracker + UE-CIL /
Phase II), `cone_ue_progress_with_nominal.png` (adds the nominal bCBF), `cone_ue_progress_side.gif`,
`cone_ue_progress_3d.gif`. Stage V does not need Phase I and can start right away (`STAGES=V`).

Disturbance per episode: d(t) = A n sin(2 pi f t + phi), n uniform on the unit sphere,
A ~ U[0, delta_d], f log-uniform in [0.02 Hz, min(1 Hz, delta_v / (2 pi A))], so ||d|| <= delta_d
and ||d_dot|| <= delta_v in every episode (`--env_disturbance_mode fixed`: A = delta_d at 0.05 Hz).
