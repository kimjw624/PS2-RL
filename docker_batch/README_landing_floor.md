# Floor-aware landing: letting the filtered vehicle reach the pad

## Why the vehicle could not reach the floor

- **The safe set had no floor.** The smooth cone continues below the pad down to its apex at −r0/tanθ (−0.3 m for r0 = 0.3, θ = 45°). So touchdown was not a boundary of S, the model could pass through the pad, and 0.2–0.8 % of Phase-I arrivals already dipped below it.
- **The filter's rows were conservative.** The paper's relative-time rows hold still at x only if the backup rollout from x never pulls away from a boundary faster than α·h. The learned backup climbs off low states at up to 30 m/s², so:
  - with α = 4, a filtered LQR aimed at the pad centre hung 6.6 cm up;
  - with floor rows at α_floor = 20 it hangs about 3.4 cm up, which matches the estimate a/(2α²).

## What changed

All changes are opt-in config fields; old checkpoints keep their meaning.

| piece | change | justification |
|---|---|---|
| safe set | `floor_constraint=True`: S = cone ∩ {z ≥ z_pad} (`ps2rl/sets/quadrotor_landing_safe_set.py`) | touchdown becomes the boundary; going below the pad is a crash in Phase I and a row in the filter |
| Phase I | failure = leaving S; optional floor band in the edge region (`--region_edge_floor_prob`) | the backup is trained and certified against the full S |
| certificate | B ⊂ cone (c_cone) and B above the floor (c_ground, z_clear ≥ 0); checked before training | Prop. 1 with the new S |
| gentle recovery (Phase-I option) | `--recovery_rate_cone/floor κ`: the backup must satisfy ∇h·v ≤ κh along its rollouts; certificate adds B ⊂ G (c_recovery_*) | with κ ≤ α, holding still is feasible at every rest state in C_N under the relative-time rows. The filter can then let the vehicle sit arbitrarily close to the floor, and π_b stays a feasible QP point (the paper's guarantee is untouched). G is a training constraint, not a safety spec: the filter's rows and C_N use S only. |
| filter | per-constraint class-K gains `alpha`, `alpha_floor` (α·dt < 1 enforced); `relative_time_floor=off` gives fixed-τ floor rows as an alternative | the floor row sets the flare, ż ≥ −α_floor z. Fixed-τ rows allow holding still at the floor with any backup, but drop the "π_b is always feasible" argument. |
| run-time floor on an old backup | `floor_constraint=on` override | a tightening: C_N is re-evaluated by rollouts against the smaller S; B is checked to lie above the pad |

## Overnight run

```bash
docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_floor_overnight.sh \
    > ~/ws_shared/PS2-RL/outputs/landing_floor_overnight.out 2>&1"
docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_floor_overnight.out
```

Stages (resumable: re-running skips finished work):

| stage | what | output |
|---|---|---|
| A | existing backup with the floor added, 6 filter settings | `outputs/landing_floor/tests/baseline/` |
| B | Phase-I variants × seeds on the GPU: `floor`, `floor_frec10` (floor recovery 10), `floor_rec10` (cone + floor 10), `floor_rec20`. Each is certified before training; an uncertified variant aborts. | `outputs/landing_floor/phase1/<run>/` |
| C | self-check per run: recoverability vs Phase I, cone stress test, floor stress test (the nominal aims 0.3 m below the pad), QP health | `outputs/landing_floor/checks/<run>/` |
| D | reach tests per run × filter settings (below) | `outputs/landing_floor/tests/<run>/` |
| E | the colleague's policy through the new filter, seed-0 run of each variant, safety check | `outputs/landing_floor/phase2_eval/<run>/` |
| F | morning report with a recommended (backup, setting) | `outputs/landing_floor/REPORT.md` |

Reach tests (`scripts/test_landing_floor_reach.py`):
- **T2, hover map.** Where holding still is allowed near the pad, and the lowest allowed height over the pad centre.
- **T1, LQR landing** to the pad centre, the pad edge, and a point below the floor (which must be refused).
  - Starts: the reference start, hover over the pad, and 16 random C_N starts.
  - Touchdown means height ≤ 2 cm over the pad disk at speed ≤ 0.3 m/s.
- **T3, tracking** `quadrotor_landing_cornercut_reference.npz`. That reference is the colleague's generator with `--offset 0 --vx0 0 --vz0 -1`: it ends at the pad centre but leaves the cone between t = 0.32 s and about 1.1 s (min h −0.15 m). The tracker alone leaves the cone; filtered, it must stay inside and still touch down.

Expected duration on the RTX 4050 is about 3.5 h with `SEEDS="0 1"` and about 5 h with `SEEDS="0 1 2"`. This is rough: the CPU smoke run here was fine, but I have not timed the GPU.

Useful knobs:

| variable | example |
|---|---|
| `SEEDS` | `"0 1 2"` |
| `TOTAL_STEPS` | `8000000` |
| `MAX_PARALLEL` | `2` |
| `VARIANTS` | `"floor:--floor_constraint true|floor_frec20:--floor_constraint true --recovery_rate_floor 20"` |
| `SETTINGS` | `"a10_f20:alpha=10,alpha_floor=20;a25:alpha=25,alpha_floor=25"` |
| `TEST_ARGS` | `"--n_random 32 --zeta_touch 0.03"` |
| `STAGES` | `"D F"` (tests and report only) |

## Reading the report

A (backup, setting) pair counts as **safe** only if all of the following hold:
- in every test, min cone margin ≥ −1 mm and min height ≥ −1 mm;
- slack ≤ 1e-2 and no QP fallbacks;
- its self-check passes;
- the floor is in S.

Among safe pairs, the report ranks by T3 touchdown, then T1 touchdown, then the lowest allowed hover height, then Phase-I μ.

## After the report

Train Phase II through the CIL with the recommended backup and setting (`LANDING_HANDOFF.md`, `landing_projection_setup(ckpt, alpha=..., alpha_floor=...)`). Use a reference that touches down on the pad; the cone-cutting one is a good test.
