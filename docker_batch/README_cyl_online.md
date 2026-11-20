# Obstacle-only online filter (simplified strike, Tarot 450, Raspberry Pi 4B)

The strike method (Phase-I safe-arrival backup, hover base set, discrete backup filter, go-to-goal nominal with a
heading search) for the case we will fly first: **reach a goal straight ahead without touching tall cylinders that
are only sensed on the way**, on a Tarot 450-class vehicle, with the filter running **on the companion computer**.

What changed from the strike task (`README_strike.md`, `README_strike_improve.md`):

| | strike (reduced arena) | this task |
|---|---|---|
| obstacles | 3 random rounded boxes / cylinders + a chaser | 2..4 random tall cylinders (r 0.15..0.6 m), no chaser; deployment demo: one virtual cylinder (r 0.1 m) 1.3 m ahead, or a two-cylinder slalom, the first 1.1 m ahead (layouts O3 / S1, section 5) |
| obstacle knowledge | all known | sensed within a range (360 deg), the known set only grows |
| action box | thrust 0..4 g, body rates 18 rad/s | thrust 0.5..1.5 g, roll/pitch rate 2 rad/s, yaw rate 0.5 rad/s (the slow-track box) |
| drone radius / floor | 0.3 m / 1 m | 0.35 m (450 mm frame + 10 in props) / 0.5 m |
| speed | 4..6 m/s | about 1 m/s |
| hover LQR / base set | c_B 50, ellipsoid 0.61 m (radius) | re-tuned for the box (below): c_B 20, ellipsoid 0.40 m |
| backup network | 256 x 256 | 32, 64 or 128 wide (trained; 256 timed only), 128 recommended |
| filter | 7 backups x 5 blends = 35 rollouts of 3 s | 1 backup x 2 blends + continuation = 3 rollouts of 1.0 s |
| plant in evaluation | the filter's model | delay, rate / thrust lag, thrust / rate gain errors, estimate noise (no wind) |
| disturbance | none | none modelled; an observer absorbs the thrust-map error (section 3) |

Everything new is numpy (`ps2rl/online/`), optionally compiled with numba (`ps2rl/online/fast.py`, identical
decisions). No JAX on the vehicle.

---

## 1. Vehicle box, hover LQR and base set

**Box.** Collective thrust 4.905..14.715 m/s^2 (0.5..1.5 g), roll/pitch rate 2 rad/s, yaw rate 0.5 rad/s: the box the
slow-track policies fly in SITL (`SLOW_TRACK_ONBOARD.md`). `StrikeConfig.omega_z_max` (new, 0 = off) gives yaw its own
limit.

**Hover LQR (design B).** The strike LQR's input weights rescaled by the input margins
(r_i <- r_i (m_ref,i / m_i)^2, the landing-x500 rule) were too gentle (1 m of braking per 1 m/s, 4 deg of tilt). Position
weights x4 and velocity weights x10 of the strike design, yaw weights 5.8 / 5.8:

    lqr_q = (8, 8, 32, 0.1, 0.1, 5, 8, 8, 5.8)      lqr_r = (0.24, 12.15, 12.15, 5.8)

**Base set c_B = 20** (`certify_base_set`): input feasibility c_U 49.6, attitude chart 1172, one-step decrease without
wind holds (V+/V 0.990 at c_B). Hover ellipsoid 0.40 m in radius horizontally, 0.17 m vertically; the base set holds
level flight up to 0.58 m/s, any velocity up to 0.92 m/s with tilt, 9 deg of tilt, 15 deg of yaw. Braking from 1 m/s:
0.77 m, in B after 0.32 s.

**Wind is not considered for now** (decision of 2026-11-19): the evaluation plant has no wind. For the record, the
adversarial one-step check with 0.3 m/s^2 does not pass at c_B (V+/V 1.012; with 2 rad/s the hover LQR's decrease per
step is small), and in a steady push the *re-anchored* LQR drifts (no position feedback). The disturbance observer of
section 3 handles both if wind comes back (`PlantConfig.wind`, `gust`).

Other designs tried (all certified): the plain rescale (ellipsoid 0.72 m, 1 m braking per m/s), Q position x4 /
attitude x4 (0.38 m, c_U 61). To redo the choice, vary `lqr_q` / `lqr_r` in the task file: the trainer prints the
certificate (`certify_base_set`) before training.

---

## 2. Phase I (safe-arrival backup)

Task file `docker_batch/configs/cyl_online_p1.json`: the strike Phase I with `n_chaser_slots 0` (new: no chaser slots),
cylinders only, 4 obstacle slots (observation 48), T 1.5 s, start speeds up to 2 m/s, regions general / blocked / shell
(the chaser and escape regions have weight 0), TD3, actor initialized as the hover LQR, 4 M steps, ReLU.

    # GPU (container), every width one after the other, then exported for the filter:
    docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_cyl_online_phase1.sh > ~/ws_shared/PS2-RL/outputs/cyl_online_phase1.out 2>&1"
    # CPU is fine too (about 10 min at width 64 on 2 cores):
    PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu WIDTHS="32" bash docker_batch/run_cyl_online_phase1.sh

Results (held-out test, recoverable within N = 75 steps; learned vs hover LQR):

| width | general | blocked | shell | weighted | CPU time |
|---|---|---|---|---|---|
| hover LQR | 0.894 | 0.033 | 1.000 | 0.571 | - |
| 32 | 0.968 | 0.318 | 1.000 | 0.714 | 8 min |
| 64 | 0.973 | 0.369 | 1.000 | 0.737 | 12 min |
| 128 | 0.972 | 0.352 | 1.000 | 0.730 | 29 min |
| 256 | not trained here (stopped: 4 times slower on the CPU; 32..128 already equal) | | | | |

Width beyond 32 buys nothing. To train 256 as well (GPU): `WIDTHS="256" bash docker_batch/run_cyl_online_phase1.sh`.
The runs (configs, summary, certificate, history, actor) are in `results/cyl_online/phase1/`.

"blocked" puts a cylinder where the hover LQR stops: with 2 rad/s and 1.5 g many of those starts cannot be saved at all.

**Export** (`scripts/export_cyl_backup.py`): `checkpoints/cyl_online/<run>/backup.json` (hover LQR, base-set extents,
box, actor layout, the run's test results and certificate) + `backup_weights.npz` (no pickle; the log-std head is
dropped). It checks the numpy runtime against the JAX code on random states: observation, plant step, margins and
backup action agree to 2e-6 (base margin 1e-4, float32 level). `checkpoints/cyl_online/lqr/` is the hover LQR alone.

---

## 3. The online filter (`ps2rl/online/filter.py`)

Per decision, for each vehicle:

* **candidates** u(lam, k) = lam u_nom + (1 - lam) pi_k(x), lam in the blend set (1 first, 0 last), k in the backup
  family ("learned": the actor, hover LQR inside the base set; "lqr");
* **certificate**: the candidate for `period` control steps (state feedback every step; the nominal is simulated with
  the decision's goal and known cylinders), every state safe, then the backup reaches the base set within
  N = horizon / dt steps without leaving the safe set (first hit, the Phase-I criterion). Period 1 is the strike check;
* **decision**: the largest certified lam; else the continuation (the held hover at the latched anchor, or the
  committed plan with its frozen scene view); else no certificate (continuation applied, reported);
* **clearances**: `margin` 0.1 m added to the drone radius everywhere, `fresh_margin` 0.15 m more for new plans (the
  strike filter's fresh-plan slack). Without it the certificate flickers off at the boundary with estimate noise: in
  the stress set (100 episodes, 1.0 s) 81 % of decisions without a certificate with the hover LQR and 37 % with the
  learned 128 backup, against 0.4 / 0.5 % with it (no contact either way in these runs, but the guarantee is gone).
  Every backup hands over to the hover LQR in the stricter base set, so a continued plan is the certified plan;
* **thrust-map error**: no wind is modelled, but a thrust map that is a few percent off is a steady vertical push the
  filter's model does not have. Without correction the re-anchored LQR then sinks at constant speed: with the
  realistic plant's +-8 % thrust gain, 23 % of the stress episodes hit the ground (and 10 % of the straight-line
  two-cylinder ones). A disturbance observer (d_hat = L (v - xi), L 2 1/s, |d_hat| <= 1 m/s^2; the slow-track one)
  removes that: its estimate enters the rollouts (x+ = f(x, u) + dt d_hat), the backups and the base set (the hover
  equilibrium that cancels it: phi_eq = (d_y, -d_x, 0) / g, thrust g - d_z) and the nominal (feed-forward). With
  d_hat = 0 this is the Phase-I model; `run_episodes(observer_gain=0)` turns it off.

**Compute** goes with rollouts x steps: rows R = |blends| x |family| + 1, steps S = period + N; a rollout stops once it
has hit the base set or failed. Two backends with identical decisions (`scripts/check_cyl_online_fast.py`):

* **numpy**: the rollouts batched, Python loops over steps: about 0.2 ms per step on a 2.1 GHz Xeon core whatever the
  number of rows (call overhead), so the horizon sets the cost;
* **numba** (`pip install numba`): every rollout compiled, about 2.5 us per step plus 0.5 us per row-step plus the actor;
  about 50 times faster. The first decision compiles (4 s on the sandbox, probably 15..30 s on the Pi; cached in
  `ps2rl/online/__pycache__`): make one decision at start-up, before the hand-over.

Timing fit on the sandbox core (`results/cyl_online/bench_sandbox.json`; S steps, R rows, F actor kFLOP):

    numpy  ms = S (0.20 + 0.004 R + 0.000047 R F) + 0.51 (period - 1)
    numba  ms = S (0.0025 + 0.00048 R + 0.000029 R F) + 0.75 (period - 1)

(width 32: F = 5.4 kFLOP; 64: 14.8; 128: 45; 256: 157). A period > 1 does not pay off with numba: the window (with the
nominal simulated) stays numpy.

### Raspberry Pi benchmark (please run)

    git clone -b feat/strike-online https://github.com/kimjw624/ps2-rl.git ~/PS2-RL
    cd ~/PS2-RL
    python3 -m venv ~/venv-filter && . ~/venv-filter/bin/activate
    pip install numpy numba
    python3 scripts/bench_cyl_filter.py --quick                     # 1 min: check that it runs
    python3 scripts/bench_cyl_filter.py --out bench_pi4.json        # about 15 min; send bench_pi4.json back

Then `report_cyl_online.py --pi_bench bench_pi4.json` replaces the assumed factor (x4 of the sandbox core) by the Pi's own
fit.

---

## 4. Sweep (`scripts/sweep_cyl_online.py`, `scripts/report_cyl_online.py`)

Realistic plant for every episode (`ps2rl/online/sim.py`): command delay 0..40 ms, rate lag 30..80 ms, thrust lag
20..60 ms, thrust gain +-8 %, rate gain +-5 %, estimate noise 2 cm / 3 cm/s / 0.01 rad, no wind; the observer on.
Scenario sets:

* **stress**: random scenes (2..4 cylinders between x 4 and 11 m, |y| <= 2.5, at least 2 m apart), a short-sighted
  nominal (heading search keeping only 0.45 m from the surfaces, 3 m look-ahead, 1.2 m/s), 4 m sensing. Without the
  filter it passes 0.07 m from a surface; the filter has to step in at almost every cylinder. That nominal keeps
  pointing where the filter cannot follow, so episodes that do not reach the goal are mostly deadlocks (the vehicle
  held in front of a cylinder until the time runs out), never contacts: stress success measures how often the filter
  and its backup still get the vehicle past;
* **blind**: the deployment scenes (section 5: from rest, the first cylinder 1.5 m ahead) with a straight-line
  nominal: a cylinder is on the line in both, the filter must not let the vehicle hit it;
* **final**: the deployment scenes with the deployment nominal (section 5).

Stages: A backup x horizon; B blends x family; C period x horizon. 60 stress episodes per setting.

| backup | blends | family | horizon [s] | period | rows | stress success | no certificate | contacts | numpy, Pi ms | numba, Pi ms |
|---|---|---|---|---|---|---|---|---|---|---|
| hover LQR | 3 | alone | 0.5 | 1 | 4 | 0.35 | 0.005 | 0 | 23 | 0.5 |
| hover LQR | 3 | alone | 0.75 | 1 | 4 | 0.30 | 0.004 | 0 | 34 | 0.8 |
| hover LQR | 3 | alone | 1 | 1 | 4 | 0.27 | 0.004 | 0 | 44 | 1.0 |
| hover LQR | 3 | alone | 1.5 | 1 | 4 | 0.27 | 0.004 | 0 | 66 | 1.4 |
| learned 32 | 2 | alone | 1 | 1 | 3 | 0.62 | 0.008 | 0 | 44 | 1.0 |
| learned 32 | 2 | + LQR | 1 | 1 | 5 | 0.28 | 0.005 | 0 | 45 | 1.2 |
| learned 32 | 3 | alone | 0.5 | 1 | 4 | 0.58 | 0.010 | 0 | 23 | 0.6 |
| learned 32 | 3 | alone | 0.75 | 1 | 4 | 0.55 | 0.007 | 0 | 34 | 0.9 |
| learned 32 | 3 | alone | 1 | 1 | 4 | 0.50 | 0.007 | 0 | 44 | 1.1 |
| learned 32 | 3 | alone | 1.5 | 1 | 4 | 0.53 | 0.007 | 0 | 66 | 1.6 |
| learned 32 | 3 | + LQR | 1 | 1 | 7 | 0.25 | 0.004 | 0 | 47 | 1.5 |
| learned 32 | 5 | alone | 1 | 1 | 6 | 0.52 | 0.007 | 0 | 46 | 1.4 |
| learned 32 | 5 | + LQR | 1 | 1 | 11 | 0.28 | 0.004 | 0 | 51 | 2.0 |
| learned 64 | 2 | alone | 1 | 1 | 3 | 0.52 | 0.005 | 0 | 44 | 1.1 |
| learned 64 | 2 | + LQR | 1 | 1 | 5 | 0.33 | 0.004 | 0 | 46 | 1.5 |
| learned 64 | 3 | alone | 0.5 | 1 | 4 | 0.52 | 0.006 | 0 | 23 | 0.7 |
| learned 64 | 3 | alone | 0.5 | 2 | 4 | 0.38 | 0.006 | 0 | 26 | 3.7 |
| learned 64 | 3 | alone | 0.5 | 5 | 4 | 0.50 | 0.009 | 0 | 35 | 12.7 |
| learned 64 | 3 | alone | 0.75 | 1 | 4 | 0.48 | 0.005 | 0 | 34 | 1.0 |
| learned 64 | 3 | alone | 1 | 1 | 4 | 0.50 | 0.006 | 0 | 45 | 1.3 |
| learned 64 | 3 | alone | 1 | 2 | 4 | 0.45 | 0.006 | 0 | 48 | 4.3 |
| learned 64 | 3 | alone | 1 | 5 | 4 | 0.53 | 0.010 | 0 | 56 | 13.4 |
| learned 64 | 3 | alone | 1.5 | 1 | 4 | 0.53 | 0.005 | 0 | 67 | 1.9 |
| learned 64 | 3 | + LQR | 1 | 1 | 7 | 0.33 | 0.004 | 0 | 48 | 1.9 |
| learned 64 | 5 | alone | 1 | 1 | 6 | 0.48 | 0.006 | 0 | 47 | 1.7 |
| learned 64 | 5 | + LQR | 1 | 1 | 11 | 0.32 | 0.004 | 0 | 52 | 2.6 |
| learned 128 | 3 | alone | 0.5 | 1 | 4 | 0.47 | 0.006 | 0 | 23 | 1.1 |
| learned 128 | 3 | alone | 0.75 | 1 | 4 | 0.63 | 0.005 | 0 | 35 | 1.6 |
| learned 128 | 3 | alone | 1 | 1 | 4 | 0.60 | 0.005 | 0 | 46 | 2.1 |
| learned 128 | 3 | alone | 1.5 | 1 | 4 | 0.60 | 0.004 | 0 | 69 | 3.1 |

Pi ms: the worst-case decision (every rollout to the end, the actor at every step) predicted from the sandbox fit x 4.
Budget: half the decision period, 10 ms at 50 Hz. Contacts and ground strikes: none in any set; every deployment
episode reached the goal. Full table and JSON: `results/cyl_online/sweep_table.md`, `sweep_report.json`; figure
`sweep_horizon.png`.

Confirmation (200 stress episodes, other scenes and plant draws; `results/cyl_online/confirm.json`):

| backup | blends | family | horizon [s] | period | rows | stress success | no certificate | contacts | numpy, Pi ms | numba, Pi ms |
|---|---|---|---|---|---|---|---|---|---|---|
| hover LQR | 2 | alone | 0.5 | 1 | 3 | 0.20 | 0.005 | 0 | 22 | 0.5 |
| learned 32 | 2 | alone | 0.5 | 1 | 3 | 0.49 | 0.007 | 0 | 22 | 0.5 |
| learned 32 | 2 | alone | 0.5 | 2 | 3 | 0.48 | 0.008 | 0 | 25 | 3.5 |
| learned 32 | 2 | alone | 0.5 | 5 | 3 | 0.34 | 0.014 | 0 | 34 | 12.5 |
| learned 32 | 2 | alone | 0.75 | 1 | 3 | 0.41 | 0.007 | 0 | 33 | 0.8 |
| learned 32 | 2 | alone | 1 | 1 | 3 | 0.46 | 0.006 | 0 | 44 | 1.0 |
| learned 32 | 2 | alone | 1.5 | 1 | 3 | 0.50 | 0.006 | 0 | 65 | 1.4 |
| learned 32 | 3 | alone | 0.5 | 1 | 4 | 0.43 | 0.008 | 0 | 23 | 0.6 |
| learned 64 | 2 | alone | 0.5 | 1 | 3 | 0.45 | 0.005 | 0 | 22 | 0.6 |
| learned 128 | 2 | alone | 0.5 | 1 | 3 | 0.49 | 0.006 | 0 | 23 | 0.9 |
| learned 128 | 2 | alone | 1 | 1 | 3 | 0.58 | 0.005 | 0 | 45 | 1.7 |

Findings:

* **No contact and no ground strike in any setting or set.**
* **The learned backup is what gets the vehicle past**: 0.47..0.63 against 0.27..0.35 for the hover LQR (confirmation:
  0.41..0.58 against 0.20), which can only stop.
* **Width 128 > 32 = 64** once the horizon is 0.75 s or more: 0.60..0.63 against 0.48..0.55 (3 blends, 60 episodes;
  the single 0.62 of width 32 with 2 blends came back as 0.46 in the confirmation), 0.58 against 0.46 (confirmation,
  1.0 s). In the deployment test (section 5) it steps in least (2 % of the steps against 7..10 %) and is the only
  width that slides a straight-line nominal round the single cylinder. (With wind, before it was removed, the widths
  were equal.)
* **Blends**: 2 (lam 1, 0) = 3 = 5. More blends only add rows.
* **Family**: adding the hover LQR to the learned backup *hurts* (0.25..0.33 against 0.48..0.62): the LQR plan often
  certifies the same lam first, becomes the committed plan, and can then only stop.
* **Horizon**: no clear trend in stress success (0.41..0.50 for 0.5..1.5 s, width 32), but in the side-by-side gap
  layout tried before the slalom a 0.5 s horizon made the blend switch about twice as often (10 against 5 per episode,
  width 32); 1.0 s and 1.5 s gave identical decisions there (every rollout resolves within 1 s at 1 m/s).
* **Period** 2 or 5 (the plan re-evaluated from the state in between): no gain (0.34..0.53); only for numpy.

**Recommended: numba backend, learned backup 128 wide, 2 blends (lam 1, 0), the learned backup alone, horizon
1.0 s, a decision every control step (50 Hz).** That is 3 rollouts of up to 51 steps per decision (the strike filter:
36 rollouts of 150 steps, 256 wide). Worst case about 0.4 ms on the sandbox core, about 1.7 ms on the Pi if it is 4 times
slower; typical decisions need 12..13 steps. If the Pi turns out much slower: the same with width 64 (steps in on 7 % of
the steps at the start of the one-cylinder course, and stops in front of an obstacle where 128 goes round) or 32. Without numba: width 32, horizon 0.5 s, a decision
every 5 steps (about 35 ms per decision on the Pi, estimated, against a 50 ms budget; stress success 0.34).

As a filter configuration:

    FilterConfig(lambdas=(1.0, 0.0), backups=("learned",), horizon=1.0, period=1, margin=0.1, fresh_margin=0.15,
                 obstacle_range=5.0, backend="numba")       # backup: checkpoints/cyl_online/td3_w128_seed0

---

## 5. Deployment test (`scripts/eval_cyl_final.py`)

**The demo layouts (chosen 2026-11-19 from the candidates below): O3 and S1.** Small virtual cylinders (r 0.1 m),
straight forward from rest at 1.5 m height, 1 m/s, sensing 6 m, the realistic plant without wind, 20 episodes per case.
These are the defaults of `eval_cyl_final.py` and `sim.SceneConfig`.

* **one (O3)**: one cylinder 1.3 m ahead, its top touching the line (centre y = -0.1 m, so the vehicle goes round it on
  the left; its surface 1.2 m from the start: the start hover is in the base set with 0.2 m to spare); goal 2.0 m beyond
  it, at x = 3.3 m.
* **two (S1, slalom)**: the first cylinder 1.1 m ahead, its top touching the line (the start hover exactly at the base
  set's limit: a new plan needs 1.0 m to the surface); the second 2.0 m further on at y = +1.6 m, across the way back to
  the goal: the vehicle has to turn back towards the line early and pass under it (a weave). Goal 1.2 m beyond the
  second, at x = 4.3 m.

Three controllers fly the same episodes: the deployment nominal (heading search keeping 1.0 m) + filter, the nominal
alone, and a straight-line nominal + filter. Episodes end in the 0.5 m arrival circle round the goal.

    python scripts/eval_cyl_final.py --out results/cyl_online/final          # defaults: td3_w128_seed0, 2 blends, 1.0 s, O3 + S1
    python scripts/eval_cyl_final.py --candidates O2,S3 --out /tmp/final_o2_s3   # any other candidate pair

Recommended setting (learned 128, 2 blends, 1.0 s, every step):

| case | controller | success | contact | time to goal [s] | min clearance [m] | mean lam | intervention | route | rollout steps per decision (mean / p99 / max) |
|---|---|---|---|---|---|---|---|---|---|
| O3 | nominal + filter | 1.00 | 0 | 4.7 | 0.68 | 1.00 | 0.00 | left | 11 / 18 / 49 |
| O3 | nominal alone | 1.00 | 0 | 4.7 | 0.68 | - | - | left | - |
| O3 | straight line + filter | 1.00 | 0 | 10.1 | 0.39 | 0.50 | 0.50 | left | 40 / 51 / 51 |
| S1 | nominal + filter | 1.00 | 0 | 6.3 | 0.63 | 0.98 | 0.02 | weave 1.00 | 13 / 51 / 51 |
| S1 | nominal alone | 1.00 | 0 | 6.2 | 0.63 | - | - | weave 1.00 | - |
| S1 | straight line + filter | 1.00 | 0 | 14.1 | 0.38 | 0.38 | 0.62 | weave 1.00 | 41 / 51 / 51 |

* O3 with the planner: the filter never steps in (the planner's own 1.0 m detour keeps the vehicle out). The filter
  shows in the straight-line run: it slides the vehicle round the cylinder along the edge of the hover set (y up to
  0.85 m), lam switching between 1 and 0 every few steps while it slides.
* S1 with the planner: the filter steps in for a few steps right after the start (the start hover is at the limit),
  then the planner weaves on its own. The straight-line run only meets the first cylinder (the straight line from there
  to the goal passes under the second).

Width comparison on O3 / S1 (`results/cyl_online/compare/w{32,64}_final`):

| width | O3 nominal + filter: lam, intervention, time [s] | S1 nominal + filter: lam, intervention, time [s] | straight line + filter: O3 / S1 success | rollout steps per decision (mean, O3 / S1) |
|---|---|---|---|---|
| 32 | 0.99, 0.01, 4.7 | 0.84, 0.16, 7.1 | 0.00 / 0.20 | 13 / 18 |
| 64 | 0.99, 0.01, 4.7 | 0.84, 0.16, 7.2 | 0.00 / 0.00 | 12 / 17 |
| 128 | 1.00, 0.00, 4.7 | 0.98, 0.02, 6.3 | 1.00 / 1.00 | 11 / 13 |

As on the r 0.3 m course, only the 128-wide backup slides a straight-line nominal round the cylinder; with 32 or 64
the straight-line runs stop in front of it and hover there (no contact, timeout). In S1 the narrower backups step in
on 16 % of the steps against 2 %.

Figures: `results/cyl_online/final/final_one.png`, `final_two.png` (top view with every episode, one episode coloured by
lam; its speed, lam and clearance).

### Earlier course (r 0.3 m)

Cylinders of radius 0.3 m, the first 1.5 m ahead, the goal 5 m beyond the last one; the slalom's second cylinder 3.5 m
on at y = +1.5 m (`sim.R03_COURSE`; the sweep's blind / final sets still use it). Results in
`results/cyl_online/final_r03/` and `compare/w{32,64}_final_r03/`; rerun with
`--radius 0.3 --obst_x 1.5 --lateral 0 --first_x 1.5 --first_y -0.3 --second_dx 3.5 --second_y 1.5 --goal_after 5`.

| case | controller | success | contact | time to goal [s] | min clearance [m] | mean lam | route | rollout steps per decision (mean / p99 / max) |
|---|---|---|---|---|---|---|---|---|
| one cylinder 1.5 m ahead | nominal + filter | 1.00 | 0 | 8.3 | 0.58 | 0.98 | - | 13 / 51 / 51 |
| one cylinder 1.5 m ahead | nominal alone | 1.00 | 0 | 8.1 | 0.55 | - | - | - |
| one cylinder 1.5 m ahead | straight line + filter | 1.00 | 0 | 15.8 | 0.39 | 0.56 | - | 36 / 51 / 51 |
| slalom | nominal + filter | 1.00 | 0 | 11.3 | 0.66 | 1.00 | weave 1.00 | 13 / 19 / 51 |
| slalom | nominal alone | 1.00 | 0 | 11.3 | 0.66 | - | weave 1.00 | - |
| slalom | straight line + filter | 0.00 | 0 | - | 0.46 | 0.13 | stopped 1.00 | 48 / 51 / 51 |

With the cylinder 1.5 m ahead the filter works right at the start: the nominal accelerates towards the cylinder while
it turns, and the filter holds lam at 0 for a few steps (the vehicle pauses at about 0.4 m/s) until the turn is under
way. The straight-line nominal + filter: the learned backup's plans slide the vehicle round the single cylinder
(15.8 s); in the slalom it gets round the first and then stops in front of the second (the straight line to the goal
runs into it), without touching either.

How short can the start be on this course (`results/cyl_online/start/start_sweep.json`; the same setting, 20 episodes
per distance; `--start_sweep 1.0,1.15,1.3,1.5,2.0,2.5,3.0,4.0,6.0` with the r 0.3 m arguments):

| start to the first cylinder [m] | surface ahead [m] | one: time [s] | one: lam switches | one: min clearance [m] | slalom: time [s] | slalom: lam switches | slalom: weave | contacts |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.70 | 8.7 | 19.1 | 0.35 | 12.4 | 18.1 | 1.00 | 0 |
| 1.15 | 0.85 | 8.5 | 11.8 | 0.50 | 12.5 | 22.1 | 1.00 | 0 |
| 1.3 | 1.00 | 8.1 | 3.4 | 0.59 | 11.3 | 6.9 | 1.00 | 0 |
| 1.5 | 1.20 | 8.3 | 8.7 | 0.58 | 11.3 | 1.3 | 1.00 | 0 |
| 2 | 1.70 | 8.3 | 0.3 | 0.61 | 11.6 | 0.0 | 1.00 | 0 |
| 2.5 | 2.20 | 8.5 | 0.0 | 0.70 | 12.0 | 0.1 | 1.00 | 0 |
| 3 | 2.70 | 8.9 | 0.0 | 0.69 | 12.5 | 0.1 | 1.00 | 0 |
| 4 | 3.70 | 9.8 | 0.0 | 0.70 | 13.4 | 0.1 | 1.00 | 0 |
| 6 | 5.70 | 11.7 | 0.1 | 0.72 | 15.4 | 0.1 | 1.00 | 0 |

From 2 m on the filter never has to step in; at 1.5 m and closer it does at the start (lam switches), and below
1.3 m the start hover is outside the base set (the filter first moves the vehicle away). Every distance reached the
goal without contact, even 1.0 m (the drone's edge 0.35 m from the surface). **1.5 m is the shortest start with a
certified hover** (1.3 m is the exact limit, no slack for the position hold before the hand-over).

Slalom spacing (first cylinder 1.5 m ahead; `results/cyl_online/spacing/spacing_sweep.json`; `--spacing_sweep
2.0,2.5,3.0,3.5` with the r 0.3 m arguments):

| second cylinder beyond the first [m] | surfaces apart [m] | time [s] | route | lam switches | contacts |
|---|---|---|---|---|---|
| 2 | 2.09 | 10.9 | over both | 3.2 | 0 |
| 2.5 | 2.48 | 11.3 | over both | 2.3 | 0 |
| 3 | 2.90 | 11.7 | over both | 1.8 | 0 |
| 3.5 | 3.34 | 11.3 | weave | 1.3 | 0 |

Closer than 3.5 m the second cylinder is seen while the vehicle is still turning round the first, and the nominal goes
over both in one arc; at 3.5 m it weaves, which is what the layout is for.

Width comparison on this course (`results/cyl_online/compare/w{32,64}_final_r03`):

| width | one: mean lam, intervention, time [s] | slalom: mean lam, intervention, time [s] | straight line + filter: one / slalom success | rollout steps per decision (mean) |
|---|---|---|---|---|
| 32 | 0.90, 0.10, 8.9 | 0.94, 0.06, 11.8 | 0.00 / 0.00 | 16.2 / 15.8 |
| 64 | 0.93, 0.07, 8.6 | 0.98, 0.02, 11.5 | 0.00 / 0.05 | 14.6 / 13.3 |
| 128 | 0.98, 0.02, 8.3 | 1.00, 0.00, 11.3 | 1.00 / 0.00 | 13.3 / 12.9 |

The 128-wide backup lets the nominal through with the fewest interventions (2 % of steps against 7..10 %), and only it
slides a straight-line nominal round the single cylinder.

Figures: `results/cyl_online/final_r03/final_one.png`, `final_two.png`.

### Compact candidates for the demo (virtual cylinders, r 0.1 m)

For a demonstration without real obstacles the cylinders shrink to r = 0.1 m and the start and the goal come in as
close as the base set allows: a new plan needs the start hover 1.0 m from the surface, so **the first cylinder at
least r + 1.0 = 1.1 m ahead**, and the goal at least that far beyond the last one (its hover). The candidates are in
`docker_batch/configs/cyl_online_candidates.json`; the first cylinder's top touches the line (centre y = -0.1 m) so the
vehicle goes round it on the left.

    python scripts/candidates_cyl_final.py --out results/cyl_online/candidates     # all candidates, 20 episodes each
    python scripts/eval_cyl_final.py --candidates O2,S3 --out /tmp/final_o2_s3     # fly another pair as the final test

Same setting (learned 128, 2 blends, 1.0 s), 20 episodes, no contact anywhere. Episodes end in the 0.5 m arrival
circle, so the flown length is about 0.5 m less than the course.

| name | first cylinder ahead | second (beyond the first, y) | goal x (course) | nominal + filter: time [s], y range [m], route | filter overrides (episodes / steps) | straight line + filter: time [s], y up to [m] |
|---|---|---|---|---|---|---|
| O1 | 1.1 m (no slack) | - | 2.3 m | 4.3, 0 .. 1.31, left | 5 % / 0 % | 10.7, 0.84 |
| O2 | 1.3 m | - | 2.5 m | 4.3, 0 .. 1.31, left | 0 / 0 | 11.2, 0.84 |
| O3 | 1.3 m | - | 3.3 m | 4.7, 0 .. 1.30, left | 0 / 0 | 10.1, 0.85 |
| O4 | 2.0 m | - | 4.0 m | 5.1, -1.47 .. 1.26, left 95 % | 0 / 0 | 10.7, 0.84 |
| S1 | 1.1 m (no slack) | 2.0 m, 1.6 m | 4.3 m | 6.3, 0 .. 1.61, weave | 100 % / 2 % | 14.1, 0.81 |
| S2 | 1.1 m (no slack) | 2.2 m, 1.5 m | 4.5 m | 6.5, -0.05 .. 1.64, weave | 95 % / 2 % | 14.7, 0.82 |
| S3 | 1.3 m | 2.0 m, 1.5 m | 4.5 m | 6.7, -0.09 .. 1.65, weave | 100 % / 3 % | 14.1, 0.80 |
| S4 | 1.3 m | 2.5 m, 1.5 m | 5.0 m | 6.5, 0 .. 1.53, weave | 65 % / 0 % | 14.7, 0.84 |

Figures: `results/cyl_online/candidates/candidates_one.png`, `candidates_two.png` (all candidates at the same scale).
Findings:

* With the small cylinders the slalom weaves at every spacing from 2.0 m (at r 0.3 it needed 3.5 m).
* With the planner (heading search, 1.0 m margin) the single cylinder never needs the filter: the planner's own detour
  keeps the vehicle out. The filter is what flies the **straight-line** runs: it slides the vehicle round the cylinder
  along the edge of the hover set (y up to 0.85 m), at 10..11 s instead of 4.3 s, and lam goes 1 / 0 every two or three
  steps while it slides (150..220 switches per run; 3 or 5 blends do not change it, a rise limit on lam would).
* In the slalom the straight-line run is only stopped by the first cylinder (the straight line from there to the goal
  passes under the second). A second cylinder on that line (y 0.4..0.8 at 2.0..2.5 m on) stops the straight-line run
  in front of it (safe, but stuck) or lets it through in 20..26 s, and the planner then goes over both instead of
  weaving: the slalom is a demo for the planner + filter, the single cylinder for the straight line + filter.
* The lateral swing (1.3 m / 1.6 m) is the planner's 1.0 m margin. At 0.75 m (`--search_margin 0.75`) it narrows to
  about 0.9 m / 1.2 m at the same times, but the filter then steps in on 10..20 % of the steps, with the same chatter.
* O4 (the cylinder 2.0 m ahead): one episode in 20 went round the right side; closer in the side is always the left.
* **Chosen: O3 and S1** (now the defaults, results above).

---

## 6. SITL (Gazebo x500) and the onboard chain

The demo flies on PX4 through the ROS 2 bridge (`dual_stage_rl/SITL_src/ps2rl_px4_bridge`, node `ps2rl_cyl_online`,
`README_cyl_online.md` there), with the flight sequence of the slow-track bridge:

    BOOT -> ARM -> TAKEOFF -> GOTO_START (hold 8 s) -> POLICY -> RECOVER (hover 3 s) -> LAND -> DONE
                                                        \-> RECOVER (hover where it is) -> LAND      (abort)

* **Task frame**: origin at the take-off point, x east, z up. PX4 climbs to 1.5 m and holds yaw 0 (east); the hand-over
  waits for the gates (position 0.15 m, speed 0.15 m/s, tilt 5 deg for 8 s) and for a certified plan from the state.
  The cylinders are virtual: they exist only for the filter and the contact check (nothing is placed in Gazebo).
* **Controller** (`ps2rl/deploy/cyl_online_runtime.py`): the simulations' objects and order of calls per step - the
  observer estimate, the sensed cylinders, the nominal, `OnlineFilter.decide`, `command`, the observer update. The
  observer runs on the thrust PX4 reports from take-off on, so it is warm at the hand-over. Course: `course` (a name
  in `cyl_online_candidates.json`, O3 / S1 for the demo) or `obstacles` / `goal`. numba if installed (decision ~0.4 ms),
  else numpy (~4 ms median with the planner, up to ~30 ms in straight-line runs: install numba for those).
* **End** (`ps2rl/deploy/cyl_online_flight.py`): inside the 0.5 m goal circle (the simulations' "reached") below
  0.25 m/s, then PX4 holds where the vehicle is for 3 s and lands it; or after 30 s. **Aborts** to PX4 hold where it is:
  contact with a virtual cylinder, tilt > 35 deg, speed > 2 m/s, below 0.3 m, a non-finite state or command, stale
  odometry, an exception, lost arming / offboard.
* **Records**: `<log_root>/<time>-<label>/flight.csv` (state, commands, lam, filter mode, rollout steps, clearance,
  observer, timing every step) and `summary.json` (outcome, goal time, minimum clearance, lam, modes, compute, rate
  response, course). `scripts/plot_cyl_online_flight.py` draws a flight (top view coloured by lam, speed / lam,
  clearance / height).

Offline rehearsal (no ROS, PX4 or Gazebo; the same flight logic and runtime against the stand-in vehicle of
`landing_plant_sim.py`): `scripts/rehearse_cyl_online_flight.py`, results in `results/cyl_online/rehearsal/`.

| case | goal circle [s] | min clearance [m] | mean lam | intervention | lam switches | y max [m] |
|---|---|---|---|---|---|---|
| O3, planner | 4.7 | 0.72 | 1.00 | 0.00 | 0 | 1.26 |
| O3, straight line | 10.3 | 0.42 | 0.53 | 0.47 | 142 | 0.85 |
| S1, planner | 6.2 | 0.65 | 0.99 | 0.01 | 4 | 1.56 |
| S1, straight line | 28.0 | 0.41 | 0.22 | 0.78 | 126 | 0.80 |

The same as the simulations (4.7 / 10.1 / 6.3 / 14.1 s), except S1 with the straight line: from the first cylinder
1.1 m ahead the filter holds the hover for ~20 s before the vehicle creeps out sideways (the stand-in's slower response;
the slalom is the planner's demo). Over the plant sweep (ideal; lags; 40 ms delay; thrust 8 % high / low; noisy
estimate; all at once) O3 planner, O3 straight line and S1 planner reach the goal every time, no contact, within
0.3 s of the default case.

    python scripts/rehearse_cyl_online_flight.py --all                                  # the table above
    python scripts/rehearse_cyl_online_flight.py --course S1 --sweep                    # plant variations
    python scripts/rehearse_cyl_online_flight.py --course O3 --nominal straight --out /tmp/o3s   # one flight, records
    python scripts/plot_cyl_online_flight.py /tmp/o3s

### SITL results (2026-11-20, Gazebo stock x500, numba; `results/cyl_online/sitl/`)

| flight | outcome | goal circle [s] (simulated, 20 episodes) | min clearance [m] (sim min) | mean lam (sim) | lam switches (sim) | y max [m] (sim) |
|---|---|---|---|---|---|---|
| O3, planner | goal, landed | 4.81 (4.68: 4.64..4.76) | 0.75 (0.68) | 1.00 (1.00) | 0 (0) | 1.14 (1.26) |
| S1, planner | goal (weave), landed | 6.37 (6.30: 6.20..6.44) | 0.59 (0.63) | 1.00 (0.98) | 0 (10) | 1.47 (1.56) |
| O3, straight line | goal, landed | 10.94 (10.15: 9.56..10.72) | 0.41 (0.39) | 0.55 (0.50) | 38 (154) | 0.85 (0.83) |

* The flown paths lie on the simulated ones (`sitl/compare.png`, from `scripts/compare_cyl_flight_sim.py`): no contact,
  no abort, height within 9 cm of 1.5 m (the straight-line run; 1.5 cm with the planner), thrust 9.4-9.9 m/s^2, rates
  up to 0.83 rad/s, disturbance estimate at most 0.19 m/s^2 (the x500 thrust map is right).
* S1: the hand-over point was 3.7 cm further from the first cylinder than the limit, so a fresh plan existed from the
  first step and the filter never had to step in (in the simulations it does for a few steps in most episodes).
* O3 straight line: the filter slid the vehicle round the cylinder as simulated; lam switched 38 times (154 simulated,
  142 rehearsed) and the pitch-rate tracking error rose to 0.42 rad/s RMS (0.04 with the planner) - the chatter.
* Rate response fitted from the flights: delay 35-50 ms + lag 36-44 ms (the simulations drew 0-40 ms + 30-80 ms).
* **Timing**: control steps follow PX4's odometry stamps, which come on an 8 ms grid in SITL, so the interval is 16 or
  24 ms (8-56 ms in all, mean 20.3-20.8 ms); 2-4 % of the steps were missed. The whole control step took 5.8-6.4 ms
  median, p99 27-37 ms, max 31-67 ms, against 1.3 ms offline: the decision itself is 1.1-1.6 ms median (numba); the
  rest is the Python work of the step competing with the node's own ROS executor threads (bridge process 91-93 % of a
  core; the slow-track bridge alone took 75-78 %) and with Gazebo. For the Pi: run the controller in its own process or
  slim the executor (section 8).
* Two flights logged a PX4 estimator reset (odometry reset_counter) during the hold before the hand-over; harmless here
  (the cylinders are virtual, placed in the estimate's frame), but with real obstacles the task frame would have to be
  re-anchored.

### Compute on a Raspberry Pi 4B (2026-11-20)

Measured on the Pi (by the user): the control step alone 7.7-8.2 ms, but 23-32 ms inside the ROS node (CPU 103-107 %),
over the 20 ms budget. The step's own work is not the problem; inside the node it shares Python's GIL with the node's
three executor threads. Changes (bundle `strike_online_sitl_v3`):

* **Controller process** (`process: true`; `ps2rl/deploy/cyl_online_worker.py`): `CylOnlineRuntime` runs in its own
  interpreter behind a proxy with the same interface; the node's control thread sends the state and waits (releasing
  the GIL). Offline the flights are identical to the in-process runtime; the round trip adds ~0.5 ms. `worker_cpu` pins
  it to a core. The flight summary records the round trip (`latency_ms`) and the worker's own time (`compute_ms`); the
  node logs the controller process's CPU next to its own.
* **One executor thread** (`executor: single`, or the environment variable `PS2RL_EXECUTOR`).
* **40 ms control period** (`control_period: 0.04`, 25 Hz): one command per 40 ms, held by PX4. The filter's rollouts
  hold every command as long (`FilterConfig.hold`, in numpy and in the numba kernel: the safe set checked every 20 ms
  model step, the base-set hand-over only at command times), and the simulator runs the controller at command times
  (`run_episodes(hold=)`). The hover LQR still contracts its Lyapunov level with the hold (worst ratio over two steps
  0.9876 held vs 0.9877 with fresh feedback, linearised). Simulation, 20 episodes, no contact anywhere:

  | case | 20 ms | 40 ms |
  |---|---|---|
  | O3 planner | 4.7 s, clearance 0.68 m | 4.7 s, 0.66 m |
  | O3 straight line | 10.1 s, 0.39 m | 10.2 s, 0.37 m |
  | S1 planner (weave) | 6.3 s, 0.63 m | 6.3 s, 0.62 m |
  | S1 straight line | 14.1 s | 13.4 s |
  | O3, r 0.27 (Pi test cylinder), planner / straight | 5.0 / 12.1 s | 5.0 / 12.2 s |

  The rehearsal (full flight logic) at 40 ms: O3 planner 4.7 s, O3 straight 10.4 s (58 lam switches instead of 142),
  S1 planner 6.2 s, S1 straight 17.1 s. hold = 1 reproduces every earlier result exactly; numba and numpy make identical
  decisions for period 1 / 2 and hold 1 / 2 (`check_cyl_online_fast.py --holds 1,2`).
* **Nominal reuse**: when the filter takes a fresh plan its nominal inputs are this step's, so the second nominal call is
  skipped (identical commands; the step 13-14 % shorter).
* **Not changed: the horizon.** 1.0 s stays. Shorter horizons do not show with the planner (the filter is idle) but slow
  the straight-line flight (the demo): O3 r 0.27 straight 12.1 s at 1.0 s, 16.6 s at 0.75 s, 29-34 s at 0.5 s (the hover
  held 53-67 % of the time, one timeout in ten) - `results/cyl_online/control_period/horizon_study.json`. A decision
  period of 2 is free (12.4 s) but does not shorten the decision steps themselves.

New defaults in `cyl_online_x500.yaml`: `control_period 0.04`, `process true`, `executor single` (the node sets
`policy_rate` from the control period). As flown in SITL on 2026-11-20: `control_period:=0.02 process:=false
executor:=multi`. Pi test: `cyl_compute_pi_test.sh` (A as flown, B / C new defaults planner / straight, D / E 20 ms with
the process and one executor thread).

**Second round: the ROS side (2026-11-20, after the Pi measurements; the bridge's `README_cyl_online.md` has the
details).** With the above, the Pi's bridge process still filled its core (92 %), three quarters of it inside rclpy's
executor (about 225 passes a second). The steps fell behind: 72 % done, step p95 133 ms. All five flights first lost
offboard after a time-sync jump. The bridge now has:
* **`io: poll`** (`polled_loop.py`): no executor.
  - It sleeps until shortly before the odometry message that makes the next step due, then steps at once
    (about 50 wake-ups a second).
  - The stream is quiet while steps publish; `setpoint_rate` is 25.
* **`outgoing_stamp: receipt`**: stamp 0, which PX4 v1.16 / v1.17 replace with their time of receipt. The old stamp
  (the guarded odometry clock) was 6 s old after a +6 s time-sync jump, so PX4 counted the offboard signal as lost.
  The fix also applies to the slow-track node.
* **Topic names from px4_msgs** (`vehicle_local_position_v1` on v1.17).

Measured against a stand-in PX4 on real ROS 2 topics (`tools/fake_px4.py`, `tools/bench_fake.py`). The bridge and the
controller each had 15 % of an x86 core ("a Pi core"); with that limit the pushed code reproduced the Pi's failure.
* Pushed code: 256 / 312 steps, step median / p95 40 / 70 ms, bridge at its limit.
* `io: poll`: 334 / 334 steps, 4.9 / 14.8 ms, bridge about 40 % of its core.
* On a core 1.5 times slower again: 337 / 338 steps, 11.8 / 22.9 ms.
* With time-sync jumps of +6 / +3 / +2.5 s, `receipt` flew to the goal; the old stamping lost offboard.

## 7. Files

| file | what |
|---|---|
| `ps2rl/envs/strike_scene.py` | + `omega_z_max` (separate yaw-rate limit), + no chaser slots (`n_chaser_slots 0`) |
| `ps2rl/phase1_sa/strike_sa.py` | region sampler without chaser slots |
| `docker_batch/configs/cyl_online_p1.json` | Phase-I task (box, LQR, base set, regions, TD3) |
| `docker_batch/run_cyl_online_phase1.sh` | train every width, export |
| `ps2rl/online/model.py` | numpy plant, hover LQR, sets, observation, actor, disturbance observer, export format |
| `ps2rl/online/filter.py` | the filter (numpy / numba backends) |
| `ps2rl/online/fast.py` | the compiled rollouts (numba) |
| `ps2rl/online/nominal.py` | go-to-goal + heading search |
| `ps2rl/online/sim.py` | scenes (defaults: the O3 / S1 demo layouts; `R03_COURSE`), sensing, realistic plant, batched closed loop |
| `scripts/export_cyl_backup.py` | Phase-I run -> numpy export, checked against JAX |
| `scripts/check_cyl_online_fast.py` | numba vs numpy decisions |
| `scripts/bench_cyl_filter.py` | decision timing (run it on the Pi) |
| `scripts/sweep_cyl_online.py`, `scripts/report_cyl_online.py` | the sweep and its report |
| `scripts/eval_cyl_final.py` | the deployment test, start-distance, slalom-spacing and gap sweeps; `--candidates` |
| `docker_batch/configs/cyl_online_candidates.json` | compact demo layouts (r 0.1 m) |
| `scripts/candidates_cyl_final.py` | flies every candidate layout: table and same-scale figures |
| `checkpoints/cyl_online/` | exported backups: `lqr`, `td3_w{32,64,128}_seed0` |
| `ps2rl/deploy/cyl_online_runtime.py`, `cyl_online_flight.py` | the onboard controller and flight logic (section 6) |
| `ps2rl/deploy/cyl_online_worker.py` | the controller in its own process (`process: true`) |
| `scripts/rehearse_cyl_online_flight.py`, `scripts/plot_cyl_online_flight.py` | offline rehearsal; flight figure |
| `scripts/compare_cyl_flight_sim.py` | flights (SITL / hardware) against the simulations of the same course |
| `results/cyl_online/` | bench, sweep and confirmation (table / report / figure), deployment test (`final`: O3 / S1; `final_r03`: the r 0.3 m course), start-distance and slalom-spacing sweeps (r 0.3 m), width comparison, compact candidates, rehearsal, SITL flights, Phase-I runs |

## 8. Open

* Raspberry Pi timing (section 3): the recommendation assumes the Pi core is 4 times slower than the sandbox's
  (the margin is large: 1.7 ms against a 10 ms budget).
* The blend chatters (lam 1 / 0 from one step to the next) when the filter steps in: right after the start in S1, and
  all the way round the cylinder in the straight-line runs; a rise limit on lam (the strike filter's `lam_rise_max`)
  would smooth it. 3 or 5 blends do not.
* Hardware (Tarot 650, confirmed 2026-11-20; radius about 0.52 m against the model's 0.35 m): obstacle positions from sensing / motion capture instead of virtual ones; a frame re-anchor on
  estimator resets.
* Compute on the Pi: confirm `io: poll` + receipt stamps on the Pi (`tools/bench_fake.py` there, then the SITL harness).
  If still short: numba the decision's Python setup and the command (now ~60 % of the step); a C++ I/O node; or a
  `rate_limit` on vehicle_odometry in PX4's `dds_topics.yaml` (firmware build).
* Wind (left out for now): the adversarial certificate of the base set (section 1).
* PS2 (a striker trained through the filter) was not repeated: the go-to-goal nominal is enough for the straight test.
* The width-128 advantage rests on one training seed per width.
* Obstacles from a forward camera (limited field of view) instead of 360 deg sensing.
