# Cone obstacle: base-set design and target-cylinder size (CPU study)

2026-10-09 · x500 limits (thrust 1.99–14.95 m/s², roll/pitch rate 3.84 rad/s, yaw rate 0.203 rad/s) · backup-CBF filter, 2 s horizon (N = 100, dt 0.02), alpha 10 (funnel) / 20 (ground), discrete safeguard on · LQR backups only (no Phase I)

Task: the outside of an apex-down funnel (tip at `tip_z`, half-angle θ, tip rounding ε) and the space above the ground are safe. The drone approaches from −x and must touch a vertical target cylinder (radius R, from the ground up to a top height H). Touching is enough; it does not need to hover inside.

Code: local branch `feat/cone-obstacle` in PS2-RL. Additions only; the landing task is unchanged. Commands are in `docker_batch/README_cone_obstacle.md`.

## Recommendation for the GPU runs

1. **Base set: the PS2-RL paper's band.** Use the 7-D hover LQR (altitude, velocity, attitude; horizontal position free), centred halfway between the ground and the tip, at c = 0.98 × min(LQR bound 28.66, geometric cap). At tip 1.2 m that gives c = 12.4. A single 9-D hover ellipsoid under the tip behaves the same in every test, so it is a good second option.
2. **Skip Phase I for now.** With the LQR backups, the filter's protected set (C_N) already covers about 100 % of the hover states and 89–96 % of the states moving at 1 m/s toward the wall. A learned safe-arrival policy has little left to add. Train Phase II directly with the band's LQR as the backup.
3. **Cylinder: R = 0.25 m, top at the tip height.** Reachability does not limit R: every radius from 0.1 to 0.75 m was touched in every run with a goal-directed nominal. The top height decides whether the funnel matters at all. A top 0.25 m below the tip makes the funnel irrelevant (the unfiltered nominal succeeds and the filter stays idle).
4. **Open decision: where to put the cylinder.** If it is centred under the tip and reaches the ground, the funnel never hides it. A trained policy will simply dive to its near side and will rarely need the filter. To make the cone stand in the way (the "slide below it" scenario), put the cylinder on the far side, e.g. centred 1.5 m past the axis with its top at the tip height. Then every straight path to its top cuts the funnel, all unfiltered goal-directed nominals collide, and the band filter slides them under the tip to 100 % success.

## Q1: why not a smaller base set on the approach side?

Nothing in the certificates blocks it.
- The 9-D hover LQR is certified up to c = 28.66, the same for every centre. The yaw-rate limit (and the thrust margin) bind.
- The ellipsoid's half-extents scale with √c: 0.212 m horizontally and 0.169 m vertically per √c.
- An ellipsoid beside the funnel at approach altitude (2 m before the axis, 0.5 m above the tip) fits up to c = 22.2. Any smaller c fits too.
- Coverage stays high. At c = 2, under the tip it drops to 91 % at hover and 77 % at 1 m/s, because the backup has to fly back out under the wall.

The cost shows up in the closed loop. The filter keeps the vehicle where the backup can still return to B within 2 s, so moving away from an approach-side B is resisted:
- An altitude-holding nominal that should slide down the wall instead **stalls on the wall** (6–12 of 12 starts). It never passes under the tip.
- A target on the far side is **unreachable**: 4 of 4 runs stall just past the tip.

Base sets below the funnel (the band, or a point under the tip) make the backup point down and under. That is what turns the filter's correction into the slide you described.

## Base-set designs compared (tip 1.2 m, θ 45°, ε 0.25)

| design | c | C_N hover: approach / under tip | C_N at 1 m/s toward axis: approach / under tip | goal-directed (aim at tip): touch R ≥ 0.1 | wall slide (altitude-holding): touch / stalls | filter cost |
|---|---|---|---|---|---|---|
| band (paper), mid-gap | 12.4 | 100 / 100 % | 96 / 100 % | 100 %, active 59 % | 33 % / 2 of 12 | 1× |
| point under the tip | 12.4 | 100 / 100 % | 96 / 100 % | 100 %, active 58 % | 50 % / 0 of 12 | 1× |
| point beside the funnel (approach side) | 21.8 | 99.8 / 98.7 % | 95 / 100 % | 100 %, active 75 % | 0 % / 9 of 12 | 1× |
| same, smaller | 2 | 99.8 / 91.2 % | 95 / 77 % | 100 % | 0 % / 9–12 of 12 | 1× |
| brake: "stop wherever clear" (new) | 2 | 100 / 100 % | 89 / 100 % | 100 %, active 54 % | 17 % / 6 of 12 | ~5× |
| no filter | – | – | – | 0 % (100 % collide) | 0 % (100 % collide) | – |

Starts: 3 m before the axis; 0 or 0.25 m to the side; 0.5, 1 or 2 m above the tip; at rest or 1 m/s toward the axis. The wall slide can touch at most 50 %: starts 0.25 m to the side slide around the tip instead of under it. The band at c_max/4 and an approach-side point under the wall give the same results as the band. The union of beside + under-tip does too, except at tip 1.6 m, where 3 of 12 wall slides stall. The same ranking holds for tip heights 0.8 / 1.6 m and half-angles 30° / 60°. No run with a recoverable start was unsafe.

**Brake set (the new design).** The base controller is a 6-D LQR on velocity and attitude only. B holds the states whose whole stopping transient clears the funnel and the ground. This includes every level vehicle at rest in S. Results:
- The first version used a fixed clearance and was not invariant: 0.3–1.7 % of sampled states left B. The rest point drifts in the nonlinear loop by about 1 % of c.
- Scaling the clearance with the remaining energy fixes it: 0 % leave B and 0 % leave S over 6 s at c = 1–8.

The brake works, but it is more conservative near walls. It can only stop, not dive, so it has more stalls and smaller C_N at speed. With a blunt tip (ε = 1), two starts 0.16 m below the wall moving 1 m/s toward it are outside its C_N but inside the band's, and those runs collided. Its filter is also about 5× slower, because each step solves the bisection for the steer target.

## Target cylinder size

- **R** (0.1–0.75 m): with the band or under-tip base set, goal-directed nominals touched every radius from every start, with no unsafe runs. This held for all tips, half-angles, and the blunt tip. A wall-sliding nominal touches only when it passes over the axis, so R barely changes its success.
- **Top height H:**
  - At the tip: the goal-directed path to the axis grazes the rounded tip. Unfiltered it always collides; filtered it always touches, with the filter active 44–59 % of steps.
  - 0.25 m below the tip: the unfiltered nominal also succeeds and the filter is idle.
- **Is the funnel in the way at all?** Geometry only, starts 1–3 m from the axis, 0.2–2.5 m above the tip, below the outer wall:

| cylinder | funnel | straight path to the nearest top edge cuts the funnel | cylinder fully hidden | detour to a visible point |
|---|---|---|---|---|
| centred, R ≥ 0.25, top at tip | ε 0.25, any θ | 0 % | 0 % | 0 % |
| centred, R 0.25, top at tip | ε 1.0 (blunt), θ 45° | 29 % | 0 % | +0.4 % (p90 +1.8 %) |
| far side (1.5 m past the axis), R 0.25–0.5, top at tip height | ε 0.25, θ 45° | 100 % | 0–3 % | +4–6.5 % (p90 +10–13 %) |

- **Far-side cylinder in the closed loop** (centre 1.5 m past the axis, top at tip height):
  - Unfiltered, the nominals aimed at the top centre, the near top edge, or straight ahead collide in 100 % of runs. Aimed at the cylinder's centre, 25 % collide.
  - With the band or under-tip filter, every goal-directed run slides under the tip and touches (R 0.25 and 0.5), with the filter active about 9 % of steps.
  - The approach-side base set stalls past the tip in 4 of 4 runs.
  - The brake reaches 100 %, but more slowly.

## Certificates (x500)

| LQR | c_U | c_chart | c_Lyap (adversarial) | c_max | position half-extent per √c |
|---|---|---|---|---|---|
| 9-D hover (points) | 28.66 | 103.1 | ≥ 28.66 | 28.66 | 0.212 m horizontal, 0.169 m vertical |
| 7-D hover (band, paper) | 28.66 | 103.1 | ≥ 28.66 | 28.66 | 0.169 m vertical |
| 6-D brake | 28.66 | 103.1 | ≥ 28.66 | 28.66 | stopping transient: 0.269 m horizontal, 0.148 m vertical |

Geometric caps for the band at mid-gap: 5.6 / 12.6 / 22.4 at tip 0.8 / 1.2 / 1.6 m. The level hardly matters: the band at c_max/4 behaved identically.

## Caveats

- Nominals are LQR stand-ins for a policy: aim at the cylinder centre, the top centre, or the near top edge, or fly straight at the start altitude. A trained policy will exploit what the geometry allows, which is why where the cylinder sits matters more than its radius.
- Results are numerical (sampled C_N, adversarial Lyapunov ascent, sampled invariance), not proofs.
- Every start should be checked to be in C_N before training. `make_recoverability_fn` does this, and the reach script now records it per run (`start_in_CN`).
