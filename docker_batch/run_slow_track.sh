#!/usr/bin/env bash
# Slow-reference tracker (forward + loop) for PX4 SITL and hardware, on the GPU:
#   R  reference   generate ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_reference.npz (skipped if it exists)
#   T  training    one SAC run per seed in SEEDS, in parallel (the cone-obstacle stage-1 tracker recipe)
#   E  evaluation  each run's best weights on the robustness scenarios (EPISODES per scenario) + figures
#   S  selection   the run with the lowest robust score -> OUT/choice.json
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_slow_track.sh \
#       > ~/ws_shared/PS2-RL/outputs/slow_track.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/slow_track.out"
#
# Resumable: a run with a summary.json is skipped, an evaluation with an eval.json is skipped.
# `touch OUT/<run>/STOP` ends a run early (it finishes normally: final evaluation, weights, summary).
# Knobs (environment variables):
#   OUT          output root                                       (outputs/slow_track)
#   REF          reference .npz                                    (ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_reference.npz)
#   REF_ARGS     arguments of the reference generator              ("": 0.4 m/s, loop radius 1.5 m)
#   SEEDS        seeds trained in parallel                         ("0 1 2")
#   STEPS        training steps per run                            (5000000)
#   EPISODES     evaluation episodes per scenario                  (64)
#   STAGES       subset of "R T E S"                               (all)
#   TRAIN_ARGS   extra arguments of scripts/train_slow_track.py    (e.g. "--env_a_max 13.0 --env_omega_xy_max 1.5")
#   TAG          run name prefix: runs are OUT/<TAG>_s<seed>       (slow_loop)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/slow_track}"
REF="${REF:-ps2rl/envs/assets/slow_loop/quadrotor_slow_loop_reference.npz}"
REF_ARGS="${REF_ARGS:-}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-5000000}"
EPISODES="${EPISODES:-64}"
STAGES="${STAGES:-R T E S}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
TAG="${TAG:-slow_loop}"
mkdir -p "${OUT}/logs"
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

ps2rl_gpu_check || { echo "[slow-track] GPU check failed (set PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu to force CPU)"; exit 1; }
echo "[slow-track] $(date '+%F %T') OUT=${OUT} REF=${REF} SEEDS=${SEEDS} STEPS=${STEPS} stages ${STAGES}"

if has_stage R && [[ ! -f "${REF}" ]]; then
    python ps2rl/envs/assets/generate_quadrotor_slow_loop_reference.py --out "${REF}" ${REF_ARGS} \
        2>&1 | tee "${OUT}/logs/reference.log" || exit 1
fi

if has_stage T; then
    pids=()
    for s in ${SEEDS}; do
        tag="${TAG}_s${s}"
        if [[ -f "${OUT}/${tag}/summary.json" ]]; then echo "[slow-track] ${tag} done, skipped"; continue; fi
        echo "[slow-track] $(date '+%T') training ${tag}"
        # several runs share one GPU: no preallocation, each takes what it needs
        XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/train_slow_track.py --output_root "${OUT}" --run_tag "${tag}" \
            --seed "${s}" --total_steps "${STEPS}" --env_reference_path "${REF}" ${TRAIN_ARGS} \
            > "${OUT}/logs/${tag}.log" 2>&1 &
        pids+=($!)
    done
    for p_ in "${pids[@]}"; do wait "${p_}"; done
    echo "[slow-track] $(date '+%T') training finished"
fi

if has_stage E; then
    for s in ${SEEDS}; do
        tag="${TAG}_s${s}"
        [[ -f "${OUT}/${tag}/best_weights.pkl" ]] || { echo "[slow-track] ${tag}: no weights"; continue; }
        if [[ -f "${OUT}/${tag}/eval_best/eval.json" ]]; then echo "[slow-track] eval ${tag} done, skipped"; continue; fi
        python scripts/eval_slow_track.py --run "${OUT}/${tag}" --weights best --episodes "${EPISODES}" \
            2>&1 | tee "${OUT}/logs/eval_${tag}.log"
    done
fi

if has_stage S; then
    python - "${OUT}" "${TAG}" ${SEEDS} <<'EOF'
import json, sys
from pathlib import Path
out, tag_prefix, seeds = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
rows = []
for s in seeds:
    p = out / f"{tag_prefix}_s{s}" / "eval_best" / "eval.json"
    if p.exists():
        ev = json.loads(p.read_text())
        rows.append((ev["score"], f"{tag_prefix}_s{s}", ev["scenarios"]))
if not rows:
    sys.exit("[slow-track] no evaluation to select from")
rows.sort(key=lambda r: r[0])
print(f"{'run':16s} {'score':>7s}  " + "  ".join(f"{k[:12]:>12s}" for k in rows[0][2]))
for sc, tag, ev in rows:
    print(f"{tag:16s} {sc:7.4f}  " + "  ".join(f"{100 * v['pos_err_mean']:9.1f} cm" + ("*" if v["lost_rate"] > 0 else " ")
                                         for v in ev.values()))
best = rows[0]
(out / f"choice_{tag_prefix}.json").write_text(json.dumps({"run": best[1], "score": best[0]}, indent=2))
print(f"[slow-track] chosen: {best[1]} (score {best[0]:.4f}); * = some episodes lost -> {out / f'choice_{tag_prefix}.json'}")
EOF
fi
echo "[slow-track] $(date '+%F %T') done"
