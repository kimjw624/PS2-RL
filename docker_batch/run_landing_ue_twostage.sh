#!/usr/bin/env bash
# Landing inside the cone under a disturbance (UE-bCBF), two-stage Phase II from a task file
# (docker_batch/configs/landing_ue_x500.json: backup, disturbance model, filter, environment):
#   V  vanilla tracker   (SAC, no CIL, episodes not cut at a violation; the warm-start source). One run per
#                        critic reward scale in VANILLA_SCALES, in parallel; the one with the best evaluation
#                        return (environment units) is the warm start (OUT/vanilla_choice.json)
#   P  PS2               (SAC through the UE-bCBF CIL, warm-started from V; the recipe of the shipped 4 g run
#                         outputs/landing_phase2_ue/ue_p2_warm_s0)
#   E  evaluation        (vanilla: none / nominal / ue; PS2: ue / none; same starts and disturbance draws,
#                         full episodes)
#   F  figures           (two-stage figure: tracker without CIL / tracker + UE-CIL / Phase II; the same with the
#                         nominal bCBF; training curves)
#
# Stage V on the GPU, stages P and E on the CPU (float64 QP; faster there):
#   docker exec -d px4sitl bash -c "STAGES=V bash ~/ws_shared/PS2-RL/docker_batch/run_landing_ue_twostage.sh \
#       > ~/ws_shared/PS2-RL/outputs/landing_ue_twostage_V.out 2>&1"
#   docker exec -d px4sitl bash -c "PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu STAGES='P E F' \
#       bash ~/ws_shared/PS2-RL/docker_batch/run_landing_ue_twostage.sh > ~/ws_shared/PS2-RL/outputs/landing_ue_twostage_PEF.out 2>&1"
# (docker_batch/run_x500_ue.sh runs both tasks of the x500 study in this order.)
#
# Resumable: a training run with a summary.json is skipped, an evaluation with an eval.json is skipped.
# Knobs (environment variables):
#   OUT            output root                                    (outputs/landing_phase2_ue_x500)
#   TASK           task file                                      (docker_batch/configs/landing_ue_x500.json)
#   SEED           seed of both stages                            (0)
#   VANILLA_STEPS  stage V steps                                  (3000000)
#   VANILLA_LOST   stage V: end an episode once the position error exceeds this [m], with the discounted floor
#                  reward of the remaining steps as the last reward (ending early never pays); 0 = off  (1.5)
#   VANILLA_SCALES critic reward scales tried in stage V           ("0.1 1")
#   PS2_STEPS      stage P steps                                  (1000000)
#   EPISODES       evaluation episodes                            (128)
#   STAGES         subset of "V P E F"                            (all)
#   VANILLA_ARGS / PS2_ARGS / EVAL_ARGS / PLOT_ARGS   extra arguments
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/landing_phase2_ue_x500}"
TASK="${TASK:-docker_batch/configs/landing_ue_x500.json}"
SEED="${SEED:-0}"
VANILLA_STEPS="${VANILLA_STEPS:-3000000}"
VANILLA_LOST="${VANILLA_LOST:-1.5}"
VANILLA_SCALES="${VANILLA_SCALES:-0.1 1}"
PS2_STEPS="${PS2_STEPS:-1000000}"
EPISODES="${EPISODES:-128}"
STAGES="${STAGES:-V P E F}"
VANILLA_ARGS="${VANILLA_ARGS:-}"
PS2_ARGS="${PS2_ARGS:-}"
EVAL_ARGS="${EVAL_ARGS:-}"
PLOT_ARGS="${PLOT_ARGS:-}"
mkdir -p "${OUT}/logs"
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }
any_alive() { for p_ in "$@"; do kill -0 "${p_}" 2>/dev/null && return 0; done; return 1; }

ps2rl_gpu_check || { echo "[land-ue] GPU check failed (set PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu to force CPU)"; exit 1; }
V_BASE="land_ue_vanilla_s${SEED}"
P_TAG="land_ue_ps2_s${SEED}"
V_CHOICE="${OUT}/vanilla_choice.json"
scale_tag() { echo "${V_BASE}_rs$(echo "$1" | tr '.' 'p')"; }
echo "[land-ue] $(date '+%F %T') OUT=${OUT} TASK=${TASK} SEED=${SEED} stages ${STAGES}"

if has_stage V; then
    pids=()
    for rs in ${VANILLA_SCALES}; do
        tag="$(scale_tag "${rs}")"
        if [ -f "${OUT}/${tag}/summary.json" ]; then
            echo "[land-ue] V: ${OUT}/${tag} done, skipped"
            continue
        fi
        echo "[land-ue] V: vanilla tracker, reward scale ${rs}, ${VANILLA_STEPS} steps -> ${OUT}/${tag} (log ${OUT}/logs/${tag}.log)"
        # shellcheck disable=SC2086
        python scripts/train_phase2_landing_ue_cached.py --task "${TASK}" --output_root "${OUT}" --run_tag "${tag}" \
            --seed "${SEED}" --use_projection false --env_terminate_on_unsafe false --env_terminate_pos_error "${VANILLA_LOST}" \
            --reward_scale "${rs}" --total_steps "${VANILLA_STEPS}" \
            --num_envs 64 --update_every 2 --replay_size 1000000 --eval_every 100352 ${VANILLA_ARGS} \
            > "${OUT}/logs/${tag}.log" 2>&1 &
        pids+=($!)
        sleep 20
    done
    if [ "${#pids[@]}" -gt 0 ]; then
        while any_alive "${pids[@]}"; do
            sleep "${V_REPORT_EVERY:-300}"
            for rs in ${VANILLA_SCALES}; do
                tag="$(scale_tag "${rs}")"
                echo "[land-ue] V $(date '+%T') ${tag}: $(grep -E '^\s*\[eval' "${OUT}/logs/${tag}.log" 2>/dev/null | tail -1 | cut -c1-170)"
            done
        done
        wait
    fi
    python - "${OUT}" "${V_CHOICE}" "${V_BASE}" ${VANILLA_SCALES} <<'PY'
import json, sys
from pathlib import Path
out, choice, v_base, scales = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
rows = []
for rs in scales:
    tag = f"{v_base}_rs{rs.replace('.', 'p')}"
    sp = out / tag / "summary.json"
    if not sp.exists():
        print(f"[land-ue] V: {tag} has no summary (failed?)"); continue
    ev = json.loads(sp.read_text())["best_eval"]
    rows.append((ev["return_mean"], tag, float(rs), ev))
    print(f"[land-ue] V: {tag}: best return {ev['return_mean']:.1f} pos_err {ev['pos_err_mean']:.3f} "
          f"final dist {ev['final_dist_mean']:.3f} landed {ev['landed_rate']:.2f} unsafe {ev['unsafe_rate']:.2f}")
if not rows:
    sys.exit("[land-ue] V: no vanilla run finished")
ret, tag, rs, ev = max(rows, key=lambda r: r[0])
choice.write_text(json.dumps({"tag": tag, "reward_scale": rs, "best_eval": ev}, indent=2))
print(f"[land-ue] V: warm start = {tag} (reward scale {rs})")
PY
    [ -f "${V_CHOICE}" ] || { echo "[land-ue] V failed"; exit 1; }
fi

[ -f "${V_CHOICE}" ] && V_TAG="$(python -c "import json;print(json.load(open('${V_CHOICE}'))['tag'])")" \
    && V_RS="$(python -c "import json;print(json.load(open('${V_CHOICE}'))['reward_scale'])")"
V_TAG="${V_TAG:-$(scale_tag "$(echo ${VANILLA_SCALES} | cut -d' ' -f1)")}"
V_RS="${V_RS:-$(echo ${VANILLA_SCALES} | cut -d' ' -f1)}"

if has_stage P; then
    if [ -f "${OUT}/${P_TAG}/summary.json" ]; then
        echo "[land-ue] P: ${OUT}/${P_TAG} done, skipped"
    else
        [ -f "${OUT}/${V_TAG}/best_weights.pkl" ] || { echo "[land-ue] P: no stage-V weights ${OUT}/${V_TAG}/best_weights.pkl (run STAGES=V first)"; exit 1; }
        echo "[land-ue] P: warm start ${V_TAG} (reward scale ${V_RS}); PS2 through the UE-bCBF CIL, ${PS2_STEPS} steps"
        # the 4 g run's stage-2 recipe (outputs/landing_phase2_ue/ue_p2_warm_s0/configs.json)
        # shellcheck disable=SC2086
        JAX_ENABLE_X64=1 python scripts/train_phase2_landing_ue_cached.py --task "${TASK}" --output_root "${OUT}" \
            --run_tag "${P_TAG}" --seed "${SEED}" --reward_scale "${V_RS}" \
            --warm_start_weights "${OUT}/${V_TAG}/best_weights.pkl" --total_steps "${PS2_STEPS}" \
            --start_steps 0 --update_after 16384 --update_every 8 --replay_size 500000 --num_envs 64 \
            --actor_lr 5e-5 --critic_lr 1e-4 --alpha_lr 5e-5 --project_target_actions true --rows_keep 32 \
            --eval_every 100352 ${PS2_ARGS} 2>&1 | tee "${OUT}/logs/${P_TAG}.log"
        [ -f "${OUT}/${P_TAG}/summary.json" ] || { echo "[land-ue] P failed"; exit 1; }
    fi
fi

if has_stage E; then
    for spec in "${V_TAG}:none,nominal,ue" "${P_TAG}:ue,nominal,none"; do
        run="${spec%%:*}"; modes="${spec#*:}"
        [ -f "${OUT}/${run}/best_weights.pkl" ] || { echo "[land-ue] E: ${run} has no weights, skipped"; continue; }
        if [ -f "${OUT}/${run}/eval/eval.json" ]; then
            echo "[land-ue] E: ${run} evaluated, skipped"
            continue
        fi
        echo "[land-ue] E: ${run} (${modes}), ${EPISODES} episodes"
        # shellcheck disable=SC2086
        JAX_ENABLE_X64=1 python scripts/eval_phase2_landing_ue.py --run "${OUT}/${run}" --modes "${modes}" \
            --episodes "${EPISODES}" --full_episodes ${EVAL_ARGS} 2>&1 | tee "${OUT}/logs/eval_${run}.log"
    done
fi

if has_stage F; then
    V_EV="${OUT}/${V_TAG}/eval"
    P_EV="${OUT}/${P_TAG}/eval"
    if [ -f "${V_EV}/eval.json" ] && [ -f "${P_EV}/eval.json" ]; then
        # shellcheck disable=SC2086
        python scripts/plot_landing_ue_twostage.py \
            --panel "Stage 1 tracker, no CIL=${V_EV}:none" \
            --panel "Stage 1 tracker + UE-CIL (Phase II start)=${V_EV}:ue" \
            --panel "Phase II trained through UE-CIL=${P_EV}:ue" \
            --out "${OUT}/landing_ue_twostage.png" ${PLOT_ARGS} 2>&1 | tee "${OUT}/logs/plot.log"
        # shellcheck disable=SC2086
        python scripts/plot_landing_ue_twostage.py \
            --panel "Stage 1 tracker, no CIL=${V_EV}:none" \
            --panel "Stage 1 tracker + nominal bCBF=${V_EV}:nominal" \
            --panel "Stage 1 tracker + UE-CIL=${V_EV}:ue" \
            --panel "Phase II trained through UE-CIL=${P_EV}:ue" \
            --out "${OUT}/landing_ue_twostage_with_nominal.png" ${PLOT_ARGS} 2>&1 | tee -a "${OUT}/logs/plot.log"
        # shellcheck disable=SC2086
        python scripts/plot_landing_ue_twostage.py \
            --panel "Phase II through UE-CIL=${P_EV}:ue" \
            --panel "Phase II through the nominal bCBF=${P_EV}:nominal" \
            --panel "Phase II, filter removed=${P_EV}:none" \
            --out "${OUT}/landing_ue_phase2_filters.png" ${PLOT_ARGS} 2>&1 | tee -a "${OUT}/logs/plot.log"
        python scripts/plot_landing_ue_curves.py --root "${OUT}" --vanilla_glob "${V_BASE}_rs*" --ps2 "${P_TAG}" \
            --out "${OUT}/landing_ue_curves.png" 2>&1 | tee -a "${OUT}/logs/plot.log"
    else
        echo "[land-ue] F: evaluations missing, skipped"
    fi
fi
echo "[land-ue] $(date '+%F %T') done"
