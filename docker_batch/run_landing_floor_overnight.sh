#!/usr/bin/env bash
# Overnight: floor-aware landing redesign. Resumable; every stage skips work that is done.
#
#   A  baseline tests    existing backup (cone-only) with the floor added to S, several filter settings
#   B  Phase-I variants  floor in S (+ optional gentle-recovery envelope), several seeds, GPU;
#                        each run is certified (Prop. 1 incl. floor/recovery bounds) before training
#   C  self-check        check_landing_bcbf.py per trained run: recoverability vs Phase I,
#                        cone stress test, floor stress test, QP health (PASS/FAIL)
#   D  reach tests       test_landing_floor_reach.py per trained run x filter settings:
#                        hover map, LQR landing (touchdown), tracking a cone-cutting on-pad reference
#   E  colleague policy  evaluate_landing_phase2_filter.py (seed-0 run of each variant, one setting)
#   F  report            summarize_landing_floor.py -> $OUT/REPORT.md
#
#     bash ~/ws_shared/PS2-RL/docker_batch/run_landing_floor_overnight.sh
#
# Detached, from the HOST:
#     docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_floor_overnight.sh \
#         > ~/ws_shared/PS2-RL/outputs/landing_floor_overnight.out 2>&1"
#     docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_floor_overnight.out
#
# Knobs (environment variables):
#   OUT=outputs/landing_floor
#   BASE_CFG=checkpoints/landing_phase1/land_v2_miss_td3_seed0/phase1_config.json   Phase-I config the variants extend
#   BASE_BACKUP=checkpoints/landing_phase1/land_v2_miss_td3_seed0                    stage-A backup
#   PHASE2_RUN=outputs/landing/land_v2_miss_s1        colleague run (reference start; stage E). Skipped if absent.
#   TRACK_REF=ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz
#   VARIANTS="name:extra train flags|name2:..."        Phase-I variants (see default below)
#   SEEDS="0 1"  BACKBONE=td3  TOTAL_STEPS=5000000  MAX_PARALLEL=2  P1_EXTRA_ARGS=""
#   SETTINGS="name:KEY=VAL,...;name2:..."              filter settings for stages A/D (floor_constraint=on is added)
#   CHECK_SETTING="alpha=10,alpha_floor=20"            filter setting for stage C / E
#   TEST_ARGS=""                                       extra flags for test_landing_floor_reach.py
#   CHECK_ARGS=""                                      extra flags for check_landing_bcbf.py
#   EVAL_EPISODES=64                                   stage-E episodes
#   STAGES="A B C D E F"
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"
ps2rl_gpu_check || exit 1

OUT="${OUT:-outputs/landing_floor}"
BASE_CFG="${BASE_CFG:-checkpoints/landing_phase1/land_v2_miss_td3_seed0/phase1_config.json}"
BASE_BACKUP="${BASE_BACKUP:-checkpoints/landing_phase1/land_v2_miss_td3_seed0}"
PHASE2_RUN="${PHASE2_RUN:-outputs/landing/land_v2_miss_s1}"
TRACK_REF="${TRACK_REF:-ps2rl/envs/assets/quadrotor_landing_cornercut_reference.npz}"
FLOOR="--floor_constraint true --region_edge_floor_prob 0.3"
VARIANTS="${VARIANTS:-floor:${FLOOR}|floor_frec10:${FLOOR} --recovery_rate_floor 10|floor_rec10:${FLOOR} --recovery_rate_cone 10 --recovery_rate_floor 10|floor_rec20:${FLOOR} --recovery_rate_cone 20 --recovery_rate_floor 20}"
SEEDS="${SEEDS:-0 1}"
BACKBONE="${BACKBONE:-td3}"
TOTAL_STEPS="${TOTAL_STEPS:-5000000}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
P1_EXTRA_ARGS="${P1_EXTRA_ARGS:-}"
SETTINGS="${SETTINGS:-a4:alpha=4;a10_f10:alpha=10,alpha_floor=10;a10_f20:alpha=10,alpha_floor=20;a20_f20:alpha=20,alpha_floor=20;a10_f20_fixedtau:alpha=10,alpha_floor=20,relative_time_floor=off}"
CHECK_SETTING="${CHECK_SETTING:-alpha=10,alpha_floor=20}"
TEST_ARGS="${TEST_ARGS:-}"
CHECK_ARGS="${CHECK_ARGS:-}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
STAGES="${STAGES:-A B C D E F}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}" "${OUT}/phase1" "${OUT}/checks" "${OUT}/tests" "${OUT}/phase2_eval"

# floor_constraint=on in every setting (a no-op for checkpoints trained with the floor)
FLOOR_SETTINGS="$(python - "${SETTINGS}" <<'PY'
import sys
out = []
for chunk in [c for c in sys.argv[1].split(";") if c.strip()]:
    name, _, body = chunk.partition(":")
    items = [t for t in body.split(",") if t.strip() and not t.startswith("floor_constraint")]
    out.append(f"{name}:" + ",".join(["floor_constraint=on"] + items))
print(";".join(out))
PY
)"
P2_ARGS=()
[ -f "${PHASE2_RUN}/configs.json" ] && P2_ARGS=(--phase2_run "${PHASE2_RUN}")
TRACK_ARGS=()
[ -f "${TRACK_REF}" ] && TRACK_ARGS=(--tracking_reference "${TRACK_REF}")
has_stage() { case " ${STAGES} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
stamp() { date '+%H:%M:%S'; }
echo "[$(stamp)] out=${OUT} stages=(${STAGES}) seeds=(${SEEDS}) steps=${TOTAL_STEPS}"
echo "[$(stamp)] settings: ${FLOOR_SETTINGS}"

# ------------------------------------------------------------------ A
if has_stage A && [ ! -f "${OUT}/tests/baseline/floor_tests.json" ]; then
    echo "[$(stamp)] [A] baseline tests with ${BASE_BACKUP}"
    # shellcheck disable=SC2086
    python scripts/test_landing_floor_reach.py --backup "${BASE_BACKUP}" "${P2_ARGS[@]}" "${TRACK_ARGS[@]}" \
        --settings "as_trained:;${FLOOR_SETTINGS}" --out "${OUT}/tests/baseline" ${TEST_ARGS} \
        > "${LOGS}/A_baseline.log" 2>&1 || echo "[$(stamp)] [A] FAILED (see ${LOGS}/A_baseline.log)"
    grep -E "^\[" "${LOGS}/A_baseline.log" | tail -n 12
fi

# ------------------------------------------------------------------ B
RUNS=()
IFS='|' read -r -a VARS <<< "${VARIANTS}"
for seed in ${SEEDS}; do
    for v in "${VARS[@]}"; do
        RUNS+=("${v%%:*}_${BACKBONE}_seed${seed}|${v#*:}|${seed}")
    done
done
if has_stage B; then
    echo "[$(stamp)] [B] ${#RUNS[@]} Phase-I runs, ${MAX_PARALLEL} at a time"
    running=0
    for r in "${RUNS[@]}"; do
        IFS='|' read -r name flags seed <<< "${r}"
        if [ -f "${OUT}/phase1/${name}/landing_backup_policy_actor.pkl" ]; then
            echo "[$(stamp)] [B] ${name}: done, skipping"
            continue
        fi
        rm -rf "${OUT:?}/phase1/${name}"
        echo "[$(stamp)] [B] start ${name}: ${flags}"
        # shellcheck disable=SC2086
        python scripts/train_phase1_landing.py --config_json "${BASE_CFG}" --backbone "${BACKBONE}" --seed "${seed}" \
            --total_steps "${TOTAL_STEPS}" --output_dir "${OUT}/phase1" --run_name "${name}" ${flags} \
            ${PS2RL_GPU_FLAG} ${P1_EXTRA_ARGS} > "${LOGS}/B_${name}.log" 2>&1 &
        running=$((running + 1))
        if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
            wait -n || true
            running=$((running - 1))
        fi
    done
    wait || true
    for r in "${RUNS[@]}"; do
        name="${r%%|*}"
        if [ -f "${OUT}/phase1/${name}/landing_backup_policy_actor.pkl" ]; then
            echo "[$(stamp)] [B] ${name}: $(grep '^\[done\]' "${LOGS}/B_${name}.log" | tail -n1)"
        else
            echo "[$(stamp)] [B] ${name}: FAILED or aborted by the certificate, see ${LOGS}/B_${name}.log"
            grep -E "not certified|Error|error" "${LOGS}/B_${name}.log" | tail -n 3
        fi
    done
fi

TRAINED=()
for r in "${RUNS[@]}"; do
    name="${r%%|*}"
    [ -f "${OUT}/phase1/${name}/landing_backup_policy_actor.pkl" ] && TRAINED+=("${name}")
done
CHECK_OVR=()
IFS=',' read -r -a _c <<< "${CHECK_SETTING}"
for kv in "${_c[@]}"; do [ -n "${kv}" ] && CHECK_OVR+=(--cbf_override "${kv}"); done

# ------------------------------------------------------------------ C
if has_stage C; then
    for name in "${TRAINED[@]}"; do
        [ -f "${OUT}/checks/${name}/check_landing_bcbf.json" ] && continue
        echo "[$(stamp)] [C] self-check ${name} (${CHECK_SETTING})"
        # shellcheck disable=SC2086
        python scripts/check_landing_bcbf.py --backup "${OUT}/phase1/${name}" "${CHECK_OVR[@]}" \
            --out "${OUT}/checks/${name}" ${CHECK_ARGS} > "${LOGS}/C_${name}.log" 2>&1
        grep -E "^\[(PASS|FAIL|check)" "${LOGS}/C_${name}.log" | tail -n 5
    done
fi

# ------------------------------------------------------------------ D
if has_stage D; then
    for name in "${TRAINED[@]}"; do
        [ -f "${OUT}/tests/${name}/floor_tests.json" ] && continue
        echo "[$(stamp)] [D] reach tests ${name}"
        # shellcheck disable=SC2086
        python scripts/test_landing_floor_reach.py --backup "${OUT}/phase1/${name}" "${P2_ARGS[@]}" "${TRACK_ARGS[@]}" \
            --settings "${FLOOR_SETTINGS}" --out "${OUT}/tests/${name}" ${TEST_ARGS} \
            > "${LOGS}/D_${name}.log" 2>&1 || echo "[$(stamp)] [D] ${name} FAILED (see ${LOGS}/D_${name}.log)"
        grep -E "^\[" "${LOGS}/D_${name}.log" | tail -n 8
    done
fi

# ------------------------------------------------------------------ E
if has_stage E && [ "${#P2_ARGS[@]}" -gt 0 ]; then
    for name in "${TRAINED[@]}"; do
        case "${name}" in *_seed0) ;; *) continue ;; esac
        dst="${OUT}/phase2_eval/${name}"
        [ -f "${dst}/summary.json" ] && continue
        echo "[$(stamp)] [E] colleague policy with ${name} (${CHECK_SETTING})"
        python scripts/evaluate_landing_phase2_filter.py "${P2_ARGS[@]}" --backup "${OUT}/phase1/${name}" \
            "${CHECK_OVR[@]}" --num_episodes "${EVAL_EPISODES}" --gif_cases nominal --out "${dst}" ${PS2RL_GPU_FLAG} \
            > "${LOGS}/E_${name}.log" 2>&1 || echo "[$(stamp)] [E] ${name} FAILED (see ${LOGS}/E_${name}.log)"
        grep -E "^\[(starts|filtered|unfiltered)\]" "${LOGS}/E_${name}.log"
    done
fi

# ------------------------------------------------------------------ F
if has_stage F; then
    python scripts/summarize_landing_floor.py --root "${OUT}" && echo "[$(stamp)] report: ${OUT}/REPORT.md"
fi
echo "[$(stamp)] done"
