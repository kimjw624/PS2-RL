#!/usr/bin/env bash
# Landing Phase II: does the BCBF filter keep a trained landing policy inside the cone?
#
#   1) (optional) import the colleague's PS2-RL.tar          -> assets + outputs/landing/<run>
#   2) derive the Phase-I config from the Phase-II run       -> cone/pad from the reference config,
#                                                               plant from the run, design region widened
#   3) train the matching safe-arrival backup on the GPU     (skipped if it already exists)
#   4) evaluate policy-only vs policy+filter on the same starts -> tables, figures, GIFs
#   5) reachability analysis: C_N slices near the pad, filtered-LQR landing tests, BCBF-setting
#      sensitivity, and attribution (policy request vs filter, binding QP rows)
#
#     bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_filter_eval.sh
#     IMPORT=~/ws_shared/PS2-RL.tar bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_filter_eval.sh
#
# Knobs (environment variables):
#     IMPORT=""                              PS2-RL.tar or extracted folder to import first
#     RUNS="outputs/landing/land_v2_miss_s1" Phase-II run dirs (space separated) to evaluate
#     CHECKPOINT=best                        best | final
#     BACKBONE=td3  SEED=0  TOTAL_STEPS=5000000   Phase-I training
#     P1_EXTRA_ARGS=""                       extra flags for train_phase1_landing.py (e.g. "--region_v_max 3.0")
#     BACKUP=""                              use this Phase-I checkpoint instead of training one
#     EPISODES=64  EVAL_EXTRA_ARGS=""        evaluation (e.g. "--gif_cases nominal,worst_unfiltered,most_filter_effort",
#                                            "--cbf_override alpha=10", "--cbf_override include_relative_time_term=false")
#     REACH=1  REACH_EXTRA_ARGS=""           step 5 (REACH=0 skips it; e.g. REACH_EXTRA_ARGS="--variants 'alpha=10;alpha=20'")
#
# Detached, from the HOST:
#     docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_filter_eval.sh \
#         > ~/ws_shared/PS2-RL/outputs/landing_phase2_filter.out 2>&1"
#     docker exec -it px4sitl tail -f ~/ws_shared/PS2-RL/outputs/landing_phase2_filter.out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"
ps2rl_gpu_check || exit 1

IMPORT="${IMPORT:-}"
RUNS="${RUNS:-outputs/landing/land_v2_miss_s1}"
CHECKPOINT="${CHECKPOINT:-best}"
BACKBONE="${BACKBONE:-td3}"
SEED="${SEED:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-5000000}"
P1_EXTRA_ARGS="${P1_EXTRA_ARGS:-}"
BACKUP="${BACKUP:-}"
EPISODES="${EPISODES:-64}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
REACH="${REACH:-1}"
REACH_EXTRA_ARGS="${REACH_EXTRA_ARGS:-}"
mkdir -p outputs/landing_phase1_cfg outputs/landing_phase1_p2 outputs/landing_phase2_filter

if [ -n "${IMPORT}" ]; then
    bash "${HERE}/import_phase2_landing_runs.sh" "${IMPORT}"
fi

for RUN in ${RUNS}; do
    [ -f "${RUN}/configs.json" ] || { echo "[p2filter] ${RUN}/configs.json not found (set IMPORT=... or RUNS=...)" >&2; exit 1; }
    echo "================ ${RUN}"
    backup="${BACKUP}"
    if [ -z "${backup}" ]; then
        # 2) Phase-I config for this run; runs with the same cone/plant/region share one backup.
        tmp_cfg="$(mktemp --suffix=.json)"
        JAX_PLATFORMS=cpu python scripts/landing_phase1_config_from_phase2.py --phase2_run "${RUN}" --out "${tmp_cfg}"
        name="$(python - "${tmp_cfg}" <<'PY'
import hashlib, json, sys
from pathlib import Path
c = json.load(open(sys.argv[1]))
key = json.dumps({"landing": c["landing"], "region": c["region"]}, sort_keys=True)
print(f"{Path(c['provenance']['reference_path']).stem}_{hashlib.sha1(key.encode()).hexdigest()[:8]}")
PY
)"
        cfg="outputs/landing_phase1_cfg/${name}.json"
        mv "${tmp_cfg}" "${cfg}"
        p1_run="${name}_${BACKBONE}_seed${SEED}"
        backup="outputs/landing_phase1_p2/${p1_run}"
        # 3) train the backup unless it exists
        if [ ! -f "${backup}/landing_backup_policy_actor.pkl" ]; then
            echo "[p2filter] training the safe-arrival backup ${backup} (${TOTAL_STEPS} steps, ${BACKBONE})"
            # shellcheck disable=SC2086
            python scripts/train_phase1_landing.py --config_json "${cfg}" --backbone "${BACKBONE}" --seed "${SEED}" \
                --total_steps "${TOTAL_STEPS}" --output_dir outputs/landing_phase1_p2 --run_name "${p1_run}" \
                ${PS2RL_GPU_FLAG} ${P1_EXTRA_ARGS} 2>&1 | tee "outputs/landing_phase1_p2/${p1_run}.log"
        else
            echo "[p2filter] reusing backup ${backup}"
        fi
    fi
    # 4) evaluation: policy only vs policy + filter (QP in float64; the script enables x64 itself)
    eval_log="$(mktemp)"
    # shellcheck disable=SC2086
    python scripts/evaluate_landing_phase2_filter.py --phase2_run "${RUN}" --checkpoint "${CHECKPOINT}" \
        --backup "${backup}" --num_episodes "${EPISODES}" ${PS2RL_GPU_FLAG} ${EVAL_EXTRA_ARGS} 2>&1 | tee "${eval_log}"
    eval_dir="$(sed -n 's/^\[done\] //p' "${eval_log}" | tail -n1)"
    rm -f "${eval_log}"
    # 5) reachability / conservativeness analysis next to the evaluation
    if [ "${REACH}" = "1" ] && [ -n "${eval_dir}" ]; then
        # shellcheck disable=SC2086
        eval python scripts/analyze_landing_filter_reach.py --backup "${backup}" --phase2_run "${RUN}" \
            --eval_dir "${eval_dir}" ${REACH_EXTRA_ARGS}
    fi
done
echo "[p2filter] results under outputs/landing_phase2_filter/"
