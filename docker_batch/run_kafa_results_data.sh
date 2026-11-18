#!/usr/bin/env bash
# KAFA results figures: per-step data and episode metadata of the chosen paper-setting episodes (scripts/kafa_export.py).
#   1. GPU EXACT records: the paper evaluations ps2_final, ps2_guid1, residual_final (1000 scenes, seed 9001, seed-0 models),
#      recording only the chosen episodes (--record_episodes) with the extra per-step fields (--traj_extra 1);
#   2. settings check, reproduction of the paper's per-episode GPU records, expected outcomes (stops on a mismatch);
#   3. CSV + JSON per episode, filter snapshots (601: the single-escape filter's first certificate loss t* and t* - 1 s,
#      both filters; 845: the last certificate loss), README_data.md, kafa_results_data_1111.zip.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_kafa_results_data.sh > ~/ws_shared/PS2-RL/outputs/kafa_results_data.out 2>&1"
#
# Knobs: ROOT (results/strike_reduced/final/run if present, else outputs/strike_final), OUT (outputs/kafa_results_data),
#        STAGES ("rec post"), ALLOW_MISMATCH (0; 1: export even if the check finds a difference, e.g. CPU tests).
#        A record whose strike_filter_traj.npz exists is reused. Never edit this file while it runs.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

if [ -z "${ROOT:-}" ]; then
    for r in results/strike_reduced/final/run outputs/strike_final; do
        [ -f "${r}/p1/td3_a0p2_seed0/configs.json" ] && { ROOT="${r}"; break; }
    done
fi
ROOT="${ROOT:-outputs/strike_final}"
OUT="${OUT:-outputs/kafa_results_data}"
STAGES="${STAGES:-rec post}"
TASK=docker_batch/configs/strike_final_ps2.json
ENV_JSON='{"actual_chaser_radius": -1.0, "actual_chaser_speed": -1.0, "chaser_accel": 2.0}'
REC="${OUT}/rec"
mkdir -p "${REC}"
for need in p1/td3_a0p2_seed0/configs.json residual/striker_rs0p1_s0/best_weights.pkl ps2/striker_rs0p1_s0/best_weights.pkl \
            eval/s0/ps2_final/strike_filter_episodes.json eval/s0/ps2_guid1/strike_filter_episodes.json \
            eval/s0/residual_final/strike_filter_episodes.json; do
    [ -f "${ROOT}/${need}" ] || { echo "[kafa-data] missing ${ROOT}/${need} (set ROOT to the paper run)"; exit 1; }
done
# the deployed filter and its single-escape ablation, as in docker_batch/run_strike_final_eval.sh
guid() { python - "${TASK}" "$1" <<'PY'
import json, sys
g = json.loads(json.load(open(sys.argv[1]))["tracker"].get("filter_guidance") or "{}")
if sys.argv[2] == "one":
    g.pop("escape_variants", None)
print(json.dumps(g))
PY
}
G_FINAL="$(guid final)"; G_ONE="$(guid one)"
stamp() { date '+%F %T'; }
CMDS="${OUT}/commands.sh"
log_cmd() {  # one line per command, single-quoted where needed (UTF-8 kept readable)
    local line="" a q="'"
    for a in "$@"; do
        if [[ "${a}" =~ ^[A-Za-z0-9_./:=,+@%-]+$ ]]; then line+="${a} "; else line+="${q}${a//${q}/${q}\\${q}${q}}${q} "; fi
    done
    echo "${line% }" >> "${CMDS}"
}
run() { log_cmd "$@"; "$@"; }
git_head() {
    git rev-parse HEAD 2>/dev/null && return 0
    local h r; [ -f .git/HEAD ] || { echo "unknown (no .git)"; return 0; }
    h="$(cat .git/HEAD)"
    if [[ "${h}" == ref:* ]]; then
        r="${h#ref: }"
        if [ -f ".git/${r}" ]; then echo "$(cat ".git/${r}") (${r#refs/heads/})"
        else echo "$(grep " ${r}\$" .git/packed-refs 2>/dev/null | cut -d' ' -f1) (${r#refs/heads/})"; fi
    else echo "${h}"; fi
}
echo "# $(stamp): docker_batch/run_kafa_results_data.sh (ROOT=${ROOT} OUT=${OUT})" >> "${CMDS}"
echo "\`$(git_head)\` + code bundle ${CODE_REF:-kafa_results_code_1111}" > "${OUT}/commit.txt"
echo "[kafa-data] $(stamp) root ${ROOT}; out ${OUT}; commit $(cat "${OUT}/commit.txt"); jax $(python -c 'import jax; print(jax.default_backend())')"
echo "[kafa-data] final filter: ${G_FINAL}"
echo "[kafa-data] single-escape filter: ${G_ONE}"

rec() {  # rec <eval run> <striker dir> <guidance> <modes> <episodes...>
    local name=$1 st=$2 g=$3 modes=$4; shift 4
    local out="${REC}/${name}"
    if [ -f "${out}/strike_filter_traj.npz" ]; then echo "[kafa-data] ${name}: recorded before, reused"; return 0; fi
    echo "[kafa-data] $(stamp) ${name}: ${st}, modes '${modes}', 1000 scenes (seed 9001), recording $*"
    run python scripts/eval_strike_filter.py --backup_run "${ROOT}/p1/td3_a0p2_seed0" --task "${TASK}" --env "${ENV_JSON}" \
        --nominal striker --striker_run "${ROOT}/${st}" --chaser_model guidance --guidance "${g}" --modes "${modes}" \
        --episodes 1000 --scene_seed 9001 --record_episodes "$@" --save_traj 1 --traj_extra 1 --save_episodes 1 \
        --n_show 0 --gif 0 --out "${out}" > "${out}.log" 2>&1 || { echo "[kafa-data] ${name} failed (${out}.log)"; exit 1; }
    grep -E "^\[(chaser|none|learned)" "${out}.log" | grep -v smoothness | cut -c1-170 | sed "s/^/[kafa-data]   /"
}

if [[ " ${STAGES} " == *" rec "* ]]; then
    rec ps2_final ps2/striker_rs0p1_s0 "${G_FINAL}" learned 601 972 146 845 957
    rec ps2_guid1 ps2/striker_rs0p1_s0 "${G_ONE}" learned 601
    rec residual_final residual/striker_rs0p1_s0 "${G_FINAL}" "none learned" 972 146 957
fi
[[ " ${STAGES} " == *" post "* ]] || exit 0

K="python scripts/kafa_export.py"
echo "[kafa-data] $(stamp) settings check, reproduction of the paper records, expected outcomes"
run ${K} check --rec "${REC}" --root "${ROOT}" --task "${TASK}" 2>&1 | grep -v Warning > "${OUT}/check.txt"
sed "s/^/[kafa-data]   /" "${OUT}/check.txt"
if grep -q "^\[check\] MISMATCH" "${OUT}/check.txt"; then
    if [ "${ALLOW_MISMATCH:-0}" = 1 ]; then
        echo "[kafa-data] WARNING: mismatch (${OUT}/check.txt); continuing because ALLOW_MISMATCH=1"
    else
        echo "[kafa-data] stopped: a setting or an expected outcome differs (${OUT}/check.txt)"
        exit 1
    fi
fi

echo "[kafa-data] $(stamp) snapshots"
times="$(${K} times --rec "${REC}" 2>/dev/null | tail -1)"
echo "[kafa-data]   ${times}"
tget() { python -c "import json, sys; v = json.loads(sys.argv[1]).get(sys.argv[2]); print('' if v is None else v)" "${times}" "$1"; }
T1="$(tget 601_t_before)"; T2="$(tget 601_t_star)"; T3="$(tget 845_t_last_loss)"
snap() {  # snap <eval run> <guidance> <episode> <times...>
    local name=$1 g=$2 ep=$3; shift 3
    [ $# -gt 0 ] && [ -n "$1" ] || { echo "[kafa-data]   ${name} ${ep}: no time (no certificate loss)"; return 0; }
    run python scripts/strike_filter_snapshot.py --replay "${REC}/${name}" --mode learned --episode "${ep}" --times "$@" \
        --backup_run "${ROOT}/p1/td3_a0p2_seed0" --task "${TASK}" --env "${ENV_JSON}" --guidance "${g}" --lang ko --terms paper \
        --plans_json 1 --out "${OUT}/snapshots/${name}" 2>&1 | grep -E "^\[snapshot\]|^wrote|rror" | sed "s/^/[kafa-data]   /"
}
snap ps2_guid1 "${G_ONE}" 601 ${T1} ${T2}
snap ps2_final "${G_FINAL}" 601 ${T1} ${T2}
snap ps2_final "${G_FINAL}" 845 ${T3}

echo "[kafa-data] $(stamp) export"
run ${K} export --rec "${REC}" --out "${OUT}" --commit "${OUT}/commit.txt" --commands "${CMDS}" --check "${OUT}/check.txt" \
    2>&1 | grep -E "^\||^wrote" | sed "s/^/[kafa-data]   /"
( cd "$(dirname "${OUT}")" && rm -f kafa_results_data_1111.zip \
    && zip -qr kafa_results_data_1111.zip "$(basename "${OUT}")" -x "*/rec/*.npz" )
echo "[kafa-data] $(stamp) done: ${OUT} and $(dirname "${OUT}")/kafa_results_data_1111.zip"
