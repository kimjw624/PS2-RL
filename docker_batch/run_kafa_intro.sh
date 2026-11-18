#!/usr/bin/env bash
# KAFA presentation, introduction figures in the paper settings (scripts/kafa_intro.py):
#   1. GPU EXACT records: the full 1000-scene evaluation (seed 9001) of A (residual striker, no filter) and B (PS2
#      striker + final filter), recording only the shortlisted scenes (--record_episodes; bit for bit the evaluation);
#   2. settings check (section 0) and reproduction of the paper's per-episode GPU records;
#   3. shortlist (top-5 per failure type, shortlist.png), shared cameras, the 4 motivation GIFs (+ PNG, MP4),
#      col1 / col2 / col3 images, README_intro.md (table, picks, check log, commands, commit), kafa_intro_1110.zip.
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_kafa_intro.sh > ~/ws_shared/PS2-RL/outputs/kafa_intro.out 2>&1"
#
# Knobs: ROOT (results/strike_reduced/final/run if present, else outputs/strike_final), OUT (outputs/kafa_gifs/intro), OBS (957), CHA (146),
#        SHORT_OBS / SHORT_CHA (top-5 candidates of the CPU screening), COL3 (the 6 random scenes), STAGES ("rec render").
# Resumable: a record whose strike_filter_traj.npz exists is reused. Never edit this file while it runs.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

# the paper run: the repo copy, else the pipeline's own output folder
if [ -z "${ROOT:-}" ]; then
    for r in results/strike_reduced/final/run outputs/strike_final; do
        [ -f "${r}/p1/td3_a0p2_seed0/configs.json" ] && { ROOT="${r}"; break; }
    done
fi
ROOT="${ROOT:-outputs/strike_final}"
for need in p1/td3_a0p2_seed0/configs.json residual/striker_rs0p1_s0/best_weights.pkl ps2/striker_rs0p1_s0/best_weights.pkl \
            eval/s0/residual_final/strike_filter_episodes.json eval/s0/ps2_final/strike_filter_episodes.json; do
    [ -f "${ROOT}/${need}" ] || { echo "[kafa] missing ${ROOT}/${need} (set ROOT to the paper run)"; exit 1; }
done
export KAFA_EVAL="${ROOT}/eval/s0"          # the paper's GPU per-episode records, compared episode by episode
OUT="${OUT:-outputs/kafa_gifs/intro}"
OBS="${OBS:-957}"
CHA="${CHA:-146}"
SHORT_OBS="${SHORT_OBS:-141 957 607 201 415}"
SHORT_CHA="${SHORT_CHA:-146 523 173 549 69}"
COL3="${COL3:-850 744 274 540 335 227}"
STAGES="${STAGES:-rec render}"
TASK=docker_batch/configs/strike_final_ps2.json
ENV_JSON='{"actual_chaser_radius": -1.0, "actual_chaser_speed": -1.0, "chaser_accel": 2.0}'
G="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['tracker']['filter_guidance'])" "${TASK}")"
REC="${OUT}/rec"
mkdir -p "${REC}"
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
echo "# $(stamp): docker_batch/run_kafa_intro.sh (ROOT=${ROOT} OUT=${OUT} OBS=${OBS} CHA=${CHA})" >> "${CMDS}"
git_head() {  # the commit, also without git in the container (reads .git directly)
    git rev-parse HEAD 2>/dev/null && return 0
    local h r; [ -f .git/HEAD ] || { echo "unknown (no .git)"; return 0; }
    h="$(cat .git/HEAD)"
    if [[ "${h}" == ref:* ]]; then
        r="${h#ref: }"
        if [ -f ".git/${r}" ]; then echo "$(cat ".git/${r}") (${r#refs/heads/})"
        else echo "$(grep " ${r}\$" .git/packed-refs 2>/dev/null | cut -d' ' -f1) (${r#refs/heads/})"; fi
    else echo "${h}"; fi
}
{ echo "\`$(git_head)\` + code bundle ${CODE_REF:-kafa_intro_code_1110b (local commit of scripts/kafa_intro.py, run_kafa_intro.sh)}"; } > "${OUT}/commit.txt"
echo "${ROOT}" > "${OUT}/root.txt"
echo "[kafa] $(stamp) root ${ROOT}; out ${OUT}; commit $(head -1 "${OUT}/commit.txt"); jax $(python -c 'import jax; print(jax.default_backend())')"
echo "[kafa] filter guidance: ${G}"

rec() {  # rec <A|B> <striker dir> <mode> <episodes...>
    local tag=$1 st=$2 mode=$3; shift 3
    local out="${REC}/${tag}"
    if [ -f "${out}/strike_filter_traj.npz" ]; then echo "[kafa] ${tag}: recorded before, reused"; return 0; fi
    echo "[kafa] $(stamp) ${tag}: ${st}, mode ${mode}, 1000 scenes (seed 9001), recording $*"
    run python scripts/eval_strike_filter.py --backup_run "${ROOT}/p1/td3_a0p2_seed0" --task "${TASK}" --env "${ENV_JSON}" \
        --nominal striker --striker_run "${ROOT}/${st}" --chaser_model guidance --guidance "${G}" --modes "${mode}" \
        --episodes 1000 --scene_seed 9001 --record_episodes "$@" --save_traj 1 --save_episodes 1 --n_show 0 --gif 0 \
        --out "${out}" > "${out}.log" 2>&1 || { echo "[kafa] ${tag} failed (${out}.log)"; exit 1; }
    head -3 "${out}.log" | sed "s/^/[kafa]   /"
    grep -E "^\[(none|learned)" "${out}.log" | cut -c1-170 | sed "s/^/[kafa]   /"
}

if [[ " ${STAGES} " == *" rec "* ]]; then
    # shellcheck disable=SC2086
    rec A residual/striker_rs0p1_s0 none ${SHORT_OBS} ${SHORT_CHA} ${COL3}
    # shellcheck disable=SC2086
    rec B ps2/striker_rs0p1_s0 learned ${SHORT_OBS} ${SHORT_CHA}
fi
[[ " ${STAGES} " == *" render "* ]] || exit 0

K="python scripts/kafa_intro.py"
echo "[kafa] $(stamp) settings check and reproduction of the paper records"
run ${K} check --a "${REC}/A" --b "${REC}/B" --root "${ROOT}" --task "${TASK}" --episodes "${OBS},${CHA}" 2>&1 | grep -v Warning \
    | tee "${OUT}/check.txt" | sed "s/^/[kafa]   /"
grep -q "paper settings OK" "${OUT}/check.txt" || { echo "[kafa] settings MISMATCH: stopped (see ${OUT}/check.txt)"; exit 1; }
grep -q "paper records OK" "${OUT}/check.txt" || echo "[kafa] WARNING: A / B not reproduced exactly against the paper's GPU records (check.txt)"

echo "[kafa] $(stamp) shortlist"
run ${K} screen --a "${REC}/A" --b "${REC}/B" --out "${OUT}" 2>&1 | grep -E "^\||^###|^wrote" | sed "s/^/[kafa]   /"
run ${K} cam --a "${REC}/A" --b "${REC}/B" --episode "${OBS}" --type obstacle 2>/dev/null | tail -1 > "${OUT}/cam_obs.json"
run ${K} cam --a "${REC}/A" --b "${REC}/B" --episode "${CHA}" --type chaser 2>/dev/null | tail -1 > "${OUT}/cam_cha.json"
echo "[kafa] cameras: $(cat "${OUT}/cam_obs.json") / $(cat "${OUT}/cam_cha.json")"

gif() {  # gif <name> <A|B> <episode> <cam>
    local name=$1 tag=$2 ep=$3 cam=$4 mode=none label="잔차 정책 (필터 없음)" chip=0
    [ "${tag}" = B ] && { mode=learned; label="PS2 + 최종 필터"; chip=1; }
    echo "[kafa] $(stamp) ${name}"
    run ${K} gif --traj "${REC}/${tag}:${mode}" --episode "${ep}" --cam "${cam}" --label "${label}" --chip "${chip}" \
        --out "${OUT}/${name}.gif" 2>&1 | grep -E "^wrote|rror" | cut -c1-220 | sed "s/^/[kafa]   /"
}
gif intro_obs_A A "${OBS}" "${OUT}/cam_obs.json"
gif intro_obs_B B "${OBS}" "${OUT}/cam_obs.json"
gif intro_cha_A A "${CHA}" "${OUT}/cam_cha.json"
gif intro_cha_B B "${CHA}" "${OUT}/cam_cha.json"

echo "[kafa] $(stamp) intro slide images"
run ${K} col1 --out "${OUT}/col1_uav.png" 2>&1 | grep -E "^wrote|rror" | sed "s/^/[kafa]   /"
run ${K} col2 --traj "${REC}/A:none" --episode "${CHA}" --cam "${OUT}/cam_cha.json" --fit_b "${REC}/B" --out "${OUT}/col2_scene.png" \
    2>&1 | grep -E "^wrote|rror" | sed "s/^/[kafa]   /"
run ${K} col3 --traj "${REC}/A:none" --episodes "$(echo ${COL3} | tr ' ' ',')" --out "${OUT}" 2>&1 | grep -E "^wrote|rror" \
    | sed "s/^/[kafa]   /"

run ${K} report --out "${OUT}" --obs "${OBS}" --cha "${CHA}" | sed "s/^/[kafa]   /"
( cd "$(dirname "${OUT}")" && rm -f kafa_intro_1110.zip && zip -qr kafa_intro_1110.zip "$(basename "${OUT}")" -x "*/rec/*" )
echo "[kafa] $(stamp) done: ${OUT} and $(dirname "${OUT}")/kafa_intro_1110.zip"
