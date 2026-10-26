#!/usr/bin/env bash
# Strike task: specific episodes of the final evaluation, replayed under the main methods and drawn side by side
# (scripts/plot_strike_episodes.py): PS2 without the filter, PS2 / pursuit / residual striker with the deployed filter,
# PS2 with the static chaser ball and with the bCBF. Same scenes and perturbations as run_strike_final_eval.sh (N scenes,
# SCENE_SEED), so episode i here is episode i of <ROOT>/eval/s<seed>. GPU by default (JAX_PLATFORMS from landing_env.sh).
#
#   docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && EPISODES='6 20 845' bash docker_batch/run_strike_episodes.sh"
#
# Knobs (environment variables):
#   ROOT        pipeline output root with p1/, residual/, ps2/                  (outputs/strike_final)
#   SEED        seed of the models                                              (0)
#   EPISODES    episode indices of the evaluation                               (required)
#   N           scenes of the evaluation the indices refer to                   (1000)
#   SCENE_SEED  seed of those scenes                                            (9001)
#   METHODS     replays: ps2 (no filter + ours), ps2_static, ps2_bcbf, pursuit, residual   (all)
#   EXACT       1: simulate all N episodes and record the chosen ones - bit for bit the episodes of the evaluation on the
#               same device (a full evaluation per method, ~6 min each on the laptop GPU); 0: replay only the chosen
#               episodes (fast; on another device or batch size the late outcomes of borderline episodes can differ)   (1)
#   TASK        task file: env block and the deployed filter                    (docker_batch/configs/strike_final_ps2.json)
#   ENV         env overrides (JSON)                                            (the chaser the seed's PS2 run was trained with)
#   OUT         output folder                                                   (<ROOT>/episodes/s<seed>)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

ROOT="${ROOT:-outputs/strike_final}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:?set EPISODES, e.g. EPISODES=\"6 20 845\"}"
N="${N:-1000}"
SCENE_SEED="${SCENE_SEED:-9001}"
METHODS="${METHODS:-ps2 ps2_static ps2_bcbf pursuit residual}"
TASK="${TASK:-docker_batch/configs/strike_final_ps2.json}"
ENV="${ENV:-}"
EXACT="${EXACT:-1}"
OUT="${OUT:-${ROOT}/episodes/s${SEED}}"
stamp() { date '+%F %T'; }

P1="${ROOT}/p1/td3_a0p2_seed${SEED}"
PS2="${ROOT}/ps2/striker_rs0p1_s${SEED}"
RES="${ROOT}/residual/striker_rs0p1_s${SEED}"
G="$(python - "${TASK}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["tracker"].get("filter_guidance") or "{}")
PY
)"
if [ -z "${ENV}" ]; then          # the chaser this seed was trained with (as run_strike_final_eval.sh)
    ENV=$(python - "${PS2}" "${RES}" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
from ps2rl.envs.strike_env import StrikeEnvConfig
d = StrikeEnvConfig()
for run in sys.argv[1:]:
    p = Path(run) / "configs.json"
    if p.exists():
        e = json.loads(p.read_text()).get("env", {})
        print(json.dumps({k: e.get(k, getattr(d, k)) for k in ("actual_chaser_radius", "actual_chaser_speed", "chaser_accel")}))
        break
PY
)
fi
envarg=(); [ -n "${ENV}" ] && envarg=(--env "${ENV}")
mkdir -p "${OUT}"
echo "[strike-episodes] $(stamp) seed ${SEED} (${ROOT}); episodes ${EPISODES} of ${N} (scene seed ${SCENE_SEED}); chaser ${ENV:-as in the task file}; exact ${EXACT}"

replay() {   # replay <name> <modes> <args...>
    local name=$1 modes=$2; shift 2
    echo "[strike-episodes] $(stamp) ${name}"
    local pick=(--only_episodes); [ "${EXACT}" = 1 ] && pick=(--record_episodes)
    # shellcheck disable=SC2086
    if python scripts/eval_strike_filter.py --backup_run "${P1}" --task "${TASK}" "${envarg[@]}" --chaser_model guidance \
            --guidance "${G}" "$@" --modes "${modes}" --episodes "${N}" --scene_seed "${SCENE_SEED}" "${pick[@]}" ${EPISODES} \
            --save_traj 1 --n_show 0 --gif 0 --out "${OUT}/${name}" > "${OUT}/${name}.log" 2>&1; then
        grep -E "^\[(none|learned)" "${OUT}/${name}.log" | sed "s/^/[strike-episodes]   /"
    else
        echo "[strike-episodes] ${name} failed (see ${OUT}/${name}.log)"; exit 1
    fi
}
panels=()
for m in ${METHODS}; do
    case "${m}" in
        ps2)        replay ps2 "none learned" --nominal striker --striker_run "${PS2}"
                    panels+=(--panel "${OUT}/ps2:none:PS2, no filter" --panel "${OUT}/ps2:learned:PS2 + ours") ;;
        ps2_static) replay ps2_static learned --nominal striker --striker_run "${PS2}" --chaser_model static
                    panels+=(--panel "${OUT}/ps2_static:learned:PS2 + static chaser ball") ;;
        ps2_bcbf)   replay ps2_bcbf learned --nominal striker --striker_run "${PS2}" --filter_type bcbf
                    panels+=(--panel "${OUT}/ps2_bcbf:learned:PS2 + bCBF") ;;
        pursuit)    replay pursuit learned --nominal pursuit
                    panels+=(--panel "${OUT}/pursuit:learned:Pursuit + ours") ;;
        residual)   replay residual learned --nominal striker --striker_run "${RES}"
                    panels+=(--panel "${OUT}/residual:learned:Residual SAC (no PS2) + ours") ;;
        *) echo "[strike-episodes] unknown method ${m}"; exit 1 ;;
    esac
done
python scripts/plot_strike_episodes.py "${panels[@]}" --cols 3 --pdf 1 --out "${OUT}" | sed "s/^/[strike-episodes]   /"
echo "[strike-episodes] $(stamp) done: ${OUT}/episode<i>.png / .pdf / .gif and ${OUT}/episodes.md"
