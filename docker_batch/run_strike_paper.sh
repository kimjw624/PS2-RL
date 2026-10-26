#!/usr/bin/env bash
# Strike task: everything for the paper from the final pipeline's root - tables, figures, the numbers quoted in the text,
# and representative episodes (2-D and 3-D figures, detail plots, 2-D and 3-D GIFs). GPU by default (JAX_PLATFORMS from
# landing_env.sh); PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu EXACT=0 runs it on the CPU. README: docker_batch/README_strike.md.
#
#   docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && bash docker_batch/run_strike_paper.sh"
#   # tables and figures only (seconds):  STEPS=tables bash docker_batch/run_strike_paper.sh
#   # filter snapshots only (needs the replays): STEPS=snap SNAP_EPISODES="845 392" bash docker_batch/run_strike_paper.sh
#   # chosen episodes:                    EPISODES="845 473 392" bash docker_batch/run_strike_paper.sh
#   # trajectory figures of the paper's four configurations (잔차 정책, 잔차 + 최종 필터, PS2 + 단일 회피 필터,
#   # PS2 + 최종 필터) for its supporting episodes:
#   STEPS="episodes traj4" METHODS="residual ps2_one ps2" RES_MODES="none learned" \
#       EPISODES="827 576 146 914 972 271 493 845" bash docker_batch/run_strike_paper.sh
#
# Knobs (environment variables):
#   ROOT        pipeline output root (p1/, residual/, ps2/, eval/)                   (outputs/strike_final)
#   SEEDS       seeds in the tables                                                   ("0 1 2")
#   PAPER_LANG  figure and table language: ko | en                                    (ko)
#   SEED        seed whose models and evaluation the episodes come from               (0)
#   EPISODES    episode indices of that evaluation (empty: picked automatically)     ()
#   N_PICK      episodes picked automatically                                         (8)
#   N           scenes of the evaluation the indices refer to                         (1000)
#   SCENE_SEED  seed of those scenes                                                  (9001)
#   EXACT       1: one full evaluation per method, the chosen episodes recorded - exactly the evaluated episodes
#               (~6 min per method on the laptop GPU); 0: replay only the chosen episodes (fast; on another device
#               borderline episodes can end differently)                              (1)
#   METHODS     replays: ps2 (no filter + ours), ps2_static, ps2_bcbf, pursuit, residual, ps2_one (PS2 + one escape:
#               the final filter without the escape variants)        (ps2 ps2_static ps2_bcbf pursuit residual)
#   RES_MODES   modes of the residual replay: learned | "none learned" (also the residual policy alone)   (learned)
#   STEPS       tables episodes viz snap traj4 (traj4: the paper's trajectory figures, needs the residual replay with
#               RES_MODES "none learned", ps2_one and ps2)            (tables episodes viz snap)
#   TRAJ_GRID   traj4: episodes of the main grid figure (grid_main; grid_all has every episode)  (first 3 of EPISODES)
#   TRAJ_SINGLE traj4: episodes drawn alone (2 x 2, one row, signals) (first of EPISODES)
#   TRAJ_PAIRS  traj4: pair figures "NAME:KEY_A,KEY_B:EPISODES:SIGNALS" (keys res res_f one ours), space separated
#               ("filter:res,res_f:<2nd>:clear,hc backup:one,ours:<4th>:hc,variant phase2:res_f,ours:<5th>:dist,eta")
#   TRAJ_SNAP   traj4: episodes with a paper-style filter snapshot of PS2 + 최종 필터 (last of EPISODES)
#   SNAP_EPISODES  episodes with filter snapshots (the filter's check redrawn at chosen steps)  (first 3 of EPISODES)
#   OUT         output folder                                                         (<ROOT>/paper)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

ROOT="${ROOT:-outputs/strike_final}"
SEEDS="${SEEDS:-0 1 2}"
PAPER_LANG="${PAPER_LANG:-ko}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-}"
N_PICK="${N_PICK:-8}"
N="${N:-1000}"
SCENE_SEED="${SCENE_SEED:-9001}"
EXACT="${EXACT:-1}"
METHODS="${METHODS:-ps2 ps2_static ps2_bcbf pursuit residual}"
STEPS="${STEPS:-tables episodes viz snap}"
SNAP_EPISODES="${SNAP_EPISODES:-}"
RES_MODES="${RES_MODES:-learned}"
TRAJ_GRID="${TRAJ_GRID:-}"
TRAJ_SINGLE="${TRAJ_SINGLE:-}"
TRAJ_PAIRS="${TRAJ_PAIRS:-}"
TRAJ_SNAP="${TRAJ_SNAP:-}"
OUT="${OUT:-${ROOT}/paper}"
TASK="docker_batch/configs/strike_final_ps2.json"
stamp() { date '+%F %T'; }
mkdir -p "${OUT}"
echo "[strike-paper] $(stamp) root ${ROOT}; seeds ${SEEDS}; language ${PAPER_LANG}; episodes from seed ${SEED}; out ${OUT}"

case " ${STEPS} " in *" tables "*)
    echo "[strike-paper] $(stamp) tables, figures, numbers"
    python scripts/strike_paper.py --root "${ROOT}" --seeds ${SEEDS} --lang "${PAPER_LANG}" --pick_seed "${SEED}" \
        --out "${OUT}" > "${OUT}/tables.log" 2>&1 || { echo "[strike-paper] tables failed (see ${OUT}/tables.log)"; exit 1; }
    tail -1 "${OUT}/tables.log" | sed "s/^/[strike-paper]   /" ;;
esac

if [ -z "${EPISODES}" ] && [ -f "${OUT}/episode_picks.json" ]; then
    EPISODES=$(python - "${OUT}/episode_picks.json" "${N_PICK}" <<'PY'
import json, sys
picks = json.load(open(sys.argv[1]))["picks"]
order = ["ps2_contact", "filter_saves_obstacle", "filter_saves_chaser", "close_chaser_evasion", "ps2_beats_pursuit",
         "static_ball_fails", "bcbf_fails", "ps2_timeup", "typical", "ps2_beats_residual"]
eps = []
for k in order + [k for k in picks if k not in order]:
    for e in picks.get(k, []):
        if e not in eps:
            eps.append(e)
print(" ".join(map(str, eps[: int(sys.argv[2])])))
PY
)
fi
[ -n "${EPISODES}" ] || { echo "[strike-paper] no episodes (run STEPS=tables first or set EPISODES)"; exit 0; }
echo "[strike-paper] episodes: ${EPISODES}"

P1="${ROOT}/p1/td3_a0p2_seed${SEED}"
PS2="${ROOT}/ps2/striker_rs0p1_s${SEED}"
RES="${ROOT}/residual/striker_rs0p1_s${SEED}"
RD="${OUT}/replays/s${SEED}"
case " ${STEPS} " in *" episodes "*|*" snap "*|*" traj4 "*)
    G="$(python - "${TASK}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["tracker"].get("filter_guidance") or "{}")
PY
)"
    # the chaser this seed was trained with (as run_strike_final_eval.sh)
    ENV_S=$(python - "${PS2}" "${RES}" <<'PY' 2>/dev/null
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
    envarg=(); [ -n "${ENV_S}" ] && envarg=(--env "${ENV_S}")
    # PS2 + 단일 회피 필터: the final guidance without the escape variants (as run_strike_final_eval.sh's G_ONE)
    G_ONE="$(python - "${G}" <<'PY'
import json, sys
g = json.loads(sys.argv[1])
g.pop("escape_variants", None)
print(json.dumps(g))
PY
)" ;;
esac
case " ${STEPS} " in *" episodes "*)
    pick=(--only_episodes); [ "${EXACT}" = 1 ] && pick=(--record_episodes)
    mkdir -p "${RD}"
    replay() {   # replay <name> <modes> <args...>
        local name=$1 modes=$2; shift 2
        echo "[strike-paper] $(stamp) replay ${name} (exact ${EXACT})"
        # shellcheck disable=SC2086
        if python scripts/eval_strike_filter.py --backup_run "${P1}" --task "${TASK}" "${envarg[@]}" --chaser_model guidance \
                --guidance "${G}" "$@" --modes "${modes}" --episodes "${N}" --scene_seed "${SCENE_SEED}" "${pick[@]}" ${EPISODES} \
                --save_traj 1 --save_episodes 0 --n_show 0 --gif 0 --out "${RD}/${name}" > "${RD}/${name}.log" 2>&1; then
            grep -E "^\[(none|learned)" "${RD}/${name}.log" | sed "s/^/[strike-paper]   /"
        else
            echo "[strike-paper] replay ${name} failed (see ${RD}/${name}.log)"; exit 1
        fi
    }
    for m in ${METHODS}; do
        case "${m}" in
            ps2)        replay ps2 "none learned" --nominal striker --striker_run "${PS2}" ;;
            ps2_static) replay ps2_static learned --nominal striker --striker_run "${PS2}" --chaser_model static ;;
            ps2_bcbf)   replay ps2_bcbf learned --nominal striker --striker_run "${PS2}" --filter_type bcbf ;;
            pursuit)    replay pursuit learned --nominal pursuit ;;
            residual)   replay residual "${RES_MODES}" --nominal striker --striker_run "${RES}" ;;
            # (the later --guidance overrides the replay's own)
            ps2_one)    replay ps2_one learned --nominal striker --striker_run "${PS2}" --guidance "${G_ONE}" ;;
            *) echo "[strike-paper] unknown method ${m}"; exit 1 ;;
        esac
    done ;;
esac

case " ${STEPS} " in *" viz "*)
    if [ "${PAPER_LANG}" = ko ]; then
        L_none="PS2 (필터 없음)"; L_ours="PS2 + 최종 필터"; L_static="PS2 + 구형 근사"; L_bcbf="PS2 + bCBF"
        L_pur="기준 접근 제어 + 최종 필터"; L_res="잔차 정책 + 최종 필터"
    else
        L_none="PS2, no filter"; L_ours="PS2 + ours"; L_static="PS2 + static ball"; L_bcbf="PS2 + bCBF"
        L_pur="Pursuit + ours"; L_res="Residual SAC + ours"
    fi
    panels=()
    for m in ${METHODS}; do
        case "${m}" in
            ps2)        panels+=(--panel "${RD}/ps2:learned:${L_ours}:ps2_ours" --panel "${RD}/ps2:none:${L_none}:ps2_none") ;;
            ps2_static) panels+=(--panel "${RD}/ps2_static:learned:${L_static}:ps2_static") ;;
            ps2_bcbf)   panels+=(--panel "${RD}/ps2_bcbf:learned:${L_bcbf}:ps2_bcbf") ;;
            pursuit)    panels+=(--panel "${RD}/pursuit:learned:${L_pur}:pursuit_ours") ;;
            residual)   panels+=(--panel "${RD}/residual:learned:${L_res}:residual_ours") ;;
        esac
    done
    names=$(python - "${OUT}/episode_picks.json" "${PAPER_LANG}" <<'PY' 2>/dev/null
import json, sys
cap = {"ko": {"filter_saves_obstacle": "필터가 장애물 충돌 방지", "filter_saves_chaser": "필터가 추격자 접촉 방지",
              "close_chaser_evasion": "추격자 근접 회피", "ps2_contact": "접촉 사례", "ps2_timeup": "시간초과 사례",
              "typical": "대표 성공", "ps2_beats_pursuit": "기준 제어는 시간초과", "ps2_beats_residual": "잔차 정책은 시간초과",
              "static_ball_fails": "구형 근사 실패", "bcbf_fails": "bCBF 실패"},
       "en": {"filter_saves_obstacle": "filter prevents an obstacle crash", "filter_saves_chaser": "filter prevents a chaser hit",
              "close_chaser_evasion": "close chaser evasion", "ps2_contact": "the contact", "ps2_timeup": "time-up",
              "typical": "typical strike", "ps2_beats_pursuit": "pursuit times out", "ps2_beats_residual": "residual times out",
              "static_ball_fails": "static ball fails", "bcbf_fails": "bCBF fails"}}[sys.argv[2]]
out = {}
for k, eps in json.load(open(sys.argv[1]))["picks"].items():
    for e in eps:
        out.setdefault(str(e), cap.get(k, k))
print(json.dumps(out, ensure_ascii=False))
PY
)
    [ -n "${names}" ] || names="{}"
    echo "[strike-paper] $(stamp) episode figures and GIFs"
    # shellcheck disable=SC2086
    python scripts/strike_episode_viz.py "${panels[@]}" --episodes ${EPISODES} --names "${names}" --main ps2_ours \
        --gif3d ps2_ours ps2_none pursuit_ours --overview_main ps2_ours ps2_none residual_ours \
        --lang "${PAPER_LANG}" --out "${OUT}/episodes" > "${OUT}/episodes.log" 2>&1 \
        || { echo "[strike-paper] episode figures failed (see ${OUT}/episodes.log)"; exit 1; }
    grep -E "^\[episode" "${OUT}/episodes.log" | sed "s/^/[strike-paper]   /" ;;
esac
case " ${STEPS} " in *" snap "*)
    # the filter's check redrawn at chosen steps (first cut of the nominal, committed plan, lost certificate)
    [ -n "${SNAP_EPISODES}" ] || SNAP_EPISODES=$(echo ${EPISODES} | cut -d' ' -f1-3)
    if [ -f "${RD}/ps2/strike_filter_traj.npz" ]; then
        echo "[strike-paper] $(stamp) filter snapshots: episodes ${SNAP_EPISODES}"
        for e in ${SNAP_EPISODES}; do
            python scripts/strike_filter_snapshot.py --replay "${RD}/ps2" --mode learned --episode "${e}" --backup_run "${P1}" \
                --task "${TASK}" "${envarg[@]}" --guidance "${G}" --lang "${PAPER_LANG}" --out "${OUT}/snapshots" \
                >> "${OUT}/snapshots.log" 2>&1 || echo "[strike-paper] snapshot of episode ${e} failed (see ${OUT}/snapshots.log)"
        done
        grep -E "^\[snapshot\]" "${OUT}/snapshots.log" | sed "s/^/[strike-paper]   /"
    else
        echo "[strike-paper] no replay for the snapshots (run the episodes step first)"
    fi ;;
esac
case " ${STEPS} " in *" traj4 "*)
    # the paper's trajectory figures: the four compared configurations, the paper's terms and sizes
    need=("${RD}/residual" "${RD}/ps2_one" "${RD}/ps2")
    for d in "${need[@]}"; do
        [ -f "${d}/strike_filter_traj.npz" ] || { echo "[strike-paper] traj4 needs ${d} (METHODS=\"residual ps2_one ps2\")"; exit 1; }
    done
    set -- ${EPISODES}
    [ -n "${TRAJ_GRID}" ] || TRAJ_GRID=$(echo ${EPISODES} | cut -d' ' -f1-3)
    [ -n "${TRAJ_SINGLE}" ] || TRAJ_SINGLE="$1"
    [ -n "${TRAJ_PAIRS}" ] || TRAJ_PAIRS="filter:res,res_f:${2:-$1}:clear,hc backup:one,ours:${4:-$1}:hc,variant phase2:res_f,ours:${5:-$1}:dist,eta"
    [ -n "${TRAJ_SNAP}" ] || TRAJ_SNAP="${!#}"
    if [ "${PAPER_LANG}" = ko ]; then
        C1="잔차 정책"; C2="잔차 + 최종 필터"; C3="PS2 + 단일 회피 필터"; C4="PS2 + 최종 필터"
    else
        C1="Residual policy"; C2="Residual + final filter"; C3="PS2 + one-escape filter"; C4="PS2 + final filter"
    fi
    pairs=(); for pr in ${TRAJ_PAIRS}; do pairs+=(--pair "${pr}"); done
    echo "[strike-paper] $(stamp) trajectory figures (four configurations): grid ${TRAJ_GRID}; alone ${TRAJ_SINGLE}"
    # shellcheck disable=SC2086
    python scripts/strike_paper_traj.py --config "${RD}/residual:none:${C1}:res" --config "${RD}/residual:learned:${C2}:res_f" \
        --config "${RD}/ps2_one:learned:${C3}:one" --config "${RD}/ps2:learned:${C4}:ours" \
        --grid_spec "main:$(echo ${TRAJ_GRID} | tr ' ' ,)" --grid_spec "all:$(echo ${EPISODES} | tr ' ' ,)" \
        --episode ${TRAJ_SINGLE} "${pairs[@]}" --terms "${PAPER_LANG}" --out "${OUT}/traj4" > "${OUT}/traj4.log" 2>&1 \
        || { echo "[strike-paper] trajectory figures failed (see ${OUT}/traj4.log)"; exit 1; }
    for e in ${TRAJ_SNAP}; do
        python scripts/strike_filter_snapshot.py --replay "${RD}/ps2" --mode learned --episode "${e}" --backup_run "${P1}" \
            --task "${TASK}" "${envarg[@]}" --guidance "${G}" --lang "${PAPER_LANG}" --terms paper --out "${OUT}/traj4/snapshots" \
            >> "${OUT}/traj4.log" 2>&1 || echo "[strike-paper] snapshot of episode ${e} failed (see ${OUT}/traj4.log)"
    done
    tail -1 "${OUT}/traj4.log" | sed "s/^/[strike-paper]   /" ;;
esac
echo "[strike-paper] $(stamp) done: ${OUT}/tables, ${OUT}/figures, ${OUT}/numbers.md, ${OUT}/episodes, ${OUT}/snapshots"
