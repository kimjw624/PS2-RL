#!/usr/bin/env bash
# Strike task, final evaluation: every seed of the final pipeline (run_strike_final.sh) on N test scenes, all baselines and
# ablations, per-episode records for the report (scripts/report_strike.py). GPU by default (JAX_PLATFORMS from
# landing_env.sh); JAX_PLATFORMS=cpu works too (slow). README: docker_batch/README_strike.md ("Final pipeline").
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_strike_final_eval.sh > ~/ws_shared/PS2-RL/outputs/strike_final_eval.out 2>&1"
#
# Resumable: a run whose strike_filter_episodes.json exists is skipped. Never edit this file while it runs.
# Knobs (environment variables):
#   ROOT        pipeline output root                                    (outputs/strike_final)
#   SEEDS       seeds                                                   ("0 1 2")
#   N           test scenes per evaluation                              (1000)
#   SCENE_SEED  seed of the test scenes (not a training pool)           (9001)
#   TASK        task file: env block (arena, episode length) and the deployed filter (tracker.filter_guidance)
#                                                                       (docker_batch/configs/strike_final_ps2.json)
#   RUNS        evaluations to run                                      (all, see below)
#   GIFS        failure GIFs for ps2_final                              (4)
# Runs (output <ROOT>/eval/s<seed>/<run>/):
#   pursuit_static   straight-line pursuit: no filter, and the filter with static chaser balls (needs a ball-format backup)
#   pursuit_guid1    pursuit, guidance filter with one escape
#   pursuit_bcbf     pursuit, bCBF (minimal-correction QP) with the deployed filter as its safeguard
#   pursuit_final    pursuit, deployed filter
#   residual_final   residual striker trained without the filter: no filter, deployed filter
#   ps2_final        PS2 striker: no filter, deployed filter, the same filter with the hover-LQR backup
#   ps2_guid1        PS2 striker, one escape (no variants)
#   ps2_bcbf         PS2 striker, bCBF with the deployed filter as its safeguard
#   ps2_nocp         PS2 striker, deployed filter without the committed chaser prediction
#   ps2_nolatch      PS2 striker, deployed filter without the hover latch
#   ps2_static       PS2 striker, static chaser balls (needs a ball-format backup)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

ROOT="${ROOT:-outputs/strike_final}"
SEEDS="${SEEDS:-0 1 2}"
N="${N:-1000}"
SCENE_SEED="${SCENE_SEED:-9001}"
TASK="${TASK:-docker_batch/configs/strike_final_ps2.json}"
RUNS="${RUNS:-pursuit_static pursuit_guid1 pursuit_bcbf pursuit_final residual_final ps2_final ps2_guid1 ps2_bcbf ps2_nocp ps2_nolatch ps2_static}"
GIFS="${GIFS:-4}"
stamp() { date '+%F %T'; }

# the deployed filter (single source of truth: the PS2 task file) and its ablations
guid() {   # guid <variant>: final | one | nocp
    python - "${TASK}" "$1" <<'PY'
import json, sys
g = json.loads(json.load(open(sys.argv[1]))["tracker"].get("filter_guidance") or "{}")
if sys.argv[2] == "one":
    g.pop("escape_variants", None)
elif sys.argv[2] == "nocp":
    g["committed_prediction"] = False
print(json.dumps(g))
PY
}
G_FINAL="$(guid final)"; G_ONE="$(guid one)"; G_NOCP="$(guid nocp)"
echo "[strike-eval] $(stamp) root ${ROOT}; seeds ${SEEDS}; ${N} scenes (seed ${SCENE_SEED}); task ${TASK}"
echo "[strike-eval] deployed filter: ${G_FINAL}"

for seed in ${SEEDS}; do
    P1="${ROOT}/p1/td3_a0p2_seed${seed}"
    RES="${ROOT}/residual/striker_rs0p1_s${seed}"
    PS2="${ROOT}/ps2/striker_rs0p1_s${seed}"
    ball_backup=$(python -c "import json; print(int(json.load(open('${P1}/configs.json'))['strike'].get('chaser_set', 'ball') == 'ball'))" 2>/dev/null || echo 0)
    for run in ${RUNS}; do
        out="${ROOT}/eval/s${seed}/${run}"
        if [ -f "${out}/strike_filter_episodes.json" ]; then echo "[strike-eval] s${seed} ${run}: done before, skipped"; continue; fi
        nominal=(--nominal pursuit); modes="learned"; filt=(--chaser_model guidance --guidance "${G_FINAL}"); extra=(); gif=0
        case "${run}" in
            pursuit_static) modes="none learned"; filt=(--chaser_model static) ;;
            pursuit_guid1)  filt=(--chaser_model guidance --guidance "${G_ONE}") ;;
            pursuit_bcbf)   extra=(--filter_type bcbf) ;;
            pursuit_final)  ;;
            residual_final) nominal=(--nominal striker --striker_run "${RES}"); modes="none learned" ;;
            ps2_final)      nominal=(--nominal striker --striker_run "${PS2}"); modes="none learned lqr"; gif="${GIFS}" ;;
            ps2_guid1)      nominal=(--nominal striker --striker_run "${PS2}"); filt=(--chaser_model guidance --guidance "${G_ONE}") ;;
            ps2_bcbf)       nominal=(--nominal striker --striker_run "${PS2}"); extra=(--filter_type bcbf) ;;
            ps2_nocp)       nominal=(--nominal striker --striker_run "${PS2}"); filt=(--chaser_model guidance --guidance "${G_NOCP}") ;;
            ps2_nolatch)    nominal=(--nominal striker --striker_run "${PS2}"); extra=(--hover_latch 0) ;;
            ps2_static)     nominal=(--nominal striker --striker_run "${PS2}"); filt=(--chaser_model static) ;;
            *) echo "[strike-eval] unknown run ${run}"; continue ;;
        esac
        case "${run}" in *_static)
            if [ "${ball_backup}" != 1 ]; then echo "[strike-eval] s${seed} ${run}: the backup is not in the ball format, skipped"; continue; fi ;;
        esac
        for need in "${P1}/strike_backup_policy_actor.pkl"; do
            [ -f "${need}" ] || { echo "[strike-eval] s${seed} ${run}: missing ${need}, skipped"; continue 2; }
        done
        mkdir -p "${out}"
        echo "[strike-eval] $(stamp) s${seed} ${run}: modes '${modes}' (log ${out}.log)"
        if python scripts/eval_strike_filter.py --backup_run "${P1}" --task "${TASK}" "${nominal[@]}" "${filt[@]}" "${extra[@]}" \
                --modes "${modes}" --episodes "${N}" --scene_seed "${SCENE_SEED}" --n_show 0 --gif "${gif}" --gif_pick failures \
                --out "${out}" > "${out}.log" 2>&1; then
            grep -E "^\[(none|lqr|learned)" "${out}.log" | sed "s/^/[strike-eval]   /"
        else
            echo "[strike-eval] s${seed} ${run}: failed (see ${out}.log)"
        fi
    done
done
echo "[strike-eval] $(stamp) report"
python scripts/report_strike.py --root "${ROOT}" --seeds ${SEEDS} > "${ROOT}/report.log" 2>&1 && sed -n 1,40p "${ROOT}/report/results.md" | sed "s/^/[strike-eval]   /"
echo "[strike-eval] $(stamp) all done; tables and figures in ${ROOT}/report"
