#!/usr/bin/env bash
# Strike task (reduced arena): the filter improvements on the full test set, against the report run's own filter.
# Each variant evaluates the report run's strikers (seed by seed) on the N test scenes with the filter options given
# below and writes, per run, the evaluation table, the per-episode records and the command-smoothness statistics
# (filter active share, kept/changed switches per second, sum |d lambda| per second, ||du|| relative to the nominal
# command, backup switches per second). A summary table compares them at the end. GPU: about 3-6 min per variant and seed.
#
#   docker exec -it px4sitl bash -c "cd ~/ws_shared/PS2-RL && bash docker_batch/run_strike_improve.sh"
#   # some variants, one seed:  VARIANTS="ps2_ref ps2_ramp" SEEDS=0 bash docker_batch/run_strike_improve.sh
#   # summary only:             STEPS=summary bash docker_batch/run_strike_improve.sh
#
# Knobs (environment variables):
#   ROOT       the report run (p1/, residual/, ps2/)                                   (outputs/strike_final)
#   SEEDS      seeds                                                                   ("0 1")
#   N          test scenes                                                             (1000)
#   SCENE_SEED seed of the test scenes                                                 (9001)
#   V2_BACKUP  the wall-aware Phase-I backup (moving chasers + arena walls; full-arena pipeline)
#                                                                                      (outputs/strike_phase1_full/td3_a0p2_seed0)
#   VARIANTS   variants to run (names from DEFS)                                       (all of DEFS)
#   DEFS       variant table, one per line: name|striker (ps2|residual)|backup (p1|v2)|guidance overrides (JSON)
#              [|extra eval_strike_filter.py arguments, e.g. the nominal's detour: --pursuit_detour 3 --pursuit_search 1]
#   STEPS      eval summary                                                            ("eval summary")
#   OUT        output folder                                                           (outputs/strike_improve)
# Resumable: a run with strike_filter_episodes.json is skipped. Never edit this file while it runs.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

ROOT="${ROOT:-outputs/strike_final}"
SEEDS="${SEEDS:-0 1}"
N="${N:-1000}"
SCENE_SEED="${SCENE_SEED:-9001}"
V2_BACKUP="${V2_BACKUP:-outputs/strike_phase1_full/td3_a0p2_seed0}"
STEPS="${STEPS:-eval summary}"
OUT="${OUT:-outputs/strike_improve}"
TASK="docker_batch/configs/strike_final_ps2.json"
# The nominal's detour (eval options, the 5th field): --pursuit_detour 3 passes chasers at 3 m; --pursuit_search 1 adds the
# look-ahead heading search around obstacles and walls; --pursuit_search_heading_weight 0.5 keeps it to one side;
# --pursuit_search_final 3 flies straight within 3 m of the target; --pursuit_search_wall_margin 2 keeps 2 m from the walls.
# Results of the first GPU round (11/01): results/strike_reduced/improvements_gpu/README.md.
SEARCH="--pursuit_detour 3 --pursuit_search 1"
FW="--pursuit_search_final 3 --pursuit_search_wall_margin 2"
# early escapes: two more backup variants (3.5 v_c, +-60 deg) whose reflex starts below h_c = 6 m (the others at 2 m)
EARLY='"escape_variants": [[2.0, 60], [2.0, -60], [3.5, 0], [3.5, 60], [3.5, -60], [3.5, 60, 6], [3.5, -60, 6]]'
DEFAULT_DEFS="ps2_ref|ps2|p1|{}
ps2_ramp20|ps2|p1|{\"lam_rise_max\": 0.2}
ps2_v2|ps2|v2|{\"learned_alone\": true}
ps2_detour|ps2|p1|{}|--pursuit_detour 3
ps2_v2detour|ps2|v2|{\"learned_alone\": true}|--pursuit_detour 3
ps2_search|ps2|p1|{}|${SEARCH}
ps2_searchh|ps2|p1|{}|${SEARCH} --pursuit_search_heading_weight 0.5
ps2_v2searchh|ps2|v2|{\"learned_alone\": true}|${SEARCH} --pursuit_search_heading_weight 0.5
residual_ref|residual|p1|{}
residual_search|residual|p1|{}|${SEARCH}
residual_searchh|residual|p1|{}|${SEARCH} --pursuit_search_heading_weight 0.5
residual_v2searchh|residual|v2|{\"learned_alone\": true}|${SEARCH} --pursuit_search_heading_weight 0.5
ps2_searchfw|ps2|p1|{}|${SEARCH} ${FW}
residual_searchfw|residual|p1|{}|${SEARCH} ${FW}
ps2_search_ramp20|ps2|p1|{\"lam_rise_max\": 0.2}|${SEARCH}
residual_search_ramp20|residual|p1|{\"lam_rise_max\": 0.2}|${SEARCH}
ps2_search_early|ps2|p1|{${EARLY}}|${SEARCH}
residual_search_early|residual|p1|{${EARLY}}|${SEARCH}
ps2_search_early_zs|ps2|p1|{${EARLY}, \"fallback_zero_slack\": true}|${SEARCH}
residual_search_early_zs|residual|p1|{${EARLY}, \"fallback_zero_slack\": true}|${SEARCH}"
DEFS="${DEFS:-${DEFAULT_DEFS}}"
VARIANTS="${VARIANTS:-$(printf '%s\n' "${DEFS}" | cut -d'|' -f1 | tr '\n' ' ')}"
stamp() { date '+%F %T'; }
mkdir -p "${OUT}"
echo "[strike-improve] $(stamp) root ${ROOT}; seeds ${SEEDS}; N ${N}; variants ${VARIANTS}; out ${OUT}"

case " ${STEPS} " in *" eval "*)
    for v in ${VARIANTS}; do
        line=$(printf '%s\n' "${DEFS}" | awk -F'|' -v n="${v}" '$1 == n')
        [ -n "${line}" ] || { echo "[strike-improve] unknown variant ${v}"; exit 1; }
        striker=$(echo "${line}" | cut -d'|' -f2); backup=$(echo "${line}" | cut -d'|' -f3); over=$(echo "${line}" | cut -d'|' -f4)
        extra=$(echo "${line}" | awk -F'|' '{print (NF >= 5) ? $5 : ""}')
        for s in ${SEEDS}; do
            out="${OUT}/s${s}/${v}"
            if [ -f "${out}/strike_filter_episodes.json" ]; then echo "[strike-improve] s${s} ${v}: done before, skipped"; continue; fi
            run="${ROOT}/${striker}/striker_rs0p1_s${s}"
            case "${backup}" in p1) bk="${ROOT}/p1/td3_a0p2_seed${s}" ;; v2) bk="${V2_BACKUP}" ;;
                *) echo "[strike-improve] ${v}: backup must be p1 or v2"; exit 1 ;; esac
            for need in "${run}/best_weights.pkl" "${bk}/strike_backup_policy_actor.pkl"; do
                [ -f "${need}" ] || { echo "[strike-improve] ${v} s${s}: missing ${need}"; continue 2; }
            done
            # the deployed filter (the PS2 task file) with the variant's options; the chaser the striker was trained with
            G=$(python - "${TASK}" "${over}" <<'PY'
import json, sys
g = json.loads(json.load(open(sys.argv[1]))["tracker"].get("filter_guidance") or "{}")
g.update(json.loads(sys.argv[2] or "{}"))
print(json.dumps(g))
PY
)
            ENV_S=$(python - "${run}" <<'PY' 2>/dev/null
import json, sys
from pathlib import Path
from ps2rl.envs.strike_env import StrikeEnvConfig
d = StrikeEnvConfig()
e = json.loads((Path(sys.argv[1]) / "configs.json").read_text()).get("env", {})
print(json.dumps({k: e.get(k, getattr(d, k)) for k in ("actual_chaser_radius", "actual_chaser_speed", "chaser_accel")}))
PY
)
            envarg=(); [ -n "${ENV_S}" ] && envarg=(--env "${ENV_S}")
            modes="learned"; case "${v}" in *_ref) modes="none learned" ;; esac
            mkdir -p "${out}"
            echo "[strike-improve] $(stamp) s${s} ${v}: ${striker}, backup ${backup}, options ${over} ${extra}"
            if python scripts/eval_strike_filter.py --backup_run "${bk}" --task "${TASK}" "${envarg[@]}" --chaser_model guidance \
                    --guidance "${G}" --nominal striker --striker_run "${run}" --modes "${modes}" --episodes "${N}" \
                    --scene_seed "${SCENE_SEED}" --n_show 0 --gif 0 ${extra} --out "${out}" > "${out}.log" 2>&1; then
                grep -E "^\[(none|learned)" "${out}.log" | sed "s/^/[strike-improve]   /"
            else
                echo "[strike-improve] s${s} ${v} failed (see ${out}.log)"
            fi
        done
    done ;;
esac

case " ${STEPS} " in *" summary "*)
    python - "${OUT}" "${SEEDS}" <<'PY'
import json, sys
from pathlib import Path
out, seeds = Path(sys.argv[1]), sys.argv[2].split()
rows = ["| variant | seeds × scenes | strike % | contact (count) | time-up % | strike time [s] | lost-cert episodes % | "
        "filter active | switches/s | Σ|Δλ|/s | ‖Δu‖ / nominal | backup switches/s |", "|" + "---|" * 12]
names = sorted({p.name for s in seeds for p in (out / f"s{s}").glob("*") if p.is_dir()})
for v in names:
    rs = []
    for s in seeds:
        f = out / f"s{s}" / v / "strike_filter_table.json"
        if f.exists():
            rs.append(json.loads(f.read_text())["results"]["learned"])
    if not rs:
        continue
    n = len(rs)
    eps = [json.loads((out / f"s{s}" / v / "strike_filter_table.json").read_text())["episodes"] for s in seeds
           if (out / f"s{s}" / v / "strike_filter_table.json").exists()]
    mean = lambda k: sum(r[k] for r in rs) / n
    contacts = sum(round(r["contact_rate"] * e) for r, e in zip(rs, eps))
    rows.append(f"| {v} | {n} × {eps[0]} | {100 * mean('strike_rate'):.1f} | {contacts} | {100 * mean('timeout_rate'):.1f} | "
                f"{mean('strike_time_mean'):.2f} | {100 * mean('episodes_with_no_certificate'):.1f} | "
                f"{mean('active_fraction'):.3f} | {mean('toggles_per_s'):.2f} | {mean('dlam_per_s'):.2f} | "
                f"{mean('du_ratio'):.2f} | {mean('variant_switches_per_s'):.2f} |")
txt = "\n".join(rows)
(out / "summary.md").write_text(txt + "\n")
print(txt)
PY
    ;;
esac
echo "[strike-improve] $(stamp) done: ${OUT}/summary.md"
