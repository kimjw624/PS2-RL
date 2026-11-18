#!/usr/bin/env bash
# Nominal bCBF vs UE-bCBF under stronger, constant disturbances (outside-cone task), on the CPU.
#
#   PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu bash docker_batch/run_cone_obstacle_disturbance_sweep.sh
#
# For every direction in DIRS and amplitude scale in SCALES (|d| = scale * delta_d, held constant for the whole
# episode; scale 1 = the strongest push within the filter's design bound), evaluates
#   the stage-1 tracker through the nominal bCBF and through the UE-bCBF   (same policy, same filter geometry)
#   the Phase II policy through the UE-bCBF
# on the same starts, then plots unsafe rate / smallest barrier value vs |d| (OUT/sweep.png, sweep.md).
# Knobs: OUT (outputs/cone_obstacle_phase2_ue/dist_sweep), RUNS (outputs/cone_obstacle_phase2_ue), TRACKER (from
# RUNS/vanilla_choice.json), PS2 (cone_ue_ps2_s0), SCALES ("1 2 4"), DIRS ("rand:  wall:1,0,1"), EPISODES (64).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"
export JAX_ENABLE_X64=1
RUNS="${RUNS:-outputs/cone_obstacle_phase2_ue}"
OUT="${OUT:-${RUNS}/dist_sweep}"
TRACKER="${TRACKER:-$(python -c "import json;print(json.load(open('${RUNS}/vanilla_choice.json'))['tag'])")}"
PS2="${PS2:-cone_ue_ps2_s0}"
SCALES="${SCALES:-1 2 4}"
DIRS="${DIRS:-rand: wall:1,0,1}"
EPISODES="${EPISODES:-64}"
BACKUP_JSON="${BACKUP_JSON:-outputs/cone_obstacle_phase1/compare/chosen_backup.json}"
B_ARGS="$(python -c "import json;d=json.load(open('${BACKUP_JSON}'));print('--backup', d['backup'], *(['--backup_path', d['backup_path']] if d['backup_path'] else []))")"
mkdir -p "${OUT}"
for spec in ${DIRS}; do
    dname="${spec%%:*}"; dvec="${spec#*:}"
    for sc in ${SCALES}; do
        for job in "tracker:${TRACKER}:nominal,ue" "ps2:${PS2}:ue"; do
            pol="${job%%:*}"; rest="${job#*:}"; run="${rest%%:*}"; modes="${rest#*:}"
            case_dir="${OUT}/${pol}__constant__${dname}__a${sc}"
            [ -f "${case_dir}/eval.json" ] && { echo "[sweep] ${case_dir} done"; continue; }
            echo "[sweep] $(date '+%T') ${pol} (${modes}) constant |d| = ${sc} x delta_d, direction ${dname} ${dvec}"
            extra=""
            [ "${pol}" = "tracker" ] && extra="${B_ARGS}"
            # shellcheck disable=SC2086
            python scripts/eval_phase2_cone_obstacle_ue.py --run "${RUNS}/${run}" --modes "${modes}" --episodes "${EPISODES}" \
                --disturbance_mode constant --amp_scale "${sc}" --disturbance_dir "${dvec}" ${extra} --out "${case_dir}" \
                2>&1 | grep -E "^\[(ue|nominal|none|cone)" | tee -a "${OUT}/sweep.log"
        done
    done
done
python scripts/plot_cone_obstacle_disturbance_sweep.py --root "${OUT}"
