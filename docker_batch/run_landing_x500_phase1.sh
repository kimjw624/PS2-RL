#!/usr/bin/env bash
# Phase I (safe-arrival backup) for another vehicle, by default the stock Gazebo x500.
# The landing defaults assume 4 g of thrust and 18 rad/s body rates; this batch re-derives the
# action box, the hover LQR and the base set from the vehicle's constants and trains with them.
# Nothing of the original design is replaced: it only passes a different config file.
#
#   design       scripts/design_landing_base_set.py         vehicle constants -> ${CONFIG} (kept if it exists)
#   certificate  scripts/certify_landing_base_set.py        full check of that base set; stops here if it fails
#   baseline     scripts/evaluate_landing_backup_regions.py recoverability of the hover LQR alone (no training)
#   B C D F      run_landing_floor_overnight.sh             train (GPU), self-check, reach tests, report
#   regions      scripts/evaluate_landing_backup_regions.py each trained backup against the hover LQR alone
#
#     bash ~/ws_shared/PS2-RL/docker_batch/run_landing_x500_phase1.sh
#
# From the HOST, with the log on screen:
#     docker exec -it px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_x500_phase1.sh 2>&1 \
#         | tee ~/ws_shared/PS2-RL/outputs/landing_x500.out"
#
# Knobs (environment variables):
#   OUT=outputs/landing_x500
#   VEHICLE=docker_batch/configs/vehicles/gz_x500.json          vehicle constants
#   CONFIG=docker_batch/configs/landing_x500_cone45_r0p3.json   Phase-I config (written by the design stage)
#   REDESIGN=0                                                  1: derive ${CONFIG} again even if it exists
#   REFERENCE_DESIGN=checkpoints/landing_phase1/floor_rec10_td3_seed0/configs.json   design carried over (cone, floor, ...)
#   TASK_REFS="ps2rl/envs/assets/quadrotor_landing_b*_reference.npz"   references the design region must cover
#   DESIGN_ARGS=""                                              extra flags for the design script, e.g.
#                                                               "--thrust_reserve 0.15 --omega_z_max 0.5 --safety_factor 0.7"
#   NAME=x500_rec10                                             run name prefix (-> ${NAME}_${BACKBONE}_seed<k>)
#   START_REF=ps2rl/envs/assets/quadrotor_landing_b2_m30_reference.npz   "reference start" of the reach tests
#   STAGES="B C D F"                                            stages of run_landing_floor_overnight.sh to run
#   SETTINGS="a10_f20:...;a4:...;a10_f20_lqr:...,backup_policy_mode=base_controller"   filter settings of the reach
#                                                               tests; the last one uses the hover LQR as the backup
#   and every knob of run_landing_floor_overnight.sh (SEEDS, TOTAL_STEPS, MAX_PARALLEL, SETTINGS, CHECK_SETTING, ...)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/landing_x500}"
VEHICLE="${VEHICLE:-docker_batch/configs/vehicles/gz_x500.json}"
CONFIG="${CONFIG:-docker_batch/configs/landing_x500_cone45_r0p3.json}"
REDESIGN="${REDESIGN:-0}"
REFERENCE_DESIGN="${REFERENCE_DESIGN:-checkpoints/landing_phase1/floor_rec10_td3_seed0/configs.json}"
TASK_REFS="${TASK_REFS:-ps2rl/envs/assets/quadrotor_landing_b*_reference.npz}"
DESIGN_ARGS="${DESIGN_ARGS:-}"
NAME="${NAME:-x500_rec10}"
START_REF="${START_REF:-ps2rl/envs/assets/quadrotor_landing_b2_m30_reference.npz}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"
stamp() { date '+%H:%M:%S'; }

# ------------------------------------------------------------------ design
if [ "${REDESIGN}" = "1" ] || [ ! -f "${CONFIG}" ]; then
    echo "[$(stamp)] [design] ${VEHICLE} -> ${CONFIG}"
    # shellcheck disable=SC2086
    JAX_PLATFORMS=cpu python scripts/design_landing_base_set.py --vehicle "${VEHICLE}" \
        --reference_design "${REFERENCE_DESIGN}" --task_references ${TASK_REFS} ${DESIGN_ARGS} \
        --out "${CONFIG}" 2>&1 | tee "${LOGS}/design.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[$(stamp)] [design] FAILED"; exit 1; }
    rm -f "${OUT}/certificate.json"
else
    echo "[$(stamp)] [design] using ${CONFIG} (REDESIGN=1 derives it again)"
fi

# ------------------------------------------------------------------ certificate
if [ ! -f "${OUT}/certificate.json" ] || [ "${CONFIG}" -nt "${OUT}/certificate.json" ]; then
    echo "[$(stamp)] [certificate] full check of the base set (on the CPU)"
    JAX_PLATFORMS=cpu python scripts/certify_landing_base_set.py --config_json "${CONFIG}" \
        --out "${OUT}/certificate.json" > "${LOGS}/certificate.log" 2>&1 \
        || { echo "[$(stamp)] [certificate] FAILED, see ${LOGS}/certificate.log"; exit 1; }
    grep -E "^(action box|input feasibility|z-independent|gentle-recovery|configured|containment|invariance)" "${LOGS}/certificate.log"
fi
python - "${OUT}/certificate.json" <<'PY' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
c = r["config"]
ok = r["chosen_ok"] and r["containment_min_h"] >= 0.0 and r["invariance_max_V"] <= c["base_set_c"]
print(f"[certificate] c_B = {c['base_set_c']} at z_des = {c['z_des']}: bound {r['c_bar_with_recovery']:.2f}, containment min h "
      f"{r['containment_min_h']:+.3f} m, invariance max V {r['invariance_max_V']:.2f} -> {'OK' if ok else 'NOT CERTIFIED'}")
sys.exit(0 if ok else 1)
PY

# ------------------------------------------------------------------ baseline: the hover LQR alone
if [ ! -f "${OUT}/lqr_baseline.json" ] || [ "${CONFIG}" -nt "${OUT}/lqr_baseline.json" ]; then
    JAX_PLATFORMS=cpu python scripts/evaluate_landing_backup_regions.py --config_json "${CONFIG}" \
        --out "${OUT}/lqr_baseline.json" > "${LOGS}/lqr_baseline.log" 2>&1 \
        || { echo "[$(stamp)] [baseline] FAILED, see ${LOGS}/lqr_baseline.log"; exit 1; }
fi
echo "[$(stamp)] [baseline] recoverability without a learned backup (what training has to beat):"
grep -E "weighted|hover LQR alone" "${LOGS}/lqr_baseline.log"

# ------------------------------------------------------------------ train, check, test, report
# The config file already holds the floor and the gentle-recovery envelope, so the variant has no extra flags.
export OUT START_REF
export BASE_CFG="${CONFIG}"
export VARIANTS="${VARIANTS:-${NAME}:}"
export PHASE2_RUN="${PHASE2_RUN:-none}"   # no Phase-II run for this vehicle yet (stage E is not run)
export STAGES="${STAGES:-B C D F}"
export FOLLOW="${FOLLOW:-1}"
export SETTINGS="${SETTINGS:-a10_f20:alpha=10,alpha_floor=20;a4:alpha=4;a10_f20_lqr:alpha=10,alpha_floor=20,backup_policy_mode=base_controller}"
bash "${HERE}/run_landing_floor_overnight.sh" || exit 1

# ------------------------------------------------------------------ trained backups against the baseline
BACKUPS=()
for d in "${PHASE1_DIR:-${OUT}/phase1}"/*/; do
    [ -f "${d}landing_backup_policy_actor.pkl" ] && BACKUPS+=("${d%/}")
done
if [ "${#BACKUPS[@]}" -gt 0 ]; then
    echo "[$(stamp)] [regions] trained backups against the hover LQR alone"
    JAX_PLATFORMS=cpu python scripts/evaluate_landing_backup_regions.py --backup "${BACKUPS[@]}" \
        --out "${OUT}/regions.json" 2>&1 | tee "${LOGS}/regions.log" | grep -vE "^\[|^$"
fi
echo "[$(stamp)] done: ${OUT}/REPORT.md, ${OUT}/regions.json"
