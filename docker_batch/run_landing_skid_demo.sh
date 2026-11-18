#!/usr/bin/env bash
# Skid-out landing demo inside the px4sitl container: the tracker heads for a spot beside the
# platform, the BCBF filter keeps it in the approach cone and lands it on the platform.
#
#   docker exec -it px4sitl bash ~/ws_shared/PS2-RL/docker_batch/run_landing_skid_demo.sh
#
# Knobs (environment variables; anything after the script name is passed to the demo as-is):
#   OUT        output directory                  (default outputs/landing_skid_demo)
#   BACKUP     Phase-I checkpoint                (default checkpoints/landing_phase1/floor_rec10_td3_seed0)
#   REFERENCE  reference .npz                    (default ps2rl/envs/assets/quadrotor_landing_skid_reference.npz)
#   EXTRA      extra demo arguments, e.g. "--gif_all --n_starts 24"
# The filter settings and trackers compared are the demo's --settings / --trackers defaults;
# override them through EXTRA or the trailing arguments.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh"
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/landing_skid_demo}"
BACKUP="${BACKUP:-checkpoints/landing_phase1/floor_rec10_td3_seed0}"
REFERENCE="${REFERENCE:-ps2rl/envs/assets/quadrotor_landing_skid_reference.npz}"
mkdir -p "${OUT}"

# the demo also runs fine on CPU (a few minutes); fall back if no GPU is attached
if ! ps2rl_gpu_check >/dev/null 2>&1; then
    echo "[skid] no GPU visible to JAX; running on CPU" >&2
    export JAX_PLATFORMS=cpu
fi

# shellcheck disable=SC2086
python scripts/demo_landing_skid.py --backup "${BACKUP}" --reference "${REFERENCE}" --out "${OUT}" ${EXTRA:-} "$@" \
    2>&1 | tee "${OUT}/demo.log"
