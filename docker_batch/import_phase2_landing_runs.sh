#!/usr/bin/env bash
# Import a colleague's Phase-II landing work (PS2-RL.tar or an extracted PS2-RL folder)
# into this repo without touching our code:
#
#   * the landing reference(s) + generator  -> ps2rl/envs/assets/
#   * the Phase-II run folders               -> outputs/landing/<run>/   (gitignored)
#
# Their modified ps2rl/envs/quadrotor_env.py is NOT copied: it only changes the training
# reward, and copying it would drop this fork's disturbance options. The evaluation never
# uses the reward.
#
#   bash docker_batch/import_phase2_landing_runs.sh /path/to/PS2-RL.tar
#   bash docker_batch/import_phase2_landing_runs.sh /path/to/PS2-RL      # extracted folder
#
# Env: RUN_GLOB (default "*") selects which outputs/landing/<run> folders to import.
set -euo pipefail

SRC="${1:?usage: import_phase2_landing_runs.sh <PS2-RL.tar | PS2-RL dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_GLOB="${RUN_GLOB:-*}"

TMP=""
cleanup() { [ -n "${TMP}" ] && rm -rf "${TMP}"; }
trap cleanup EXIT

if [ -f "${SRC}" ]; then
    TMP="$(mktemp -d)"
    tar -xf "${SRC}" -C "${TMP}"
    envs_dir="$(find "${TMP}" -maxdepth 4 -type d -path '*/ps2rl/envs' | head -n1)"
    [ -n "${envs_dir}" ] || { echo "no ps2rl/envs inside ${SRC}" >&2; exit 1; }
    SRC_DIR="$(cd "${envs_dir}/../.." && pwd)"
else
    SRC_DIR="$(cd "${SRC}" && pwd)"
fi
[ -d "${SRC_DIR}/ps2rl/envs/assets" ] || { echo "no ps2rl/envs/assets under ${SRC_DIR}" >&2; exit 1; }
echo "[import] source repo: ${SRC_DIR}"

# 1) landing references (+ their _config.json / .png) and the generator
shopt -s nullglob
n_ref=0
for f in "${SRC_DIR}"/ps2rl/envs/assets/*landing*; do
    [ -f "${f}" ] || continue
    dst="${ROOT}/ps2rl/envs/assets/$(basename "${f}")"
    if [ -f "${dst}" ] && ! cmp -s "${f}" "${dst}"; then
        echo "[import] ${dst} differs from the incoming file; keeping ours as ${dst}.bak"
        cp -p "${dst}" "${dst}.bak"
    fi
    cp -p "${f}" "${dst}"
    echo "[import] asset  $(basename "${f}")"
    n_ref=$((n_ref + 1))
done
[ "${n_ref}" -gt 0 ] || echo "[import] warning: no *landing* assets found" >&2

# 2) Phase-II run folders (anything with configs.json + weights)
mkdir -p "${ROOT}/outputs/landing"
n_run=0
for d in "${SRC_DIR}"/outputs/landing/${RUN_GLOB}/; do
    d="${d%/}"
    [ -f "${d}/configs.json" ] || continue
    ls "${d}"/*_weights.pkl >/dev/null 2>&1 || continue
    name="$(basename "${d}")"
    mkdir -p "${ROOT}/outputs/landing/${name}"
    # metrics.jsonl is large and not needed for evaluation
    (cd "${d}" && find . -path ./metrics.jsonl -prune -o -path '*/.ipynb_checkpoints' -prune -o -type f -print0 \
        | tar --null -cf - -T - ) | tar -xf - -C "${ROOT}/outputs/landing/${name}"
    echo "[import] run    outputs/landing/${name}"
    n_run=$((n_run + 1))
done
echo "[import] ${n_ref} asset file(s), ${n_run} run folder(s) imported into ${ROOT}"
