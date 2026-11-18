#!/usr/bin/env bash
# Phase-II landing training on the GPU, for each reference:
#   B  trackers  (SAC, no safety layer, landing reward v2; preset "tracker"), one run per seed in TRACKER_SEEDS
#   S  selection: every checkpoint (best and final of every seed) is rolled out and the one with the
#                smallest tracking error becomes <ref>_tracker
#   C  PS2       (SAC through the landing CIL, warm-started from the selected tracker). The projection
#                is thousands of tiny QPs, which a GPU does not necessarily run faster than the CPU, so
#                the stage first times a short PS2 run on both (PS2_DEVICE=auto) and trains on the faster
#   D  evaluation: tracker alone / tracker + filter / PS2, same starts, figures + GIF
#   E  REPORT.md
#
#   docker exec -d px4sitl bash -c "bash ~/ws_shared/PS2-RL/docker_batch/run_landing_phase2_train.sh \
#       > ~/ws_shared/PS2-RL/outputs/landing_phase2.out 2>&1"
#   docker exec -it px4sitl bash -c "tail -F ~/ws_shared/PS2-RL/outputs/landing_phase2.out"
#
# While it runs, OUT/training.html shows every run's curves (it reloads itself). `touch OUT/<run>/STOP`
# ends that run early: it finishes normally (final evaluation, weights, summary) and the batch moves on.
# Resumable: a run with a summary.json is skipped; an evaluation with a summary.md is skipped.
# Stale results are moved aside as <name>_old_<timestamp> (never deleted): a single-seed <ref>_tracker
# of the first batch version (old recipe), a PS2 run older than its tracker's selection, and an
# evaluation older than its PS2 run.
# Knobs (environment variables):
#   OUT            output root                                   (outputs/landing_phase2)
#   REFS           "name:path name:path"                         (the six chosen descent references, see below)
#   BACKUP         Phase-I checkpoint                            (checkpoints/landing_phase1/floor_rec10_td3_seed0)
#   FILTER         landing CIL settings                          (alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1)
#   TRACKER_SEEDS  tracker seeds                                 ("0 1 2")
#   SEED           PS2 training seed                             (0)
#   TRACKER_STEPS  tracker training steps (upper bound)          (3000000)
#   PS2_STEPS      PS2 training steps (upper bound)              (1500000)
#   PS2_HOURS      wall-time budget per PS2 run, 0 = none        (3)
#   PS2_DEVICE     auto | gpu | cpu                              (auto: timed, see stage C)
#   PS2_PROBE_STEPS  env steps of each timing run                (8192)
#   MAX_PARALLEL   runs at once on the GPU                       (3)
#   CURVES_EVERY   seconds between refreshes of the curves, 0 = off  (30)
#   STAGES         subset of "B S C D E"                         (all)
#   TRACKER_ARGS / PS2_ARGS / EVAL_ARGS   extra arguments for the trainer / evaluator
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/landing_env.sh" || exit 1
cd "${PS2RL_ROOT}"

OUT="${OUT:-outputs/landing_phase2}"
# Default references: candidates B2_-30, B2_-20, B2_-10, B3_-30, B3_-20, B3_-10 of
# scripts/landing_reference_candidates.py (hover at 2 m, near-vertical descent drifting toward the axis late,
# touchdown 20 / 30 cm beyond the platform rim, 4 s). Older ones: skid / nearside / cornercut in ps2rl/envs/assets.
_A=ps2rl/envs/assets
REFS="${REFS:-b2_m30:${_A}/quadrotor_landing_b2_m30_reference.npz b2_m20:${_A}/quadrotor_landing_b2_m20_reference.npz b2_m10:${_A}/quadrotor_landing_b2_m10_reference.npz b3_m30:${_A}/quadrotor_landing_b3_m30_reference.npz b3_m20:${_A}/quadrotor_landing_b3_m20_reference.npz b3_m10:${_A}/quadrotor_landing_b3_m10_reference.npz}"
BACKUP="${BACKUP:-checkpoints/landing_phase1/floor_rec10_td3_seed0}"
FILTER="${FILTER:-alpha=10,alpha_floor=20,control_weights=1/0.1/0.1/1}"
SEED="${SEED:-0}"
TRACKER_SEEDS="${TRACKER_SEEDS:-0 1 2}"
TRACKER_STEPS="${TRACKER_STEPS:-3000000}"
PS2_STEPS="${PS2_STEPS:-1500000}"
PS2_HOURS="${PS2_HOURS:-3}"
PS2_DEVICE="${PS2_DEVICE:-auto}"
PS2_PROBE_STEPS="${PS2_PROBE_STEPS:-8192}"
CURVES_EVERY="${CURVES_EVERY:-30}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
STAGES="${STAGES:-B S C D E}"
mkdir -p "${OUT}/logs"

ps2rl_gpu_check || { echo "[phase2] GPU check failed (set PS2RL_ALLOW_CPU=1 JAX_PLATFORMS=cpu to force CPU)"; exit 1; }
echo "[phase2] $(date '+%F %T') OUT=${OUT} BACKUP=${BACKUP} FILTER=${FILTER} SEED=${SEED}"
echo "[phase2] tracker ${TRACKER_STEPS} steps, seeds ${TRACKER_SEEDS}; PS2 up to ${PS2_STEPS} steps / ${PS2_HOURS} h; ${MAX_PARALLEL} at once; stages ${STAGES}"
stamp="$(date '+%Y%m%d_%H%M%S')"
# move_aside PATH: keep stale results under a new name instead of deleting them
move_aside() { [ -e "$1" ] && { echo "[phase2] stale: $1 -> $1_old_${stamp}"; mv "$1" "$1_old_${stamp}"; }; return 0; }

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# Live training curves: OUT/training.html (reloads itself) and OUT/training_<mode>.png. Not a job of this
# shell (disown), so it neither takes one of the MAX_PARALLEL slots nor holds up `wait`.
if [ "${CURVES_EVERY}" != "0" ]; then
    python scripts/plot_landing_training.py --root "${OUT}" --watch "${CURVES_EVERY}" > "${OUT}/logs/curves.log" 2>&1 &
    curves_pid=$!
    disown "${curves_pid}"
    trap 'kill "${curves_pid}" 2>/dev/null; python scripts/plot_landing_training.py --root "${OUT}" > /dev/null 2>&1' EXIT
    echo "[phase2] training curves: ${OUT}/training.html (refreshed every ${CURVES_EVERY} s); to end a run early: touch ${OUT}/<run>/STOP"
fi

# run_job NAME CMD...: runs in the background, log in OUT/logs/NAME.log, lines echoed with a prefix
run_job() {
    local name="$1"; shift
    while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do sleep 5; done
    echo "[phase2] $(date '+%T') start ${name}"
    ( "$@" 2>&1 | tee "${OUT}/logs/${name}.log" | sed -u "s/^/[${name}] /"; \
      echo "[phase2] $(date '+%T') end ${name} (exit ${PIPESTATUS[0]})" ) &
}

ref_names=()
for item in ${REFS}; do
    name="${item%%:*}"; path="${item#*:}"
    [ -f "${path}" ] || { echo "[phase2] reference not found: ${path}"; exit 1; }
    ref_names+=("${name}")
    declare "REF_${name}=${path}"
done

# A tracker of the single-seed batch version was trained with the old recipe (preset tracker_v1). It is
# kept aside and takes no part in the selection, so every PS2 run starts from the same kind of tracker.
for name in "${ref_names[@]}"; do
    old="${OUT}/${name}_tracker"
    if [ -d "${old}" ] && [ ! -f "${old}/selection.json" ]; then move_aside "${old}"; fi
done

if has_stage B; then
    for seed in ${TRACKER_SEEDS}; do
        for name in "${ref_names[@]}"; do
            ref_var="REF_${name}"; run="${name}_tracker_s${seed}"
            if [ -f "${OUT}/${run}/summary.json" ]; then echo "[phase2] skip ${run} (done)"; continue; fi
            # shellcheck disable=SC2086
            run_job "${run}" python scripts/train_phase2_landing.py --reference "${!ref_var}" --backup "${BACKUP}" \
                --filter "${FILTER}" --projection off --seed "${seed}" --total_steps "${TRACKER_STEPS}" \
                --out_root "${OUT}" --run_name "${run}" ${TRACKER_ARGS:-}
        done
    done
    wait
fi

if has_stage S; then
    for name in "${ref_names[@]}"; do
        cands=()
        for d in "${OUT}/${name}"_tracker_s*; do [ -f "${d}/summary.json" ] && cands+=("${d}"); done
        [ "${#cands[@]}" -gt 0 ] || { echo "[phase2] ${name}: no tracker runs to select from; run stage B first"; continue; }
        sel="${OUT}/${name}_tracker/selection.json"; fresh=1
        [ -f "${sel}" ] || fresh=0
        for d in "${cands[@]}"; do [ "${fresh}" = 1 ] && [ "${d}/summary.json" -nt "${sel}" ] && fresh=0; done
        if [ "${fresh}" = 1 ]; then echo "[phase2] skip select_${name} (up to date)"; continue; fi
        run_job "select_${name}" python scripts/select_landing_tracker.py --candidates "${cands[@]}" --out "${OUT}/${name}_tracker"
    done
    wait
fi

# ps2_train NAME RUN OUT_ROOT STEPS HOURS [extra arguments]: PS2 training for reference NAME on ${PS2_PLATFORM},
# with the selected tracker's weights and reset library (so tracker and PS2 see the same starts)
ps2_train() {
    local name="$1" run="$2" root="$3" steps="$4" hours="$5"; shift 5
    local ref_var="REF_${name}"
    # shellcheck disable=SC2086
    JAX_PLATFORMS="${PS2_PLATFORM}" python scripts/train_phase2_landing.py --reference "${!ref_var}" --backup "${BACKUP}" \
        --filter "${FILTER}" --projection on --seed "${SEED}" --total_steps "${steps}" \
        --wall_time_budget_hours "${hours}" --warm_start_weights "${OUT}/${name}_tracker/best_weights.pkl" \
        --init_library "${OUT}/${name}_tracker/init_states.npz" \
        --out_root "${root}" --run_name "${run}" ${PS2_ARGS:-} "$@"
}

# probe_rate PLATFORM NAME: MAX_PARALLEL short PS2 runs at once (as in the real stage); prints the mean
# steady-state steps/s per run (the last logging interval), 0 if they failed
probe_rate() {
    local dev="$1" name="$2" i
    PS2_PLATFORM="${dev}"
    rm -rf "${OUT}/probe/${dev}_"*
    for i in $(seq 1 "${MAX_PARALLEL}"); do
        run_job "probe_${dev}_${i}" ps2_train "${name}" "${dev}_${i}" "${OUT}/probe" "${PS2_PROBE_STEPS}" 0 \
            --log_every "$((PS2_PROBE_STEPS / 4))" --eval_every "$((PS2_PROBE_STEPS * 2))" --eval_episodes 1 >&2
    done
    wait
    python - "${OUT}/probe" "${dev}" <<'PY'
import json, pathlib, sys
rates = []
for f in sorted(pathlib.Path(sys.argv[1]).glob(f"{sys.argv[2]}_*/metrics.jsonl")):
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if rows and (pathlib.Path(f).parent / "summary.json").exists():
        rates.append(rows[-1]["perf/steps_per_sec"])
print(f"{sum(rates) / len(rates):.1f}" if rates else "0")
PY
}

if has_stage C; then
    todo=()
    for name in "${ref_names[@]}"; do
        run="${name}_ps2"; sel="${OUT}/${name}_tracker/selection.json"
        [ -f "${OUT}/${name}_tracker/best_weights.pkl" ] && [ -f "${sel}" ] || { echo "[phase2] ${run}: no selected tracker in ${OUT}/${name}_tracker; run stages B S first"; continue; }
        # a PS2 run trained before the current tracker selection started from another tracker
        if [ -e "${OUT}/${run}" ] && { [ ! -f "${OUT}/${run}/summary.json" ] || [ "${OUT}/${run}/summary.json" -ot "${sel}" ]; }; then
            move_aside "${OUT}/${run}"
        fi
        if [ -f "${OUT}/${run}/summary.json" ]; then echo "[phase2] skip ${run} (done)"; continue; fi
        todo+=("${name}")
    done
    if [ "${#todo[@]}" -gt 0 ]; then
        case "${PS2_DEVICE}" in
            gpu) PS2_PLATFORM="${JAX_PLATFORMS}" ;;
            cpu) PS2_PLATFORM="cpu" ;;
            auto)
                if [ "${JAX_PLATFORMS}" = "cpu" ]; then
                    PS2_PLATFORM="cpu"
                else
                    echo "[phase2] $(date '+%T') timing PS2 on ${JAX_PLATFORMS} and cpu (${MAX_PARALLEL} runs at once, ${PS2_PROBE_STEPS} steps each, reference ${todo[0]})"
                    rate_gpu="$(probe_rate "${JAX_PLATFORMS}" "${todo[0]}")"
                    rate_cpu="$(probe_rate cpu "${todo[0]}")"
                    PS2_PLATFORM="$(python -c "import sys; g, c = float(sys.argv[1]), float(sys.argv[2]); print('cpu' if c > g else sys.argv[3])" "${rate_gpu}" "${rate_cpu}" "${JAX_PLATFORMS}")"
                    echo "[phase2] $(date '+%T') PS2 speed per run with ${MAX_PARALLEL} at once: ${JAX_PLATFORMS} ${rate_gpu} steps/s, cpu ${rate_cpu} steps/s -> training on ${PS2_PLATFORM}"
                fi ;;
            *) echo "[phase2] PS2_DEVICE must be auto, gpu or cpu (got ${PS2_DEVICE})"; exit 1 ;;
        esac
        echo "[phase2] PS2 runs on ${PS2_PLATFORM}"
        for name in "${todo[@]}"; do
            run_job "${name}_ps2" ps2_train "${name}" "${name}_ps2" "${OUT}" "${PS2_STEPS}" "${PS2_HOURS}"
        done
        wait
    fi
fi

if has_stage D; then
    for name in "${ref_names[@]}"; do
        ev="${OUT}/eval_${name}"
        [ -f "${OUT}/${name}_tracker/selection.json" ] || { echo "[phase2] eval_${name}: no selected tracker; run stages B S first"; continue; }
        ps2=""; newest="${OUT}/${name}_tracker/selection.json"
        if [ -f "${OUT}/${name}_ps2/summary.json" ]; then ps2="--ps2_run ${OUT}/${name}_ps2"; newest="${OUT}/${name}_ps2/summary.json"; fi
        if [ -e "${ev}" ] && { [ ! -f "${ev}/summary.md" ] || [ "${ev}/summary.md" -ot "${newest}" ]; }; then move_aside "${ev}"; fi
        if [ -f "${ev}/summary.md" ]; then echo "[phase2] skip eval_${name} (done)"; continue; fi
        # shellcheck disable=SC2086
        run_job "eval_${name}" python scripts/evaluate_landing_phase2_runs.py --tracker_run "${OUT}/${name}_tracker" \
            ${ps2} --filter "${FILTER}" --backup "${BACKUP}" --title "${name}" --out "${ev}" ${EVAL_ARGS:-}
    done
    wait
fi

if has_stage E; then
    python scripts/summarize_landing_phase2.py --root "${OUT}" --names "${ref_names[@]}" 2>&1 | tee "${OUT}/logs/report.log"
fi
echo "[phase2] $(date '+%F %T') finished; report: ${OUT}/REPORT.md"
