#!/usr/bin/env python
"""KAFA results figures: per-step data (CSV) and episode metadata (JSON) of the chosen paper-setting episodes, with the
settings check, the reproduction of the paper's GPU per-episode records and the expected outcomes.

Inputs: REC/<eval run>/ from eval_strike_filter.py --record_episodes ... --save_traj 1 --traj_extra 1 (driver:
docker_batch/run_kafa_results_data.sh), runs ps2_final, ps2_guid1, residual_final (as in the paper evaluation).

    python scripts/kafa_export.py check  --rec REC --root ROOT --task TASK        # prints, exits 1 on a mismatch
    python scripts/kafa_export.py times  --rec REC                               # snapshot times (601, 845)
    python scripts/kafa_export.py export --rec REC --out OUT [--commit TXT] [--commands CMDS]

CSV rows: one per simulation step k = 0 .. end_step - 1 at t = k dt (the state at the start of the step, the command
applied during it, the reward of the step). The ending event (strike / contact) happens at end_time = end_step dt; the
post-step distances of the last step are in the JSON (terminal).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OC = ["strike", "crash_obstacle", "crash_floor", "crash_chaser", "out_of_arena", "timeout"]
SHORT = {"strike": "strike", "crash_obstacle": "obstacle", "crash_chaser": "chaser", "crash_floor": "floor",
         "out_of_arena": "out", "timeout": "timeout"}
KO = {"strike": "타격 성공", "crash_obstacle": "장애물 충돌", "crash_chaser": "요격기 충돌", "crash_floor": "지면 충돌",
      "out_of_arena": "경기장 이탈", "timeout": "시간 초과"}
CONFIG = {"ps2_guid1": "PS2 + 단일 회피 필터", "ps2_final": "PS2 + 최종 필터", "residual_final": "잔차 정책 + 최종 필터",
          "residual_nofilter": "잔차 정책, 필터 없음"}
# episode, config (file name), eval run (REC subfolder), mode, expected outcome, expected end time [s] (None: any),
# tolerance [s] ("≈" values of the request: 0.25 s)
ITEMS = [(601, "ps2_guid1", "ps2_guid1", "learned", "crash_chaser", 8.5, 0.25),
         (601, "ps2_final", "ps2_final", "learned", "strike", 6.5, 0.25),
         (972, "residual_final", "residual_final", "learned", "strike", 14.5, 0.25),
         (972, "ps2_final", "ps2_final", "learned", "strike", 5.5, 0.25),
         (146, "ps2_final", "ps2_final", "learned", "strike", 7.56, 0.005),
         (146, "residual_nofilter", "residual_final", "none", "crash_chaser", 3.52, 0.005),
         (845, "ps2_final", "ps2_final", "learned", "crash_chaser", None, 0.0),
         (957, "ps2_final", "ps2_final", "learned", "strike", 5.62, 0.005),
         (957, "residual_nofilter", "residual_final", "none", "crash_obstacle", 3.44, 0.005)]
RUNS = {"ps2_final": ("ps2/striker_rs0p1_s0", "final", ("learned",)),
        "ps2_guid1": ("ps2/striker_rs0p1_s0", "one", ("learned",)),
        "residual_final": ("residual/striker_rs0p1_s0", "final", ("none", "learned"))}
U_NAMES = ("thrust", "wx", "wy", "wz")


def load(rec, run):
    z = np.load(Path(rec) / run / "strike_filter_traj.npz", allow_pickle=True)
    return z, {int(e): j for j, e in enumerate(z["episode_index"])}


def quat_euler(q):
    """(w, x, y, z) -> roll, pitch, yaw [deg] (ZYX)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def obstacle_distance(pos_xy, obst):
    """The env's obstacle distance (ps2rl.envs.strike_scene.obstacle_distances, min over obstacles) along a path."""
    import jax
    import jax.numpy as jnp
    from ps2rl.envs.strike_scene import obstacle_distances
    f = jax.jit(jax.vmap(lambda p: jnp.min(obstacle_distances(p, jnp.asarray(obst, jnp.float32)))))
    return np.asarray(f(jnp.asarray(pos_xy, jnp.float32)))


def h_c(pos, vel, cps, z):
    """The guidance filter's chaser margin (ps2rl.cil.strike_guidance.h_chasers), min over the active chasers."""
    r_c, r_d, v_c = float(z["chaser_radius"]), float(z["r_drone"]), float(z["chaser_speed"])
    delay, a_e = float(z["delay"]), float(z["a_evade"])
    out = np.full(len(pos), np.inf)
    for cp in cps:
        rel = pos - cp
        R = np.linalg.norm(rel, axis=1) + 1e-12
        s = np.maximum(v_c - np.sum(rel / R[:, None] * vel, axis=1), 0.0)
        out = np.minimum(out, R - (r_c + r_d) - s * delay - s ** 2 / (2 * a_e))
    return out


# ------------------------------------------------------------------------------------------------ check
def check(rec, root, task):
    lines, ok = [], True

    def item(name, value, good):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"[{'OK' if good else 'MISMATCH'}] {name}: {value}")

    root, task = Path(root), Path(task)
    tk = json.loads(task.read_text())
    g_final = json.loads(tk["tracker"]["filter_guidance"])
    res = json.loads((root / "residual/striker_rs0p1_s0/configs.json").read_text())
    ps2 = json.loads((root / "ps2/striker_rs0p1_s0/configs.json").read_text())
    p1 = json.loads((root / "p1/td3_a0p2_seed0/configs.json").read_text())
    item("task file", task.name, task.name == "strike_final_ps2.json")
    item("residual striker", f"best_weights, trained without filter (filter_backup_run '{res['tracker'].get('filter_backup_run', '')}', "
         f"w_filter {res['tracker'].get('w_filter')})", res["tracker"].get("filter_backup_run", "") == ""
         and float(res["tracker"].get("w_filter", 0)) == 0.0 and (root / "residual/striker_rs0p1_s0/best_weights.pkl").exists())
    item("PS2 striker", f"trained through {ps2['tracker'].get('filter_guidance')}",
         json.loads(ps2["tracker"].get("filter_guidance") or "{}") == g_final and float(ps2["tracker"].get("w_filter", 0)) > 0)
    item("backup (Phase I)", f"TD3, {p1['sa']['total_steps']:,} steps; chaser r_c {p1['strike']['chaser_radius']} m, "
         f"v_c {p1['strike']['chaser_speed']} m/s", int(p1["sa"]["total_steps"]) == 10_000_000
         and float(p1["strike"]["chaser_radius"]) == 0.4 and float(p1["strike"]["chaser_speed"]) == 2.0)
    for tag, c in (("residual", res), ("PS2", ps2)):
        e_ = c.get("env", {})
        r_, v_, a_ = e_.get("actual_chaser_radius"), e_.get("actual_chaser_speed"), e_.get("chaser_accel", 2.0)
        item(f"{tag} striker training chaser", f"actual radius {r_}, actual speed {v_}, accel {a_}",
             (r_ is None or float(r_) < 0) and (v_ is None or float(v_) < 0) and float(a_ if a_ is not None else 2.0) == 2.0)
    e = tk["env"]
    item("environment", f"arena {e['arena_x']:g} x {2 * e['arena_y_half']:g} m, obstacles {e['obstacles_min']}-{e['obstacles_max']}, "
         f"chasers {e['chasers_active_min']}-{e['chasers_active_max']}, {e['episode_seconds']:g} s, strike radius {e['strike_radius']} m",
         e["arena_x"] == 40.0 and e["arena_y_half"] == 10.0 and e["obstacles_min"] == e["obstacles_max"] == 3
         and e["chasers_active_min"] == e["chasers_active_max"] == 1 and e["episode_seconds"] == 15.0 and e["strike_radius"] == 0.6)
    for run, (striker, filt, modes) in RUNS.items():
        d = Path(rec) / run
        t = json.loads((d / "strike_filter_table.json").read_text())
        z = np.load(d / "strike_filter_traj.npz", allow_pickle=True)
        env = json.loads(t["env_overrides"]) if t.get("env_overrides") else {}
        pu = t.get("pursuit") or {}
        g = t["guidance"]
        labels, lams = [str(x) for x in z["variant_labels"]], [float(x) for x in z["lambdas"]]
        lines.append(f"[INFO] {run}: striker {striker} (identified by the reproduction below), nominal '{t['nominal']}', "
                     f"backup {t['backup_run']}, modes {list(z['modes'])}")
        item(f"{run}: chaser", f"actual radius {env.get('actual_chaser_radius', -1)}, speed {env.get('actual_chaser_speed', -1)} "
             f"(-1: 0.4 m, 2 m/s), accel {env.get('chaser_accel', 2.0)}; filter assumes r_c {float(z['chaser_radius'])}, "
             f"v_c {float(z['chaser_speed'])}", float(env.get("actual_chaser_radius", -1)) < 0
             and float(env.get("actual_chaser_speed", -1)) < 0 and float(env.get("chaser_accel", 2.0)) == 2.0
             and float(z["chaser_radius"]) == 0.4 and float(z["chaser_speed"]) == 2.0)
        item(f"{run}: scenes", f"{t['episodes']} (seed {t['scene_seed']}), {float(z['dt']) * int(json.loads((d / 'strike_filter_episodes.json').read_text())['max_steps']):g} s",
             t["episodes"] == 1000 and t["scene_seed"] == 9001)
        item(f"{run}: nominal options", f"detour {pu.get('detour_radius', 0)}, heading search {pu.get('search', False)}",
             float(pu.get("detour_radius", 0) or 0) == 0.0 and not pu.get("search", False))
        n_esc = len(g.get("escape_variants") or [])
        want_var = 7 if filt == "final" else 2
        item(f"{run}: filter", f"{len(labels)} recovery policies (j = 0..{len(labels) - 1}) x lambda {lams} = {len(labels) * len(lams)} "
             f"candidates, horizon {g['horizon']} s, committed prediction {g['committed_prediction']}, escape variants {n_esc}",
             len(labels) == want_var and lams == [1.0, 0.75, 0.5, 0.25, 0.0] and float(g["horizon"]) == 3.0
             and g["committed_prediction"] and n_esc == (5 if filt == "final" else 0))
        item(f"{run}: filter options off", f"early escapes {any(len(v) > 2 for v in (g.get('escape_variants') or []))}, "
             f"zero-slack {g.get('fallback_zero_slack')}, lam_rise_max {g.get('lam_rise_max')}",
             not any(len(v) > 2 for v in (g.get("escape_variants") or [])) and not g.get("fallback_zero_slack")
             and float(g.get("lam_rise_max") or 0.0) == 0.0)
        if filt == "final":
            for i, l_ in enumerate(labels):
                lines.append(f"       j = {i}: {l_}")
        # reproduction of the paper's GPU per-episode records
        paper = json.loads((root / "eval/s0" / run / "strike_filter_episodes.json").read_text())
        mine = json.loads((d / "strike_filter_episodes.json").read_text())
        for mode in modes:
            p_, m_ = paper["modes"][mode], mine["modes"][mode]
            n = len(m_["outcome"])
            if n != 1000:
                ok = False
                lines.append(f"[NOT CHECKED] {run}/{mode}: {n} episodes simulated, not the full evaluation (CPU replay)")
                continue
            so = sum(int(a == b) for a, b in zip(m_["outcome"], p_["outcome"]))
            ss = sum(int(a == b) for a, b in zip(m_["end_step"], p_["end_step"]))
            item(f"{run}/{mode}: reproduction of the paper record", f"outcome {so}/1000, end step {ss}/1000", so == 1000 and ss == 1000)
    # expected outcomes (the request's table)
    for e, cfg, run, mode, out_exp, t_exp, tol in ITEMS:
        z, idx = load(rec, run)
        j = idx[e]
        out = OC[int(z[f"{mode}/outcome"][j])]
        t_end = int(z[f"{mode}/end_step"][j]) * float(z["dt"])
        good = out == out_exp and (t_exp is None or abs(t_end - t_exp) <= tol + 1e-9)
        item(f"expected {e} {cfg}", f"{KO[out]} {t_end:.2f} s (expected {KO[out_exp]}"
             + (f" {'≈ ' if tol > 0.01 else ''}{t_exp:g} s" if t_exp is not None else "") + ")", good)
    return ok, lines


# ------------------------------------------------------------------------------------------------ snapshot times
def snapshot_times(rec):
    z, idx = load(rec, "ps2_guid1")
    j = idx[601]
    k = int(z["learned/end_step"][j])
    nc = np.asarray(z["learned/nocert"][:k, j]) > 0.5
    k1 = int(np.argmax(nc)) if nc.any() else None
    z2, idx2 = load(rec, "ps2_final")
    j2 = idx2[845]
    k2 = int(z2["learned/end_step"][j2])
    nc2 = np.asarray(z2["learned/nocert"][:k2, j2]) > 0.5
    k_last = None
    if nc2.any():
        on = np.flatnonzero(nc2 & ~np.concatenate([[False], nc2[:-1]]))       # onsets of loss stretches
        k_last = int(on[-1])
    dt = float(z["dt"])
    return {"601_t_star": None if k1 is None else round(k1 * dt, 2), "601_t_before": None if k1 is None else round(max(k1 * dt - 1.0, 0.0), 2),
            "845_t_last_loss": None if k_last is None else round(k_last * dt, 2),
            "845_loss_steps_from_last_onset": None if k_last is None else int(nc2[k_last:].sum())}


# ------------------------------------------------------------------------------------------------ export
COLUMNS = {
    "t": "시간 [s] (스텝 시작, t = k dt)",
    "x, y, z": "드론 위치 [m]", "vx, vy, vz": "드론 속도 [m/s]", "speed_xy": "수평 속력 [m/s]",
    "heading_deg": "수평 속도 방향 [deg] (atan2(vy, vx))", "roll, pitch, yaw": "자세 [deg] (ZYX 오일러각, 쿼터니언에서)",
    "cx, cy, cz, cvx, cvy, cvz": "요격기 위치 [m] · 속도 [m/s] (활성 요격기 1대; 여러 대면 c0_, c1_ 접두어)",
    "tx, ty, tz": "목표 대상 위치 [m]", "dist_target": "드론-목표 거리 [m] (성공: < 0.6 m)",
    "lam": "적용된 λ (1 = 명목 입력 유지, 0 = 회복 정책 그대로); 인증 상실 · 이전 계획 유지 스텝은 0",
    "j": "적용된 입력을 인증한 회복 정책 번호 (λ = 1이면 명목 입력을 인증한 회복 정책)",
    "committed": "1: 새 인증 없이 이전 계획 유지 (committed plan)", "cert_lost": "1: 인증 상실 (새 계획도 이전 계획도 실패)",
    "n_certified": "그 스텝에서 인증된 새 후보 수 (n_candidates 중: 최종 35, 단일 회피 10)",
    "cert_mask": "인증된 새 후보 (0/1 문자열, 회복 정책 j 순 × λ = 1, .75, .5, .25, 0 순)",
    "h_c": "요격기 여유도 [m] h_c = R − (r_c + r_d) − s δ − s²/(2 a_e), s = max(v_c − eᵀv, 0) (활성 요격기 중 최소)",
    "d_obs": "장애물 표면까지 최소 수평 거리 [m] (드론 중심 기준; 충돌: < r_d = 0.3 m)",
    "d_wall": "경기장 벽 (x = 0, 40, y = ±10)까지 최소 거리 [m] (이탈 판정은 벽 밖 1 m)",
    "z_margin": "바닥 · 천장 최소 여유 [m] min(z − z_floor, z_max − z)",
    "u_nom_*": "명목 입력 (thrust: 집단 추력 가속도 [m/s²], wx, wy, wz: 기체 각속도 [rad/s])", "u_*": "적용된 입력 (같은 단위)",
    "eta_norm": "‖(u − u_nom) / u_scale‖ (필터 수정량)", "reward": "그 스텝의 환경 보상 (작업 보상, 적용 입력 u 기준)"}


def fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.6g}"


def export_one(rec, e, cfg, run, mode, out_dir, commit, commands, exp):
    from ps2rl.envs.strike_env import StrikeEnvConfig
    z, idx = load(rec, run)
    j = idx[e]
    dt = float(z["dt"])
    k_end = int(z[f"{mode}/end_step"][j])
    g = lambda key: np.asarray(z[f"{mode}/{key}"])[:k_end, j] if f"{mode}/{key}" in z.files else None
    pos, vel, quat, tg = g("pos"), g("vel"), g("quat"), g("target")
    ch = np.asarray(z[f"{mode}/chaser"])[:k_end, j]
    chv = g("chaser_v")
    on = np.asarray(z["chaser_on"])[j] > 0.5
    act = [c for c in range(ch.shape[1]) if on[c]]
    u, un = g("u"), g("u_nom")
    u_scale = np.asarray(z["u_scale"], float)
    obst = np.asarray(z["obst"])[j]
    ax_x, ay = (float(v) for v in z["arena"])
    z_floor, z_max = float(z["z_floor"]), float(z["z_max"])
    roll, pitch, yaw = quat_euler(quat)
    hc = h_c(pos, vel, [ch[:, c] for c in act], z)
    d_obs = obstacle_distance(pos[:, :2], obst)
    d_wall = np.minimum.reduce([pos[:, 0], ax_x - pos[:, 0], pos[:, 1] + ay, ay - pos[:, 1]])
    z_m = np.minimum(pos[:, 2] - z_floor, z_max - pos[:, 2])
    filt = mode != "none"
    lam, var, com, nc, cert = g("lam"), g("variant"), g("committed"), g("nocert"), g("certified")
    rew = g("reward")
    eta = np.linalg.norm((u - un) / u_scale, axis=1)
    pre = (lambda c: "") if len(act) == 1 else (lambda c: f"c{act.index(c)}_")
    header = ["t", "x", "y", "z", "vx", "vy", "vz", "speed_xy", "heading_deg", "roll", "pitch", "yaw"]
    for c in act:
        header += [pre(c) + k for k in ("cx", "cy", "cz", "cvx", "cvy", "cvz")]
    header += ["tx", "ty", "tz", "dist_target", "lam", "j", "committed", "cert_lost", "n_certified", "cert_mask",
               "h_c", "d_obs", "d_wall", "z_margin"]
    header += [f"u_nom_{n}" for n in U_NAMES] + [f"u_{n}" for n in U_NAMES] + ["eta_norm", "reward"]
    rows = []
    for k in range(k_end):
        r = [k * dt, *pos[k], *vel[k], np.hypot(vel[k, 0], vel[k, 1]), np.degrees(np.arctan2(vel[k, 1], vel[k, 0])),
             roll[k], pitch[k], yaw[k]]
        for c in act:
            r += [*ch[k, c], *(chv[k, c] if chv is not None else [None] * 3)]
        r += [*tg[k], float(np.linalg.norm(tg[k] - pos[k]))]
        if filt:
            r += [lam[k], int(round(float(var[k]))), bool(com[k] > 0.5), bool(nc[k] > 0.5),
                  int(np.sum(cert[k])) if cert is not None else None,
                  "".join("1" if b else "0" for b in cert[k]) if cert is not None else None]
        else:
            r += [None] * 6
        r += [hc[k], d_obs[k], d_wall[k], z_m[k], *un[k], *u[k], eta[k], rew[k] if rew is not None else None]
        rows.append(r)
    name = f"ep{e}_{cfg}"
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([x if isinstance(x, str) else fmt(x) for x in r])
    out = OC[int(z[f"{mode}/outcome"][j])]
    tk = json.loads((REPO / "docker_batch/configs/strike_final_ps2.json").read_text())
    ec = StrikeEnvConfig.from_dict(tk.get("env", {}))
    t_tab = json.loads((Path(rec) / run / "strike_filter_table.json").read_text())
    obstacles = []
    for o in obst:
        if o[6] < 0.5:
            continue
        cx, cy, bx, by, yw, rr = (float(v) for v in o[:6])
        if bx == 0 and by == 0:
            obstacles.append({"type": "disk", "type_ko": "원판 (원기둥)", "center": [cx, cy], "radius": rr})
        else:
            obstacles.append({"type": "rect", "type_ko": "직사각형 (사각기둥)", "center": [cx, cy], "half_widths": [bx, by],
                              "corner_radius": rr, "yaw_deg": float(np.degrees(yw))})
    for o in obstacles:
        o["height"] = None
        o["height_note"] = "모델에서 무한 높이의 수직 기둥 (수평 거리만 판정); 서론 GIF에서는 8 m로 그림"
    lastk = k_end - 1
    gl = lambda key: float(np.asarray(z[f"{mode}/{key}"])[lastk, j]) if f"{mode}/{key}" in z.files else None
    meta = {
        "episode": e, "config": cfg, "config_ko": CONFIG[cfg], "eval_run": run, "mode": mode, "evaluation_set": "seed 9001, 1000 scenes",
        "outcome": SHORT[out], "outcome_ko": KO[out], "end_time": round(k_end * dt, 4), "end_step": k_end, "dt": dt,
        "rows": len(rows), "rows_note": "k = 0 .. end_step − 1 (스텝 시작 상태); 종료 사건은 end_time에 발생",
        "expected": exp,
        "arena": {"x": [0.0, ax_x], "y": [-ay, ay], "z_floor": z_floor, "z_max": z_max,
                  "walls": [{"x": 0.0}, {"x": ax_x}, {"y": -ay}, {"y": ay}],
                  "out_of_arena": f"x < -1 or x > {ax_x + 1:g} or |y| > {ay + 1:g} or z > {z_max:g}",
                  "floor_crash": f"z < {z_floor:g}"},
        "obstacles": obstacles,
        "drone_radius": float(z["r_drone"]), "chaser_radius": float(z["chaser_radius"]),
        "chaser_contact_distance": float(z["contact_chaser"]) if "contact_chaser" in z.files else float(z["chaser_radius"]) + float(z["r_drone"]),
        "chaser_speed": float(z["chaser_speed"]), "chaser_accel": ec.chaser_accel, "strike_radius": float(z["strike_radius"]),
        "target_initial": np.round(tg[0], 4).tolist(), "drone_initial": {"pos": np.round(pos[0], 4).tolist(), "vel": np.round(vel[0], 4).tolist()},
        "chasers_initial": [np.round(ch[0, c], 4).tolist() for c in act],
        "terminal": {"dist_target": gl("dist_next"), "d_obs": gl("d_obs_next"), "d_chaser_center": gl("d_ch_next"),
                     "note": "마지막 스텝 직후 (종료 시점) 값; d_chaser_center는 중심 간 거리 (접촉: < chaser_contact_distance)"},
        "h_c_constants": {"r_c": float(z["chaser_radius"]), "r_d": float(z["r_drone"]), "v_c": float(z["chaser_speed"]),
                          "delta": float(z["delay"]), "a_e": float(z["a_evade"])},
        "u_scale": u_scale.tolist(), "u_names": list(U_NAMES), "u_low": np.asarray(z["u_low"]).tolist(), "u_high": np.asarray(z["u_high"]).tolist(),
        "reward": {"definition": "w_progress (d_prev − d) − w_time − w_action mean((du/u_scale)²) − envelope + strike_bonus·strike − crash_penalty·crash, du = u − u_hover",
                   "weights": {k: getattr(ec, k) for k in ("w_progress", "w_time", "w_action", "w_speed_excess", "speed_soft",
                                                            "w_tilt_excess", "tilt_soft_deg", "strike_bonus", "crash_penalty")}},
        "nominal": t_tab.get("nominal"), "commit": commit, "command": commands.get(run, ""),
        "columns": COLUMNS}
    if filt:
        g_ = t_tab["guidance"]
        meta["filter"] = {"recovery_policies": [f"j = {i}: {s}" for i, s in enumerate(str(x) for x in z["variant_labels"])],
                          "lambdas": [float(v) for v in z["lambdas"]],
                          "n_candidates": len(z["variant_labels"]) * len(z["lambdas"]), "horizon_s": float(z["horizon"]),
                          "committed_prediction": bool(g_["committed_prediction"]), "escape_variants": g_.get("escape_variants"),
                          "fresh_margin_hc": float(z["fresh_margin"]), "fresh_obstacle_margin": float(z["fresh_obstacle_margin"]),
                          "hover_latch": int(z["hover_latch"]) if "hover_latch" in z.files else None,
                          "summary": {"active_fraction": round(float(np.mean((lam < 0.999) | (com > 0.5) | (nc > 0.5))), 4),
                                      "committed_steps": int(np.sum(com > 0.5)), "cert_lost_steps": int(np.sum(nc > 0.5)),
                                      "min_h_c": round(float(hc.min()), 4), "j_used": sorted(set(int(round(float(v)))
                                                                                                   for v in var[(lam < 0.999) | (com > 0.5)]))}}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return dict(file=name, episode=e, config=CONFIG[cfg], outcome=KO[out], time=round(k_end * dt, 2), rows=len(rows), expected=exp)


def export(rec, out, commit_file=None, commands_file=None):
    out = Path(out)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    commit = Path(commit_file).read_text().strip() if commit_file and Path(commit_file).exists() else "unknown"
    commands = {}
    if commands_file and Path(commands_file).exists():
        for line in Path(commands_file).read_text().splitlines():
            for run in RUNS:                                                  # the evaluation that recorded the run
                if "eval_strike_filter.py" in line and line.rstrip().rstrip("'\"").endswith(f"/{run}"):
                    commands[run] = line
    summary = []
    for e, cfg, run, mode, out_exp, t_exp, tol in ITEMS:
        z, idx = load(rec, run)
        j = idx[e]
        o = OC[int(z[f"{mode}/outcome"][j])]
        t_end = int(z[f"{mode}/end_step"][j]) * float(z["dt"])
        exp = {"outcome": SHORT[out_exp], "time": t_exp, "tolerance": tol,
               "match": o == out_exp and (t_exp is None or abs(t_end - t_exp) <= tol + 1e-9)}
        summary.append(export_one(rec, e, cfg, run, mode, out / "csv", commit, commands, exp))
        print(f"wrote {summary[-1]['file']} ({summary[-1]['rows']} rows)")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def readme(out, summary, check_txt, times, commit, commands_txt, preview=False):
    lines = ["# KAFA 결과 그림용 데이터 (논문 설정)", ""]
    if preview:
        lines += ["> **CPU 미리보기**: --only_episodes CPU 재생. 최종 데이터는 `docker_batch/run_kafa_results_data.sh` (GPU EXACT).", ""]
    else:
        lines += ["> GPU EXACT: 평가 1,000개 전체를 시뮬레이션하면서 고른 에피소드만 기록 (`--record_episodes`), 논문 GPU 기록과 에피소드별 대조.", ""]
    lines += [f"- 커밋: {commit}",
              "- 설정: seed 0 모델 (잔차 striker · PS2 striker · Phase I TD3 1,000만 스텝), 요격기 0.4 m · 2 m/s · 2 m/s², scene seed 9001, 15 s",
              "- 최종 필터: 회복 정책 7개 × λ {1, .75, .5, .25, 0}, 3 s 예측, committed prediction; heading search · early escape · "
              "zero-slack · λ ramp · detour OFF. 단일 회피 필터 (ps2_guid1): 같은 필터에서 escape_variants 제거 (회복 정책 2개 × λ 5 = 후보 10).", "",
              "## 결과", "", "| 파일 | 에피소드 | 구성 | 결과 · 시간 | 기대값 | 일치 | 행 수 |", "|---|---|---|---|---|---|---|"]
    for s in summary:
        ex = s["expected"]
        exs = KO[next(k for k, v in SHORT.items() if v == ex["outcome"])] + (f" {'≈ ' if ex['tolerance'] > 0.01 else ''}{ex['time']:g} s" if ex["time"] is not None else "")
        lines.append(f"| {s['file']}.csv | {s['episode']} | {s['config']} | {s['outcome']} {s['time']:.2f} s | {exs} | "
                     f"{'예' if ex['match'] else '아니오'} | {s['rows']} |")
    lines += ["", "## 열 (CSV)", ""] + [f"- `{k}`: {v}" for k, v in COLUMNS.items()]
    lines += ["", "필터 없는 run (잔차 정책, 필터 없음)은 필터 열 (lam, j, committed, cert_lost, n_certified, cert_mask)이 빈 칸.", ""]
    if times:
        lines += ["## 스냅숏 시점", "", f"- 601: 단일 회피 필터가 인증을 처음 잃은 t* = {times.get('601_t_star')} s, 그 1 s 전 = "
                  f"{times.get('601_t_before')} s (두 구성 모두)", f"- 845: 마지막 인증 상실 시작 t = {times.get('845_t_last_loss')} s "
                  f"(그 뒤 인증 없는 스텝 {times.get('845_loss_steps_from_last_onset')}개)",
                  "- `snapshots/<구성>/snapshots_ep<i>.json`: 회복 정책 7개 (단일 회피는 2개)의 λ = 0 3초 예측 궤적 (x, y, z 점열, dt 0.02 s), "
                  "후보 인증 행렬 (회복 정책 × λ), 선택 후보의 계획과 요격기 예측, 실행 중 기록된 인증 행렬 (`certified_recorded`, 정확값)과 "
                  "재검사 일치 여부. PNG/PDF는 같은 시점의 그림 (참고용).", ""]
    lines += ["## 설정 확인", "", "```", check_txt.strip(), "```", "", "## 명령", "", "```bash", commands_txt.strip(), "```", ""]
    (Path(out) / "README_data.md").write_text("\n".join(lines))
    print("wrote", Path(out) / "README_data.md")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--rec", required=True)
    c.add_argument("--root", required=True)
    c.add_argument("--task", default=str(REPO / "docker_batch/configs/strike_final_ps2.json"))
    t = sub.add_parser("times")
    t.add_argument("--rec", required=True)
    x = sub.add_parser("export")
    x.add_argument("--rec", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--commit", default="")
    x.add_argument("--commands", default="")
    x.add_argument("--check", default="")
    x.add_argument("--preview", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "check":
        ok, lines = check(a.rec, a.root, a.task)
        nc = any(l_.startswith("[NOT CHECKED]") for l_ in lines)
        print("\n".join(lines))
        print("[check] " + ("OK" if ok else ("NOT CHECKED (CPU replay)" if nc and not any("MISMATCH" in l_ for l_ in lines) else "MISMATCH")))
        sys.exit(0 if ok or (nc and not any("MISMATCH" in l_ for l_ in lines)) else 1)
    elif a.cmd == "times":
        print(json.dumps(snapshot_times(a.rec)))
    else:
        summary = export(a.rec, a.out, a.commit, a.commands)
        times = snapshot_times(a.rec)
        (Path(a.out) / "snapshot_times.json").write_text(json.dumps(times, indent=1))
        readme(a.out, summary, Path(a.check).read_text() if a.check and Path(a.check).exists() else "",
               times, Path(a.commit).read_text().strip() if a.commit and Path(a.commit).exists() else "unknown",
               Path(a.commands).read_text() if a.commands and Path(a.commands).exists() else "", bool(a.preview))
        for s in summary:
            print(f"| {s['file']} | {s['outcome']} {s['time']:.2f} s | {'일치' if s['expected']['match'] else '불일치'} | {s['rows']} |")


if __name__ == "__main__":
    main()
