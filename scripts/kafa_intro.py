#!/usr/bin/env python
"""KAFA presentation, introduction figures (paper settings): motivation episodes, their 3-D GIFs, intro slide images.

Inputs are eval_strike_filter.py --save_traj 1 outputs of the paper's seed-0 models (residual striker without filter = A,
PS2 striker + final filter = B) on the evaluation scenes (seed 9001), and the GPU per-episode records of the paper
evaluation (results/strike_reduced/final/run/eval/s0/{residual_final,ps2_final}).

    # candidates: metrics, scores, top-5 tables per failure type, shortlist.png (A and B top views overlaid)
    python scripts/kafa_intro.py screen --a A_DIR --b B_DIR [--b B_DIR2] --out outputs/kafa_gifs/intro
    # a camera shared by A and B of one scene (JSON), chosen so the failure point and the evasion are not hidden
    python scripts/kafa_intro.py cam --a A_DIR --b B_DIR --episode 641 --type obstacle > cam.json
    # one motivation GIF (1280 x 768, 3-D, one run), its last frame as a 300 dpi PNG, and an MP4
    python scripts/kafa_intro.py gif --traj A_DIR:none --episode 641 --cam cam.json --label "잔차 정책 (필터 없음)" \
        --out outputs/kafa_gifs/intro/intro_obs_A.gif
    python scripts/kafa_intro.py gif --traj B_DIR:learned --episode 641 --cam cam.json --label "PS2 + 최종 필터" --chip 1 ...
    # intro slide images (1600 x 1000, transparent)
    python scripts/kafa_intro.py col1 --out outputs/kafa_gifs/intro/col1_uav.png
    python scripts/kafa_intro.py col2 --traj A_DIR:none --episode 69 --out outputs/kafa_gifs/intro/col2_scene.png
    python scripts/kafa_intro.py col3 --traj ALL_DIR:none --out outputs/kafa_gifs/intro [--episodes 1,2,3,4,5,6]

Objects are drawn to scale in the GIFs and col2 (drone: rotor tips at r_drone = 0.3 m; chaser: a missile 2 r_c = 0.8 m
long); col3's small cells enlarge the icons (stated on the figure). Text: Noto Sans CJK KR (the KR face of the system's
Noto CJK collection is registered when matplotlib does not list it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import strike_gif as sg  # noqa: E402

OC = ["strike", "crash_obstacle", "crash_floor", "crash_chaser", "out_of_arena", "timeout"]
END_KO = {0: "타격 성공", 1: "장애물 충돌", 2: "지면 충돌", 3: "요격기 충돌", 4: "경기장 이탈", 5: "시간 초과"}
# the paper's GPU per-episode records (KAFA_EVAL, set by docker_batch/run_kafa_intro.sh to ROOT/eval/s0)
GPU_EVAL = Path(os.environ.get("KAFA_EVAL", str(REPO / "results/strike_reduced/final/run/eval/s0")))
WHITE = "#ffffff"


# ------------------------------------------------------------------------------------------------ text
def font_kr():
    """Noto Sans CJK KR for all text; registers the KR face of the Noto CJK .ttc collections if needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    name = "Noto Sans CJK KR"
    if name not in {f.name for f in font_manager.fontManager.ttflist}:
        cache = Path.home() / ".cache" / "strike_gif"
        for style in ("Regular", "Bold"):
            ttc = sorted(glob.glob(f"/usr/share/fonts/**/NotoSansCJK-{style}.ttc", recursive=True))
            if not ttc:
                continue
            out = cache / f"NotoSansCJKkr-{style}.otf"
            if not out.exists():
                from fontTools.ttLib import TTCollection
                cache.mkdir(parents=True, exist_ok=True)
                face = next((f for f in TTCollection(ttc[0]).fonts if f["name"].getDebugName(1) == name), None)
                if face is None:
                    continue
                face.save(str(out))
            font_manager.fontManager.addfont(str(out))
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
    else:
        sg.setup_font()
        print(f"[kafa] {name} not found; using {plt.rcParams['font.family']}")
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ------------------------------------------------------------------------------------------------ geometry
def obst_sdf(o, p):
    """xy distance from p to obstacle o (cx, cy, bx, by, yaw, rr, present), negative inside."""
    cx, cy, bx, by, yaw, rr = o[:6]
    d = np.asarray(p[:2], float) - [cx, cy]
    c, s = np.cos(yaw), np.sin(yaw)
    q = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    if bx == 0 and by == 0:
        return float(np.linalg.norm(q) - rr)
    a = np.abs(q) - [bx, by]
    return float(np.linalg.norm(np.maximum(a, 0)) + min(max(a[0], a[1]), 0.0) - rr)


def h_c(p, v, cp, z):
    """The guidance filter's chaser margin (ps2rl/cil/strike_guidance.h_chasers) with the run's constants."""
    r_c, r_d, v_c = float(z["chaser_radius"]), float(z["r_drone"]), float(z["chaser_speed"])
    delay, a_e = float(z["delay"]), float(z["a_evade"])
    rel = p - cp
    R = np.linalg.norm(rel, axis=-1) + 1e-12
    e = rel / R[..., None]
    s = np.maximum(v_c - np.sum(e * v, axis=-1), 0.0)
    return R - (r_c + r_d) - s * delay - s ** 2 / (2 * a_e)


class Run(sg.Traj):
    """A recorded run (strike_gif.Traj) plus the raw npz (constants for h_c)."""

    def __init__(self, spec):
        super().__init__(spec)
        self.z = np.load(self.dir / "strike_filter_traj.npz", allow_pickle=True)

    def hc(self, e):
        j, k = self.ep(e)
        out = np.full(k, np.inf)
        for c in range(self.ch.shape[2]):
            if self.on[j, c] > 0.5:
                out = np.minimum(out, h_c(self.pos[:k, j], self.vel[:k, j], self.ch[:k, j, c], self.z))
        return out


def find_run(runs, e):
    for r in runs:
        if e in r.idx:
            return r
    return None


# ------------------------------------------------------------------------------------------------ screening
def gpu_records():
    A = json.loads((GPU_EVAL / "residual_final/strike_filter_episodes.json").read_text())
    B = json.loads((GPU_EVAL / "ps2_final/strike_filter_episodes.json").read_text())
    return A["modes"]["none"], B["modes"]["learned"], float(A["dt"])


def metrics(e, qa, qb, ga, gb, dt):
    """Per-episode screening numbers (A and B outcomes from the GPU records; geometry from the replays)."""
    ja, ka = qa.ep(e)
    jb, kb = qb.ep(e)
    pa = qa.pos[ka - 1, ja]
    obs = [o for o in qa.obst[ja]]
    d_obs = [obst_sdf(o, pa) if o[6] > 0.5 else 1e9 for o in obs]
    hit = int(np.argmin(d_obs))
    pb = qb.pos[:kb, jb]
    lam = qb.lam[:kb, jb]
    com = qb.committed[:kb, jb] > 0.5 if qb.committed is not None else np.zeros(kb, bool)
    act = (lam < 0.999) | com
    var = qb.variant[:kb, jb].astype(int)
    hc = qb.hc(e)
    k_min = int(np.argmin(hc))
    near = np.linalg.norm(pb[:, :2] - pa[:2], axis=1)
    sdf_hit = np.array([obst_sdf(obs[hit], q) for q in pb])
    win = slice(max(0, k_min - int(1.0 / dt)), min(kb, k_min + int(1.0 / dt)))
    v = qb.vel[:kb, jb]
    hd = np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))
    spd = np.linalg.norm(v, axis=1)
    return dict(
        episode=e, A=OC[ga["outcome"][e]], tA=round(ga["end_step"][e] * dt, 2), B=OC[gb["outcome"][e]],
        tB=round(gb["end_step"][e] * dt, 2), nc=int(gb["no_cert_steps"][e]),
        A_replay=OC[int(qa.outcome[ja])], tA_replay=round(ka * dt, 2), B_replay=OC[int(qb.outcome[jb])], tB_replay=round(kb * dt, 2),
        nc_replay=int(np.sum(qb.nocert[:kb, jb] > 0.5)) if qb.nocert is not None else -1,
        min_hc=round(float(hc.min()), 2), t_min_hc=round(k_min * dt, 2), act_frac=round(float(act.mean()), 3),
        j_used=sorted(set(var[act].tolist())), crash_xy=np.round(pa[:2], 2).tolist(), hit_obstacle=hit,
        B_pass_crash=round(float(near.min()), 2), B_act_near_crash=bool(np.any(act & (near < 3.0))),
        B_min_sdf_hit=round(float(sdf_hit.min()), 2), B_act_near_hit=bool(np.any(act & (sdf_hit < 1.5))),
        B_act_at_min_hc=bool(np.any(act[win])), B_j2_at_min_hc=bool(np.any((var[win] >= 2) & act[win])),
        B_turn_at_min_hc=round(float(np.ptp(hd[win])), 1), B_slow_at_min_hc=round(float(spd[win].max() - spd[win].min()), 2))


def score(m, typ):
    tB = m["tB"]
    s = 1.0 * (tB <= 8.0) + float(np.clip((8.0 - tB) / 4.0, 0, 1)) - m["nc"] / 5.0
    if typ == "obstacle":
        s += 2.0 * (2.0 <= m["tA"] <= 4.5) + 2.0 * m["B_act_near_crash"] + 1.0 * m["B_act_near_hit"]
        s += 1.0 * (m["B_min_sdf_hit"] < 1.5) - 1.0 * (m["B_pass_crash"] > 4.0) - 1.0 * (m["min_hc"] < 1.0)
    else:
        h = m["min_hc"]                 # new plans keep h_c >= fresh_margin (0.2 m): ~0.2 is the filter's own floor
        s += 2.0 * (0.15 <= h <= 1.0) + 0.5 * (0.0 <= h < 0.15) + 1.0 * (1.0 < h <= 1.5) - 2.0 * (h < 0)
        s += 2.0 * (m["B_act_at_min_hc"] or m["B_j2_at_min_hc"]) + 0.5 * m["B_j2_at_min_hc"]
        s += 1.0 * (m["B_turn_at_min_hc"] >= 30.0 or m["B_slow_at_min_hc"] >= 1.5) + 1.0 * (m["B_turn_at_min_hc"] >= 45.0)
        s += 0.5 * (m["tA"] >= 3.0)                                          # a longer chase before A is hit
    return round(s, 2)


def screen(a, a_dirs_b, out, n_top=5):
    plt = font_kr()
    ga, gb, dt = gpu_records()
    qa = Run(a)
    qbs = [Run(b) for b in a_dirs_b]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for typ, need, t_min in (("obstacle", "crash_obstacle", 1.5), ("chaser", "crash_chaser", 2.5)):
        rows = []
        for e in sorted(set().union(*[set(q.idx) for q in qbs])):
            qb = find_run(qbs, e)
            if OC[ga["outcome"][e]] != need or OC[gb["outcome"][e]] != "strike" or ga["end_step"][e] * dt < t_min:
                continue
            if gb["no_cert_steps"][e] >= 10 or e not in qa.idx:
                continue
            m = metrics(e, qa, qb, ga, gb, dt)
            m["score"] = score(m, typ)
            m["eval_set"] = True
            rows.append(m)
        rows.sort(key=lambda m: -m["score"])
        res[typ] = rows
    (out / "screen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    lines = []
    for typ, title in (("obstacle", "(i) 장애물 충돌"), ("chaser", "(ii) 요격기 충돌")):
        lines += [f"### {title}: 상위 {n_top}", "",
                  "| 환경 | 평가 세트 | A 결과 · 시간 | B 결과 · 시간 | B min h_c [m] | B 필터 개입 비율 | B 인증 상실 스텝 | B 사용 j | 점수 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in res[typ][:n_top]:
            lines.append(f"| {m['episode']} | {'예' if m['eval_set'] else '아니오'} | {END_KO[OC.index(m['A'])]} {m['tA']:.2f} s | "
                         f"{END_KO[OC.index(m['B'])]} {m['tB']:.2f} s | {m['min_hc']:.2f} | {m['act_frac']:.3f} | {m['nc']} | "
                         f"{','.join(map(str, m['j_used'])) or '-'} | {m['score']:.2f} |")
        lines.append("")
    (out / "shortlist.md").write_text("\n".join(lines))
    print("\n".join(lines))
    shortlist_png(qa, qbs, res, out / "shortlist.png", plt)
    return res


def shortlist_png(qa, qbs, res, out, plt, n=3):
    th = sg.THEMES["light"]
    sk = sg.Skins("light", "missile", "glass", "vehicle")
    fig, axs = plt.subplots(2, n, figsize=(5.4 * n, 6.8), facecolor=WHITE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.03, wspace=0.03, hspace=0.36)
    for r, (typ, title) in enumerate((("obstacle", "(i) 장애물 충돌"), ("chaser", "(ii) 요격기 충돌"))):
        for c in range(n):
            ax = axs[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c >= len(res[typ]):
                ax.axis("off")
                continue
            m = res[typ][c]
            e = m["episode"]
            qb = find_run(qbs, e)
            ja, ka = qa.ep(e)
            jb, kb = qb.ep(e)
            ax.set_facecolor(th["bg"])
            ax.set_xlim(-1, 41), ax.set_ylim(-11, 11), ax.set_aspect("equal")
            ax.plot([0, 40, 40, 0, 0], [-10, -10, 10, 10, -10], color=th["wall"], lw=1.4)
            for o in qa.obst[ja]:
                if o[6] > 0.5:
                    sk.building(ax, o)
            for q, j, k, col, ls, lab in ((qa, ja, ka, "#8c8b86", "--", "A 잔차 (필터 없음)"), (qb, jb, kb, th["drone"], "-", "B PS2 + 최종 필터")):
                ax.plot(q.pos[:k, j, 0], q.pos[:k, j, 1], color=col, lw=2.0, ls=ls, label=lab, zorder=6)
                for ci in range(q.ch.shape[2]):
                    if q.on[j, ci] > 0.5:
                        ax.plot(q.ch[:k, j, ci, 0], q.ch[:k, j, ci, 1], color=th["chaser"], lw=1.0, ls=ls, alpha=0.7, zorder=5)
            ax.plot(*qa.pos[ka - 1, ja, :2], "x", color=th["bad"], ms=12, mew=3, zorder=8)
            ax.plot(*qb.tg[kb - 1, jb, :2], "D", color=th["target"], ms=8, mec="white", zorder=8)
            ax.plot(*qa.pos[0, ja, :2], "o", color=th["drone"], ms=6, zorder=8)
            ax.set_title(f"{title} #{c + 1}  ·  환경 {e}\nA {END_KO[OC.index(m['A'])]} {m['tA']:.2f} s → B 타격 {m['tB']:.2f} s  ·  "
                         f"min h_c {m['min_hc']:.2f} m  ·  점수 {m['score']:.1f}", loc="left", fontsize=9.5)
            if r == 0 and c == 0:
                ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.suptitle("Motivation 후보 (평가 세트, seed 9001): A 잔차 정책 · 필터 없음 (회색 점선, ✗ 충돌 지점) vs B PS2 + 최종 필터 (파랑)",
                 x=0.01, y=0.985, ha="left", fontsize=11.5)
    fig.savefig(out, dpi=130, facecolor=WHITE)
    plt.close(fig)
    print("wrote", out)


# ------------------------------------------------------------------------------------------------ camera
def ray_blocked(p, eye_dir, obst, top=8.0, skip=None, length=45.0):
    """True when the ray from p toward the camera passes through an obstacle prism (other than `skip`) below its top."""
    for t in np.linspace(0.4, length, 180):
        q = p + t * eye_dir
        if q[2] > top:
            return False
        for i, o in enumerate(obst):
            if o[6] > 0.5 and i != skip and obst_sdf(o, q) < 0.05:
                return True
    return False


def eye(elev, azim):
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def key_points(qa, qb, e):
    """Points the motivation GIFs must keep in frame: both drone paths (and their ground shadows), the chasers, the target."""
    pts = []
    for q in (qa, qb):
        j, k = q.ep(e)
        p = q.pos[:k, j]
        pts += [p, p * [1, 1, 0]]
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                pts.append(q.ch[:k, j, c])
        tp = q.tg[k - 1, j]
        ring = np.array([[np.cos(a) * 2.1, np.sin(a) * 2.1, 0.0] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
        pts += [tp[None] + ring, tp[None] * [1, 1, 0] + ring]                # reticle and vehicle
    return np.concatenate(pts)[::2]


def fit_cam(cam, pts, size=(1280, 768), margin=(0.05, 0.15)):
    """Zoom and pan (axes offsets) so the projected points fill the frame inside the margins (the same for A and B)."""
    plt = font_kr()
    W, H = size
    fig = plt.figure(figsize=(W / 100, H / 100))
    cam0 = dict(cam, zoom=1.0, v_shift=0.0, h_shift=0.0)
    ax = scene_axes(fig, cam0)
    fig.canvas.draw()
    xy = np.array([project(ax, p) for p in pts])
    plt.close(fig)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    z = min(W * (1 - 2 * margin[0]) / (x1 - x0), H * (1 - 2 * margin[1]) / (y1 - y0))
    cx, cy = W / 2, H / 2
    dx = W / 2 - (cx + z * ((x0 + x1) / 2 - cx))
    dy = H / 2 - (cy + z * ((y0 + y1) / 2 - cy))
    return dict(cam, zoom=round(float(z), 4), h_shift=round(float(dx / W), 4), v_shift=round(float(dy / H), 4))


def auto_cam(qa, qb, e, typ, elev=28.0, focal=1.0, size=(1280, 768)):
    """A fixed camera shared by A and B: +x to the right (camera on the -y side), A's failure point and B's evasion not
    behind another building, then zoom and pan fitted to both paths, the chasers and the target."""
    ja, ka = qa.ep(e)
    jb, kb = qb.ep(e)
    obst = qa.obst[ja]
    pa = qa.pos[ka - 1, ja]
    hit, n = None, None
    if typ == "obstacle":
        hit = int(np.argmin([obst_sdf(o, pa) if o[6] > 0.5 else 1e9 for o in obst]))
        n = pa[:2] - obst[hit][:2]
        n = n / (np.linalg.norm(n) + 1e-9)
        pb = qb.pos[int(np.argmin(np.linalg.norm(qb.pos[:kb, jb, :2] - pa[:2], axis=1))), jb]
    else:
        pb = qb.pos[int(np.argmin(qb.hc(e))), jb]
    best = None
    for az in range(-140, -44, 4):
        ev = eye(elev, az)
        s = -abs(az + 100) / 40.0                                           # prefer a side view slightly from behind
        s -= 3.0 * ray_blocked(pa, ev, obst, skip=hit) + 2.0 * ray_blocked(pb, ev, obst)
        if n is not None:
            s += 2.0 * float(ev[:2] @ n > 0.2)
        if best is None or s > best[0]:
            best = (s, az)
    pts = key_points(qa, qb, e)
    lo, hi = pts.min(0), pts.max(0)
    cam = dict(elev=elev, azim=float(best[1]), focal=focal,
               xlim=[float(lo[0]) - 1.0, float(hi[0]) + 1.0], ylim=[float(lo[1]) - 1.0, float(hi[1]) + 1.0], zlim=[0.0, 8.0])
    return fit_cam(cam, pts, size)


def apply_cam(ax, cam):
    (x0, x1), (y0, y1), (z0, z1) = cam["xlim"], cam["ylim"], cam["zlim"]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=cam.get("zoom", 1.0))
    ax.set_proj_type("persp", focal_length=cam.get("focal", 1.0))
    ax.view_init(elev=cam["elev"], azim=cam["azim"])


def scene_axes(fig, cam, rect=None):
    """A 3-D axes filling the figure width (3-D axes draw in a square viewport), transparent, camera applied."""
    W, H = fig.get_size_inches()
    if rect is None:
        h = W / H
        rect = [cam.get("h_shift", 0.0), 0.5 - h / 2 + cam.get("v_shift", 0.0), 1.0, h]
    ax = fig.add_axes(rect, projection="3d")
    ax.patch.set_alpha(0.0)
    ax.computed_zorder = False
    ax.set_axis_off()
    apply_cam(ax, cam)
    return ax


def static_scene(ax, sk, q, j, cam, ground=True, building_top=8.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    th = sk.th
    ax_x, ay = q.arena[0], q.arena[1]
    if ground:
        ax.add_collection3d(Poly3DCollection([[(0, -ay, 0), (ax_x, -ay, 0), (ax_x, ay, 0), (0, ay, 0)]], facecolor=th["ground"],
                                             edgecolor="none", zorder=0))
        for gx in np.arange(5, ax_x, 5):
            ax.plot([gx, gx], [-ay, ay], [0, 0], color=th["grid"], lw=0.8, zorder=0.5)
        for gy in np.arange(-ay + 5, ay, 5):
            ax.plot([0, ax_x], [gy, gy], [0, 0], color=th["grid"], lw=0.8, zorder=0.5)
        ax.plot([0, ax_x, ax_x, 0, 0], [-ay, -ay, ay, ay, -ay], [0] * 5, color=th["wall"], lw=1.6, zorder=0.6)
    sk.buildings3d(ax, q.obst[j], building_top)
    apply_cam(ax, cam)


# ------------------------------------------------------------------------------------------------ motivation GIF
def chip_of(q, j, k):
    if q.nocert is not None and q.nocert[k, j] > 0.5:
        return "인증 상실", "bad"
    lam = float(q.lam[k, j])
    com = q.committed is not None and q.committed[k, j] > 0.5
    if lam >= 0.999 and not com:
        return "명목 입력 유지", "ok"
    if lam <= 1e-3 or com:
        return f"회피 정책 j = {int(q.variant[k, j])}", "info"
    return f"필터 개입 λ = {round(lam, 2):g}", "warn"


def draw_movers(ax, sk, q, j, k, t, r_d, L_c, trail_s=4.0, chaser_trail_s=2.5, full_path=True):
    th = sk.th
    arts = []
    p, tp = q.pos[k, j], q.tg[k, j]
    if full_path:
        arts += ax.plot(q.pos[:k + 1, j, 0], q.pos[:k + 1, j, 1], q.pos[:k + 1, j, 2], color=th["trail"], lw=0.9, alpha=0.3,
                        zorder=5)
        arts += ax.plot(q.pos[:k + 1, j, 0], q.pos[:k + 1, j, 1], np.full(k + 1, 0.02), color=(0, 0, 0, 0.12), lw=1.0, zorder=1)
    arts += sg.fade_line3d(ax, q.pos[max(0, k - int(trail_s / q.dt)):k + 1, j], th["trail"], lw=2.6, z=6,
                           alpha_min=0.05 if full_path else 0.0)
    arts += sk.target3d(ax, tp, sg.heading(q.tg[:, j], k), t)
    for c in range(q.ch.shape[2]):
        if q.on[j, c] > 0.5:
            arts += sg.fade_line3d(ax, q.ch[max(0, k - int(chaser_trail_s / q.dt)):k + 1, j, c], th["chaser"], lw=2.0,
                                   alpha_max=0.45, z=7, alpha_min=0.0)
            arts += sk.chaser3d(ax, q.ch[k, j, c], sg.heading3(q.ch[:, j, c], k), t, length=L_c)
    arts += sk.quad3d(ax, p, q.quat[k, j] if q.quat is not None else None, radius=r_d)
    return arts


def crash3d(ax, th, p, k):
    k = min(k, 8)
    a = np.linspace(0, 2 * np.pi, 40)
    r = 0.6 + 0.35 * k
    arts = [ax.scatter(*p, s=(80 + 110 * k) * 3, c=[th["bad"]], alpha=max(0.8 - 0.09 * k, 0.08), edgecolors="none", zorder=15,
                       depthshade=False)]
    arts += ax.plot(p[0] + r * np.cos(a), p[1] + r * np.sin(a), np.full(40, p[2]), color=th["bad"], lw=1.6, alpha=0.75, zorder=15)
    arts.append(ax.scatter(*p, s=320, marker="x", c=th["bad"], linewidths=3.5, zorder=24, depthshade=False))
    return arts


def render_gif(spec, episode, out, cam, label, chip=False, speed=1.0, hold=2.0, stride=2, size=(1280, 768), dpi=100,
               mp4=True, png_dpi=300, building_top=8.0):
    plt = font_kr()
    from matplotlib import animation
    q = Run(spec)
    th = sg.THEMES["light"]
    sk = sg.Skins3D("light", "missile", "glass", "vehicle")
    j, k_end = q.ep(episode)
    r_d, L_c = float(q.z["r_drone"]), 2.0 * float(q.z["chaser_radius"])
    W, H = size[0] / dpi, size[1] / dpi
    fig = plt.figure(figsize=(W, H), facecolor=WHITE)
    ax = scene_axes(fig, cam)
    static_scene(ax, sk, q, j, cam, building_top=building_top)
    ax.plot(q.tg[:k_end, j, 0], q.tg[:k_end, j, 1], np.zeros(k_end) + 0.03, color=th["target"], lw=0.8, ls=":", alpha=0.6,
            zorder=1)
    fig.text(0.018, 0.968, label, fontsize=15, fontweight="bold", color=th["text"], va="top")
    chip_t = fig.text(0.018, 0.905, "", fontsize=13, fontweight="bold", color="white", va="top") if chip else None
    tt = fig.text(0.982, 0.03, "", fontsize=12, color=th["sub"], ha="right", va="bottom")
    badge = fig.text(0.5, 0.075, "", fontsize=21, fontweight="bold", color="white", ha="center", va="center")
    n_hold = int(round(hold * speed / (q.dt * stride)))
    frames = list(range(0, k_end, stride))
    if frames[-1] != k_end - 1:
        frames.append(k_end - 1)
    n_move = len(frames)
    frames += [k_end - 1] * n_hold
    dyn = []
    code = int(q.outcome[j])

    def draw(fi):
        for a_ in dyn:
            a_.remove()
        dyn.clear()
        k = frames[fi]
        t = k * q.dt
        ended = fi >= n_move - 1
        held = fi - (n_move - 1)
        dyn.extend(draw_movers(ax, sk, q, j, k, t, r_d, L_c))
        if ended:
            dyn.extend(sk.flash3d(ax, q.tg[k, j], held) if code == 0 else crash3d(ax, th, q.pos[k, j], held))
            col = th["ok"] if code == 0 else th["bad"]
            badge.set_text(f" {'✓' if code == 0 else '✗'} {END_KO[code]} {k_end * q.dt:.1f} s ")
            badge.set_bbox(dict(boxstyle="round,pad=0.45", fc=col, ec="none", alpha=0.92))
        else:
            badge.set_text("")
            badge.set_bbox(None)
        tt.set_text(f"t = {t:.1f} s" + (f"   ×{speed:g}" if abs(speed - 1.0) > 1e-6 else ""))
        if chip_t is not None:
            txt, key = chip_of(q, j, k)
            chip_t.set_text(f" {txt} ")
            chip_t.set_bbox(dict(boxstyle="round,pad=0.32", fc=th[key], ec="none", alpha=0.94))
        return dyn

    fps = speed / (q.dt * stride)
    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": WHITE})
    sg.finish(out, mp4)
    draw(len(frames) - 1)
    fig.savefig(out.with_suffix(".png"), dpi=png_dpi, facecolor=WHITE)
    plt.close(fig)
    info = dict(file=out.name, episode=int(episode), label=label, run=str(q.dir), mode=q.mode, outcome=END_KO[code],
                time=round(k_end * q.dt, 2), min_hc=round(float(q.hc(episode).min()), 2), fps=fps, frames=len(frames),
                size=list(size), speed=speed, cam=cam)
    if q.mode != "none":
        lam = q.lam[:k_end, j]
        act = (lam < 0.999) | (q.committed[:k_end, j] > 0.5)
        info.update(act_frac=round(float(act.mean()), 3), j_used=sorted(set(q.variant[:k_end, j].astype(int)[act].tolist())),
                    no_cert_steps=int(np.sum(q.nocert[:k_end, j] > 0.5)))
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print("wrote", out, json.dumps(info, ensure_ascii=False))
    return info


# ------------------------------------------------------------------------------------------------ intro slide images
def col1(out, size=(1600, 1000), dpi=100):
    """The renderer's drone, close up: 3/4 view, pitched forward, ground shadow, transparent background."""
    plt = font_kr()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    th = sg.THEMES["light"]
    W, H = size[0] / dpi, size[1] / dpi
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_alpha(0.0)
    r = 0.3
    cam = dict(elev=24.0, azim=-58.0, zoom=1.0, xlim=[-0.3, 0.3], ylim=[-0.3, 0.3], zlim=[0.0, 0.36], v_shift=0.04, focal=0.8)
    ax = scene_axes(fig, cam)
    pitch = np.radians(14.0)                                                # nose down, flying +x
    c, s = np.cos(pitch / 2), np.sin(pitch / 2)
    quat = np.array([c, 0.0, s, 0.0])
    R = sg.quat_R(quat)
    p = np.array([0.0, 0.0, 0.2])
    sz = r / 1.277
    tips = p + (np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]]) * 0.62 * sz) @ R.T
    a = np.linspace(0, 2 * np.pi, 48)
    for rr, al in ((0.24, 0.07), (0.19, 0.08), (0.13, 0.09)):              # soft ground shadow
        ax.add_collection3d(Poly3DCollection([np.stack([p[0] + 0.03 + rr * np.cos(a), p[1] - 0.02 + rr * 0.8 * np.sin(a),
                                                        np.zeros(48)], 1)], facecolor=(0, 0, 0, al), edgecolor="none", zorder=1))
    for a_, b_ in ((0, 2), (1, 3)):
        ax.plot(*np.stack([tips[a_], tips[b_]], 1), color=th["drone_arm"], lw=13, solid_capstyle="round", zorder=10)
    body = np.array([[x, y, zz] for zz in (-0.022, 0.03) for y in (-0.055, 0.055) for x in (-0.085, 0.085)])
    body = p + body @ R.T
    f = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    nrm = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)]
    ax.add_collection3d(Poly3DCollection([[body[i] for i in ff] for ff in f],
                                         facecolors=[sg.shade(th["drone"], R @ np.asarray(n_)) for n_ in nrm],
                                         edgecolors=th["drone_arm"], linewidths=1.2, zorder=12))
    ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * 0.40 * sz
    for t_ in tips:
        hub_top = t_ + R @ np.array([0, 0, 0.02])
        ax.plot(*np.stack([t_, hub_top], 1), color=th["drone_arm"], lw=9, zorder=12.5)
        ax.add_collection3d(Poly3DCollection([hub_top + ring @ R.T], facecolor=th["rotor"], edgecolor=th["drone"], linewidths=1.6,
                                             alpha=0.55, zorder=13))
        for ang in (0.3, 0.3 + np.pi):                                       # two-blade propeller
            d = R @ (np.array([np.cos(ang), np.sin(ang), 0.0]) * 0.36 * sz)
            ax.plot(*np.stack([hub_top - d, hub_top + d], 1), color=th["drone_arm"], lw=3.0, zorder=13.5)
    nose = p + R @ np.array([0.085, 0, 0.004])
    ax.scatter(*nose, s=160, c="white", edgecolors=th["drone_arm"], linewidths=1.2, zorder=14, depthshade=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, transparent=True)
    plt.close(fig)
    print("wrote", out)


def project(ax, p):
    from mpl_toolkits.mplot3d import proj3d
    x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
    return ax.transData.transform((x, y))                                   # display pixels


def col2_moment(q, j, k_end, dist=(2.5, 3.5)):
    """The step shown in col2: the chaser closing in at a distance in `dist` [m], latest such step; a chaser behind the
    drone is preferred (in the evaluation set the chaser starts between the drone and the target, so its approach is
    usually head-on). Returns (step, 'behind' | 'ahead')."""
    found = {}
    for c in range(q.ch.shape[2]):
        if q.on[j, c] < 0.5:
            continue
        for k in range(k_end - 1, int(0.5 / q.dt), -1):
            rel = q.pos[k, j] - q.ch[k, j, c]
            d = float(np.linalg.norm(rel))
            closing = float(np.dot(q.vel[k, j] - (q.ch[k, j, c] - q.ch[k - 1, j, c]) / q.dt, rel)) < 0
            side = "behind" if float(np.dot(q.vel[k, j], -rel)) < 0 else "ahead"
            if dist[0] <= d <= dist[1] and closing and side not in found:
                found[side] = k
    if "behind" in found:
        return found["behind"], "behind"
    if "ahead" in found:
        return found["ahead"], "ahead"
    return max(0, k_end - int(0.8 / q.dt)), "fallback"


def col2(spec, episode, out, k=None, size=(1600, 1000), dpi=100, trail_s=1.5, cam=None):
    plt = font_kr()
    q = Run(spec)
    th = sg.THEMES["light"]
    sk = sg.Skins3D("light", "missile", "glass", "vehicle")
    j, k_end = q.ep(episode)
    k, side = col2_moment(q, j, k_end) if k is None else (int(k), "given")
    cam = cam or dict(elev=30.0, azim=-112.0, zoom=1.0, xlim=[-1.0, 40.0], ylim=[-10.0, 10.0], zlim=[0.0, 8.0], v_shift=0.0, focal=1.0)
    r_d, L_c = float(q.z["r_drone"]), 2.0 * float(q.z["chaser_radius"])
    outs = []
    for labeled in (False, True):
        W, H = size[0] / dpi, size[1] / dpi
        fig = plt.figure(figsize=(W, H))
        fig.patch.set_alpha(0.0)
        ax = scene_axes(fig, cam)
        static_scene(ax, sk, q, j, cam)
        t = k * q.dt
        draw_movers(ax, sk, q, j, k, t, r_d, L_c, trail_s=trail_s, chaser_trail_s=trail_s, full_path=False)
        if labeled:
            fig.canvas.draw()
            items = [("드론", q.pos[k, j], (-110, 120)), ("요격기", q.ch[k, j, int(np.argmax(q.on[j]))], (-60, -150)),
                     ("목표 대상", q.tg[k, j] + [0, 0, 0.3], (60, 120))]
            obs = [o for o in q.obst[j] if o[6] > 0.5]
            o = min(obs, key=lambda o: abs(o[0] - 20.0))
            items.append(("장애물", np.array([o[0], o[1], 6.5]), (90, 90)))
            for txt, p3, (dx, dy) in items:
                xy = project(ax, p3)
                fx, fy = fig.transFigure.inverted().transform(xy)
                tx, ty = fig.transFigure.inverted().transform(xy + np.array([dx, dy]))
                fig.add_artist(__import__("matplotlib").patches.ConnectionPatch((tx, ty), (fx, fy), "figure fraction",
                                                                                "figure fraction", color=th["text"], lw=1.1,
                                                                                arrowstyle="-", shrinkB=6))
                fig.text(tx, ty, f" {txt} ", fontsize=15, fontweight="bold", color=th["text"], ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=th["text"], lw=0.8, alpha=0.95))
        o_ = Path(out)
        o_ = o_.with_name(o_.stem + ("_labeled" if labeled else "") + o_.suffix)
        o_.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o_, dpi=dpi, transparent=True)
        plt.close(fig)
        outs.append(str(o_))
        print("wrote", o_)
    return dict(episode=int(episode), step=int(k), time=round(k * q.dt, 2), chaser=side, files=outs, cam=cam)


def pick_random(q, n=6, seed=0, pool=None):
    """n evaluation scenes as different as possible: greedy farthest points on (obstacle layout, start, target, chaser)."""
    eps = sorted(q.idx) if pool is None else list(pool)
    F = []
    for e in eps:
        j, _ = q.ep(e)
        o = q.obst[j]
        o = o[np.argsort(o[:, 0])]
        f = list((o[:, :2] / [40.0, 20.0]).ravel()) + [float(np.mean(o[:, 2] + o[:, 3] == 0))]
        f += [q.pos[0, j, 1] / 20.0, q.tg[0, j, 1] / 20.0]
        c = int(np.argmax(q.on[j]))
        f += list(q.ch[0, j, c, :2] / [40.0, 20.0])
        F.append(f)
    F = np.asarray(F)
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(eps)))]
    d = np.linalg.norm(F - F[sel[0]], axis=1)
    while len(sel) < n:
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(F - F[i], axis=1))
    return [eps[i] for i in sel]


def col3(spec, outdir, episodes=None, size=(1600, 1000), dpi=100, icon=3.0):
    plt = font_kr()
    from PIL import Image
    q = Run(spec)
    th = sg.THEMES["light"]
    sk = sg.Skins3D("light", "missile", "glass", "vehicle")
    eps = episodes or pick_random(q, 6)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cam = dict(elev=34.0, azim=-115.0, zoom=1.0, xlim=[-1.0, 40.0], ylim=[-10.0, 10.0], zlim=[0.0, 8.0], v_shift=0.0, focal=1.0)
    r_d, L_c = float(q.z["r_drone"]), 2.0 * float(q.z["chaser_radius"])

    def env_frame(ax, e):
        j, _ = q.ep(e)
        static_scene(ax, sk, q, j, cam)
        arts = sk.target3d(ax, q.tg[0, j], sg.heading(q.tg[:, j], 0), 0.0, size=1.3)
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                arts += sk.chaser3d(ax, q.ch[0, j, c], q.pos[0, j] - q.ch[0, j, c], 0.0, length=L_c * icon)
        arts += sk.quad3d(ax, q.pos[0, j], q.quat[0, j] if q.quat is not None else None, radius=r_d * icon)
        return arts

    # grid 2 x 3
    W, H = size[0] / dpi, size[1] / dpi
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_alpha(0.0)
    cw, ch_ = 1 / 3, 0.5
    for i, e in enumerate(eps):
        r, c = divmod(i, 3)
        x0, y0 = c * cw, 1 - (r + 1) * ch_
        side_h = cw * W / H                                                 # square axes, as wide as the cell
        ax = fig.add_axes([x0, y0 + ch_ * 0.60 - side_h / 2, cw, side_h], projection="3d")
        ax.patch.set_alpha(0.0)
        ax.computed_zorder = False
        ax.set_axis_off()
        apply_cam(ax, dict(cam, zoom=0.9))
        env_frame(ax, e)
        apply_cam(ax, dict(cam, zoom=0.9))
        fig.text(x0 + cw / 2, y0 + 0.10, f"환경 {e}", fontsize=15, fontweight="bold", color=th["text"], ha="center")
    fig.text(0.995, 0.005, f"드론 · 요격기 아이콘 ×{icon:g} 확대", fontsize=9, color=th["sub"], ha="right", va="bottom")
    fig.savefig(outdir / "col3_random.png", dpi=dpi, transparent=True)
    plt.close(fig)
    print("wrote", outdir / "col3_random.png")

    # top views 2 x 3
    sk2 = sg.Skins("light", "missile", "glass", "vehicle")
    fig, axs = plt.subplots(2, 3, figsize=(W, H))
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.06, wspace=0.04, hspace=0.18)
    for ax, e in zip(axs.ravel(), eps):
        j, _ = q.ep(e)
        ax.set_facecolor(th["bg"])
        ax.set_xlim(-1, 41), ax.set_ylim(-11, 11), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.plot([0, 40, 40, 0, 0], [-10, -10, 10, 10, -10], color=th["wall"], lw=1.6)
        for o in q.obst[j]:
            if o[6] > 0.5:
                sk2.building(ax, o)
        sk2.target(ax, q.tg[0, j], sg.heading(q.tg[:, j], 0), 0.0)
        for c in range(q.ch.shape[2]):
            if q.on[j, c] > 0.5:
                d = q.pos[0, j, :2] - q.ch[0, j, c, :2]
                sk2.chaser(ax, q.ch[0, j, c], float(np.arctan2(d[1], d[0])), 0.0)
        sk2.quad(ax, q.pos[0, j], sg.quat_yaw(q.quat[0, j]) if q.quat is not None else 0.0)
        ax.set_title(f"환경 {e}", fontsize=15, fontweight="bold", color=th["text"], y=-0.13)
    fig.savefig(outdir / "col3_random_top.png", dpi=dpi, transparent=True)
    plt.close(fig)
    print("wrote", outdir / "col3_random_top.png")

    # cross-fading GIF (white background): each scene 1 s, 0.2 s fades
    frames = []
    for e in eps:
        fig = plt.figure(figsize=(W, H), facecolor=WHITE)
        ax = scene_axes(fig, cam)
        env_frame(ax, e)
        apply_cam(ax, cam)
        fig.text(0.5, 0.06, f"환경 {e}", fontsize=24, fontweight="bold", color=th["text"], ha="center")
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)
    fps, hold_n, fade_n = 20, 16, 4
    seq = []
    for i, im in enumerate(frames):
        seq += [im] * hold_n
        nxt = frames[(i + 1) % len(frames)]
        seq += [Image.blend(im, nxt, (f + 1) / (fade_n + 1)) for f in range(fade_n)]
    gif = outdir / "col3_random.gif"
    seq[0].save(gif, save_all=True, append_images=seq[1:], duration=int(1000 / fps), loop=0, optimize=False)
    sg.finish(gif, mp4=False)
    print("wrote", gif)
    return dict(episodes=[int(e) for e in eps], cam=cam, icon_scale=icon)


# ------------------------------------------------------------------------------------------------ paper-setting check
def check(a_dir, b_dir, root, task):
    """Section 0 (paper settings) from the runs' own records: prints every item, returns (ok, lines)."""
    import inspect
    from ps2rl.cil.strike_filter import make_filter
    lines, ok = [], True

    def item(name, value, good):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"[{'OK' if good else 'MISMATCH'}] {name}: {value}")

    root = Path(root)
    res = json.loads((root / "residual/striker_rs0p1_s0/configs.json").read_text())
    ps2 = json.loads((root / "ps2/striker_rs0p1_s0/configs.json").read_text())
    p1 = json.loads((root / "p1/td3_a0p2_seed0/configs.json").read_text())
    tk = json.loads(Path(task).read_text())
    item("task file", Path(task).name, Path(task).name == "strike_final_ps2.json")
    item("A striker", "residual/striker_rs0p1_s0 (best_weights), trained without filter: filter_backup_run "
         f"'{res['tracker'].get('filter_backup_run', '')}', w_filter {res['tracker'].get('w_filter')}",
         res["tracker"].get("filter_backup_run", "") == "" and float(res["tracker"].get("w_filter", 0)) == 0.0
         and (root / "residual/striker_rs0p1_s0/best_weights.pkl").exists())
    item("B striker", f"ps2/striker_rs0p1_s0, trained through filter {ps2['tracker'].get('filter_guidance')}",
         ps2["tracker"].get("filter_guidance") == tk["tracker"]["filter_guidance"] and float(ps2["tracker"].get("w_filter", 0)) > 0)
    item("backup (Phase I)", f"p1/td3_a0p2_seed0, TD3, {p1['sa']['total_steps']:,} steps", int(p1["sa"]["total_steps"]) == 10_000_000)
    item("chaser (backup model)", f"r_c {p1['strike']['chaser_radius']} m, v_c {p1['strike']['chaser_speed']} m/s",
         float(p1["strike"]["chaser_radius"]) == 0.4 and float(p1["strike"]["chaser_speed"]) == 2.0)
    for tag, c in (("A striker", res), ("B striker", ps2)):                  # the chaser each striker was trained against
        e_ = c.get("env", {})
        r_, v_, a_ = e_.get("actual_chaser_radius"), e_.get("actual_chaser_speed"), e_.get("chaser_accel", 2.0)
        item(f"{tag} training chaser", f"actual radius {r_}, actual speed {v_} (None / -1: 0.4 m, 2 m/s), accel {a_}, "
             f"chaser_speed_scale {e_.get('chaser_speed_scale', 1.0)}",
             (r_ is None or float(r_) < 0) and (v_ is None or float(v_) < 0) and float(a_ if a_ is not None else 2.0) == 2.0
             and float(e_.get("chaser_speed_scale", 1.0) or 1.0) == 1.0)
    lam = inspect.signature(make_filter).parameters["lambdas"].default
    for tag, d in (("A", a_dir), ("B", b_dir)):
        t = json.loads((Path(d) / "strike_filter_table.json").read_text())
        z = np.load(Path(d) / "strike_filter_traj.npz", allow_pickle=True)
        env = json.loads(t["env_overrides"]) if t.get("env_overrides") else {}
        pu = t.get("pursuit", {})
        item(f"{tag} chaser (simulated)", f"actual radius {env.get('actual_chaser_radius', -1)} (-1: 0.4 m), actual speed "
             f"{env.get('actual_chaser_speed', -1)} (-1: 2 m/s), accel {env.get('chaser_accel', 2.0)} m/s^2, filter assumes "
             f"r_c {float(z['chaser_radius'])} m, v_c {float(z['chaser_speed'])} m/s",
             float(env.get("actual_chaser_radius", -1)) < 0 and float(env.get("actual_chaser_speed", -1)) < 0
             and float(env.get("chaser_accel", 2.0)) == 2.0 and float(z["chaser_radius"]) == 0.4 and float(z["chaser_speed"]) == 2.0)
        item(f"{tag} scenes", f"{t['episodes']} evaluation scenes, seed {t['scene_seed']}", t["episodes"] == 1000 and t["scene_seed"] == 9001)
        item(f"{tag} nominal pursuit options", f"detour {pu.get('detour_radius')}, heading search {pu.get('search')}",
             float(pu.get("detour_radius", 0)) == 0.0 and not pu.get("search", False))
        if tag == "B":
            g = t["guidance"]
            labels = [str(x) for x in z["variant_labels"]]
            item("B filter", f"{len(labels)} backup variants (j = 0..{len(labels) - 1}) x lambda {list(lam)}, horizon {g['horizon']} s, "
                 f"committed prediction {g['committed_prediction']}", len(labels) == 7 and list(lam) == [1.0, 0.75, 0.5, 0.25, 0.0]
                 and not g.get("lambda_grid") and float(g["horizon"]) == 3.0 and g["committed_prediction"])
            item("B filter, off", f"early-escape variants {any(len(v) > 2 for v in g['escape_variants'])}, zero-slack fallback "
                 f"{g['fallback_zero_slack']}, lam_rise_max {g['lam_rise_max']}",
                 not any(len(v) > 2 for v in g["escape_variants"]) and not g["fallback_zero_slack"] and float(g["lam_rise_max"]) == 0.0)
            for i, l_ in enumerate(labels):
                lines.append(f"       j = {i}: {l_}")
    e = tk["env"]
    item("environment", f"arena {2 * e['arena_y_half']:g} x {e['arena_x']:g} m... {e['arena_x']:g} x {2 * e['arena_y_half']:g} m, "
         f"obstacles {e['obstacles_min']}-{e['obstacles_max']}, chasers active {e['chasers_active_min']}-{e['chasers_active_max']}, "
         f"{e['episode_seconds']:g} s, strike radius {e['strike_radius']} m",
         e["arena_x"] == 40.0 and e["arena_y_half"] == 10.0 and e["obstacles_min"] == e["obstacles_max"] == 3
         and e["chasers_active_min"] == e["chasers_active_max"] == 1 and e["episode_seconds"] == 15.0 and e["strike_radius"] == 0.6)
    lines[-1] = lines[-1].replace(f"arena {2 * e['arena_y_half']:g} x {e['arena_x']:g} m... ", "arena ")
    return ok, lines


def reproduce(a_dir, b_dir, episodes):
    """Recorded runs against the paper's GPU per-episode records (outcome and end step of every simulated episode)."""
    ga, gb, dt = gpu_records()
    lines, ok = [], True
    for tag, d, g, mode in (("A", a_dir, ga, "none"), ("B", b_dir, gb, "learned")):
        r = json.loads((Path(d) / "strike_filter_episodes.json").read_text())["modes"][mode]
        n = len(r["outcome"])
        idx = list(range(n)) if n == 1000 else None
        if idx is None:
            ok = False
            lines.append(f"[NOT CHECKED] {tag}: {n} episodes recorded, not the full 1000-episode evaluation (an --only_episodes "
                         "replay); run docker_batch/run_kafa_intro.sh on the GPU for the exact check")
            continue
        same_o = sum(int(r["outcome"][i] == g["outcome"][i]) for i in idx)
        same_s = sum(int(r["end_step"][i] == g["end_step"][i]) for i in idx)
        sel = [(e, OC[r["outcome"][e]], r["end_step"][e] * dt, OC[g["outcome"][e]], g["end_step"][e] * dt) for e in episodes]
        good = all(x[1] == x[3] and abs(x[2] - x[4]) < 1e-6 for x in sel)
        ok &= good
        lines.append(f"[{'OK' if good else 'DIFFERS'}] {tag}: 1000 episodes, outcome equal {same_o}/1000, end step equal {same_s}/1000; "
                     + "; ".join(f"{e}: {o} {t:.2f} s (paper {o2} {t2:.2f} s)" for e, o, t, o2, t2 in sel))
    return ok, lines


# ------------------------------------------------------------------------------------------------ report
def report(out, obs, cha, preview=False):
    """README_intro.md: report table, picks and reasons, settings check, exact commands and commit."""
    out = Path(out)
    scr = json.loads((out / "screen.json").read_text())
    pick = {m["episode"]: m for m in scr["obstacle"] + scr["chaser"]}
    rows = []
    for f in sorted(out.glob("intro_*.json")):
        i = json.loads(f.read_text())
        cfg = "A = 잔차 정책, 필터 없음" if i["mode"] == "none" else "B = PS2 + 최종 필터"
        rows.append(f"| {i['file']} | {i['episode']} | 예 (seed 9001) | {cfg} | {i['outcome']} {i['time']:.2f} s | {i['min_hc']:.2f} | "
                    f"{i.get('act_frac', '-')} | {','.join(map(str, i.get('j_used', []))) or '-'} | {i.get('no_cert_steps', '-')} |")
    c2 = json.loads((out / "col2.json").read_text()) if (out / "col2.json").exists() else {}
    c3 = json.loads((out / "col3.json").read_text()) if (out / "col3.json").exists() else {}
    cmds = (out / "commands.sh").read_text() if (out / "commands.sh").exists() else "(see docker_batch/run_kafa_intro.sh)"
    chk = (out / "check.txt").read_text() if (out / "check.txt").exists() else ""
    commit = (out / "commit.txt").read_text().strip() if (out / "commit.txt").exists() else "?"
    root = (out / "root.txt").read_text().strip() if (out / "root.txt").exists() else "results/strike_reduced/final/run"
    mo, mc = pick.get(obs, {}), pick.get(cha, {})
    shortlist = (out / "shortlist.md").read_text() if (out / "shortlist.md").exists() else ""
    lines = [
        "# KAFA 발표 서론 그림 (논문 설정)", "",
        ("> **CPU 미리보기**: 후보 선별용 CPU 재생으로 만든 그림. 최종본은 `docker_batch/run_kafa_intro.sh`의 GPU EXACT 기록으로 다시 만든다." if preview
         else "> GPU EXACT: 평가 1,000개를 전부 시뮬레이션하면서 고른 에피소드만 기록 (`--record_episodes`), 논문 GPU 기록과 에피소드별로 대조."), "",
        f"- 커밋: {commit}",
        f"- 모델: `{root}` seed 0 — A 잔차 striker (필터 없이 학습, best), B PS2 striker (최종 필터를 거쳐 학습), "
        "회복 정책 Phase I TD3 1,000만 스텝",
        "- 필터: 7개 회복 정책 (j = 0…6) × λ ∈ {1, 0.75, 0.5, 0.25, 0}, 3 s 예측, committed prediction. heading search · early escape · "
        "zero-slack fallback · λ ramp · chaser detour 모두 OFF (`docker_batch/configs/strike_final_ps2.json`)",
        "- 요격기: 반경 0.4 m, 속력 2 m/s, 가속도 2 m/s². 환경: 40 × 20 m, 장애물 3 · 요격기 1, 15 s, 성공 = 목표 0.6 m 이내", "",
        "## 결과 표", "",
        "| 파일 | 환경 | 평가 세트 | 구성 | 결과 · 시간 | min h_c [m] | 필터 개입 비율 | 사용된 j | 인증 상실 스텝 |",
        "|---|---|---|---|---|---|---|---|---|", *rows, "",
        "- min h_c: 요격기 여유도 $h_c = R-(r_c+r_d)-s\\delta-\\frac{s^2}{2a_e}$ ($s=\\max(v_c-e^\\top v,0)$, δ = 0.2 s, a_e = 5 m/s²)의 에피소드 최솟값. "
        "B의 ≈ 0.2 m는 필터가 새 계획에 요구하는 여유 (fresh margin 0.2 m)에서 정해짐.",
        "- 필터 개입 비율: λ < 1 이거나 committed plan을 따른 스텝의 비율. 사용된 j: 그 스텝들에서 적용된 회복 정책.", "",
        "## 선정", "", shortlist,
        f"**(i) 장애물 충돌: 환경 {obs}.** A가 {mo.get('tA', '?')} s에 상자형 장애물에 충돌. B는 같은 장애물 면을 따라 "
        f"{mo.get('B_min_sdf_hit', '?')} m 간격 (드론 반경 0.3 m + 필터 장애물 여유 0.15 m)으로 지나가며 A의 충돌 지점 "
        f"{mo.get('B_pass_crash', '?')} m 옆에서 필터가 개입한 뒤 {mo.get('tB', '?')} s에 타격. 요격기는 멀리 있어 (min h_c {mo.get('min_hc', '?')} m) "
        "장애물 이야기만 보이고, 상자형 장애물이라 (ii)의 원기둥 배치와 확연히 다르다. 상위 후보들 (점수 차 < 0.1)의 차이는 이 두 점.", "",
        f"**(ii) 요격기 충돌: 환경 {cha}.** A가 {mc.get('tA', '?')} s 동안 접근한 요격기와 충돌. B는 min h_c {mc.get('min_hc', '?')} m까지 "
        f"접근한 뒤 감속 ({mc.get('B_slow_at_min_hc', '?')} m/s) · 방향 전환 ({mc.get('B_turn_at_min_hc', '?')}°)으로 빠져나가 "
        f"(회피 정책 j = {','.join(str(x) for x in mc.get('j_used', []) if x >= 2)}) {mc.get('tB', '?')} s에 타격. 가까이 왔다가 감속 · 방향 전환으로 "
        "벗어나는 장면이 후보 중 가장 분명하고, 인증 상실 없음.", "",
        "참고: 평가 세트의 A 요격기 충돌 13개는 모두 정면 요격 (충돌 1.5 s 전 요격기 방위가 드론 진행 방향의 20° 이내)이다. 요격기가 드론과 목표 사이에서 출발하고 "
        "속력 2 m/s라 5–6 m/s로 나는 드론을 뒤에서 따라잡지 못하기 때문. 그래서 col2는 '뒤에서 쫓는' 대신 '다가오는' 순간 "
        f"(t = {c2.get('time', '?')} s, 거리 약 3 m)을 썼다.", "",
        "## 그림", "",
        "- `intro_obs_A/B.gif`, `intro_cha_A/B.gif`: 1280 × 768, 3D, 한 run씩, 실시간 1× (25 fps), 마지막 프레임 2 s 유지 후 반복; 같은 이름의 PNG (300 dpi, 3840 × 2304) · MP4. "
        "A · B는 같은 고정 카메라 (`cam_obs.json`, `cam_cha.json`). 드론 (회전익 끝 반경 0.3 m)과 요격기 (길이 2 r_c = 0.8 m)는 실제 크기.",
        f"- `col1_uav.png`: 렌더러 드론 클로즈업. `col2_scene(_labeled).png`: 환경 {cha}, A의 t = {c2.get('time', '?')} s, 지난 1.5 s 궤적.",
        f"- `col3_random.png` · `col3_random.gif` · `col3_random_top.png`: 평가 세트 환경 {', '.join(map(str, c3.get('episodes', [])))} "
        "(장애물 배치 · 시작점 · 목표 · 요격기 특징의 최원점 선택), t = 0. 3D 격자에서는 드론 · 요격기 아이콘 ×3 확대 (그림에 표기).",
        "- `shortlist.png`: 유형별 상위 3개, A (회색 점선, ✗ 충돌) · B (파랑) top view.", "",
        "## 설정 확인 (로그)", "", "```", chk.strip(), "```", "",
        "## 명령", "", "```bash", cmds.strip(), "```", ""]
    (out / "README_intro.md").write_text("\n".join(lines))
    print("wrote", out / "README_intro.md")


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--a", required=True, help="A replay DIR (residual striker, mode none)")
    s.add_argument("--b", action="append", required=True, help="B replay DIR(s) (PS2 striker, mode learned)")
    s.add_argument("--out", required=True)
    c = sub.add_parser("cam")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--episode", type=int, required=True)
    c.add_argument("--type", choices=("obstacle", "chaser"), required=True)
    c.add_argument("--elev", type=float, default=28.0)
    g = sub.add_parser("gif")
    g.add_argument("--traj", required=True, help="DIR:MODE")
    g.add_argument("--episode", type=int, required=True)
    g.add_argument("--cam", required=True, help="camera JSON file (the same for A and B)")
    g.add_argument("--label", required=True)
    g.add_argument("--chip", type=int, default=0)
    g.add_argument("--speed", type=float, default=1.0)
    g.add_argument("--out", required=True)
    g.add_argument("--mp4", type=int, default=1)
    c1 = sub.add_parser("col1")
    c1.add_argument("--out", required=True)
    c2 = sub.add_parser("col2")
    c2.add_argument("--traj", required=True)
    c2.add_argument("--episode", type=int, required=True)
    c2.add_argument("--step", type=int, default=None)
    c2.add_argument("--cam", default="", help="camera JSON (e.g. the motivation GIF's); refitted to 1600 x 1000 with --fit_b")
    c2.add_argument("--fit_b", default="", help="B replay DIR of the same scene: refit the camera to A and B for 16:10")
    c2.add_argument("--out", required=True)
    ck = sub.add_parser("check", help="section 0: paper settings from the recorded runs; reproduction of the paper's records")
    ck.add_argument("--a", required=True)
    ck.add_argument("--b", required=True)
    ck.add_argument("--root", default=str(REPO / "results/strike_reduced/final/run"))
    ck.add_argument("--task", default=str(REPO / "docker_batch/configs/strike_final_ps2.json"))
    ck.add_argument("--episodes", default="957,146")
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--obs", type=int, required=True)
    rp.add_argument("--cha", type=int, required=True)
    rp.add_argument("--preview", type=int, default=0)
    c3 = sub.add_parser("col3")
    c3.add_argument("--traj", required=True)
    c3.add_argument("--episodes", default="")
    c3.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "screen":
        screen(a.a + ":none", [b + ":learned" for b in a.b], a.out)
    elif a.cmd == "cam":
        print(json.dumps(auto_cam(Run(a.a + ":none"), Run(a.b + ":learned"), a.episode, a.type, a.elev)))
    elif a.cmd == "gif":
        render_gif(a.traj, a.episode, a.out, json.loads(Path(a.cam).read_text()), a.label, bool(a.chip), a.speed, mp4=bool(a.mp4))
    elif a.cmd == "col1":
        col1(a.out)
    elif a.cmd == "check":
        ok, lines = check(a.a, a.b, a.root, a.task)
        ok2, lines2 = reproduce(a.a, a.b, [int(x) for x in a.episodes.split(",")])
        print("\n".join(lines + lines2))
        nc = any(x.startswith("[NOT CHECKED]") for x in lines2)
        print("[check] paper settings " + ("OK" if ok else "MISMATCH") + "; reproduction of the paper records "
              + ("OK" if ok2 else ("NOT CHECKED (CPU replay)" if nc else "DIFFERS")))
    elif a.cmd == "report":
        report(a.out, a.obs, a.cha, bool(a.preview))
    elif a.cmd == "col2":
        cam = json.loads(Path(a.cam).read_text()) if a.cam else None
        if cam is not None and a.fit_b:
            qa = Run(a.traj)
            cam = fit_cam(cam, key_points(qa, Run(a.fit_b + ":learned"), a.episode), (1600, 1000), margin=(0.04, 0.08))
        r = col2(a.traj, a.episode, a.out, a.step, cam=cam)
        Path(a.out).with_name("col2.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(json.dumps(r, ensure_ascii=False))
    else:
        eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None
        r = col3(a.traj, a.out, eps)
        (Path(a.out) / "col3.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r))


if __name__ == "__main__":
    main()
