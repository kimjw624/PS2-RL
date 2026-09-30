"""Table/macro stage of the landing Phase-I report.

Every number that appears in the report is emitted here from the cached data, as
booktabs tables (``tables/*.tex``) and ``numbers.tex`` macros, so nothing is
transcribed by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

REGIONS = ("general", "edge", "shell")


def bounds_at(d: dict, z: float) -> dict[str, float]:
    """Uncapped z-dependent level bounds at an arbitrary hover height."""
    from ps2rl.sets import landing_certificate as lc
    from ps2rl.sets.quadrotor_cone_sets import QuadrotorConeSafeSet

    cfg = d["cfg"]
    cone = QuadrotorConeSafeSet(r0=cfg["cone_r0"], theta_deg=cfg["cone_theta_deg"], eps=cfg["cone_eps"],
                                pad_x=cfg["pad_x"], pad_y=cfg["pad_y"], pad_z=cfg["pad_z"])
    p_inv = np.linalg.inv(d["lqr"]["P"])
    return {"z_des": float(z),
            "c_cone_exact": lc.c_cone_exact(p_inv, cone, z_des=z, c_hi=100.0),
            "c_cone_note": lc.c_cone_note(p_inv, cone, z_des=z, c_hi=100.0),
            "c_ground": lc.c_ground_bound(p_inv, z_des=z, z_clear=cfg["z_clear"])}


def pretty(run: str) -> str:
    if "_seed" in run:
        bb, seed = run.split("_seed")
        return f"{bb.upper()}, seed {seed}"
    return run.replace("_", " ")


def f(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def pm(vals: list[float], nd: int = 3) -> str:
    v = np.asarray(vals, dtype=float)
    if len(v) < 2:
        return f(float(v.mean()), nd)
    return f"{v.mean():.{nd}f} $\\pm$ {v.std(ddof=1):.{nd}f}"


def table(colspec: str, header: list[str], rows: list[list[str]], *, midrules: tuple[int, ...] = ()) -> str:
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append("\\midrule")
        out.append(" & ".join(r) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def entry_index(tr: dict, N: int) -> int | None:
    hit = np.where(tr["inB"][1:])[0]
    fail = np.where(tr["h"][1:] < 0)[0]
    tb = hit[0] + 1 if hit.size else None
    tf = fail[0] + 1 if fail.size else None
    if tb is not None and tb <= N and (tf is None or tb < tf):
        return int(tb)
    return None


def case_summary(d: dict, tr: dict) -> dict[str, Any]:
    cfg = d["cfg"]
    N, dt = cfg["num_steps"], cfg["dt"]
    tb = entry_index(tr, N) if "inB" in tr else None
    fail = np.where(tr["h"][1:] < 0)[0]
    tf = int(fail[0] + 1) if fail.size else None
    end = tb if tb is not None else (tf if tf is not None else N)
    seg_u = tr["u"][:end]
    return {
        "outcome": "arrives" if tb is not None else ("crash" if tf is not None else "timeout"),
        "t": (tb if tb is not None else (tf if tf is not None else N)) * dt,
        "hmin": float(tr["h"][1 : end + 1].min()) if end > 0 else float(tr["h"][0]),
        "zmin": float(tr["x"][: end + 1, 2].min() - cfg["pad_z"]),
        "amax": float(seg_u[:, 0].max()) if len(seg_u) else float("nan"),
        "amin": float(seg_u[:, 0].min()) if len(seg_u) else float("nan"),
        "wmax": float(np.abs(seg_u[:, 1:]).max()) if len(seg_u) else float("nan"),
        "tilt_max": float(tr["tilt"][: end + 1].max()) if "tilt" in tr else float("nan"),
    }


def make_tables(d: dict) -> tuple[dict[str, str], dict[str, str]]:
    cfg, rc = d["cfg"], d["region_cfg"]
    cert = d["certificate"]
    lqr = d["lqr"]
    run, cmp_run = d["run"], d["compare_run"]
    at_z = bounds_at(d, cfg["z_des"])
    runs = d["runs"]
    S = d["summaries"]
    RT = d["region_tables"]
    T: dict[str, str] = {}
    M: dict[str, str] = {}

    # ---------------------------------------------------------------- parameters
    T["params"] = table("llr", ["Symbol", "Meaning", "Value"], [
        [r"$\Delta$", "sample period (explicit Euler)", f"{cfg['dt']:.2f} s"],
        [r"$N$, $T$", "backup horizon", f"{cfg['num_steps']} steps, {cfg['T']:.1f} s"],
        [r"$g$", "gravity", f"{cfg['gravity']:.2f} m/s$^2$"],
        [r"$[a_{\min}, a_{\max}]$", "collective thrust box", f"[{cfg['a_cmd_min']:.0f}, {cfg['a_cmd_max']:.2f}] m/s$^2$ ($4g$)"],
        [r"$\bar\omega$", "body-rate bound (each axis)", f"{cfg['omega_max']:.0f} rad/s"],
        [r"$p_{\mathrm{pad}}$", "pad position", f"({cfg['pad_x']:.0f}, {cfg['pad_y']:.0f}, {cfg['pad_z']:.0f}) m"],
        [r"$r_0$, $\theta$, $\varepsilon$", "cone base radius, half-angle, smoothing", f"{cfg['cone_r0']:.2f} m, {cfg['cone_theta_deg']:.0f}$^\\circ$, {cfg['cone_eps']:.3f} m"],
        [r"$z_{\mathrm{des}}$", "hover height of the equilibrium above the pad", f"{cfg['z_des']:.2f} m"],
        [r"$c_{\mathcal{B}}$", "base-set level", f"{cfg['base_set_c']:.0f}"],
        [r"$z_{\mathrm{clear}}$", "required clearance of $\\mathcal{B}$ above the pad", f"{cfg['z_clear']:.2f} m"],
        [r"$\beta$", "discount factor", f"{d['configs'][run]['backup_ra']['beta']:.2f}"],
    ])

    # --------------------------------------------------------------- LQR weights
    wkeys = [("lqr_q_x", "$q_x$"), ("lqr_q_y", "$q_y$"), ("lqr_q_z", "$q_\\zeta$"), ("lqr_q_vx", "$q_{v_x}$"), ("lqr_q_vy", "$q_{v_y}$"),
             ("lqr_q_vz", "$q_{v_z}$"), ("lqr_q_thetax", "$q_{\\phi_x}$"), ("lqr_q_thetay", "$q_{\\phi_y}$"), ("lqr_q_thetaz", "$q_{\\phi_z}$"),
             ("lqr_r_a_cmd", "$r_a$"), ("lqr_r_omega_x", "$r_{\\omega_x}$"), ("lqr_r_omega_y", "$r_{\\omega_y}$"), ("lqr_r_omega_z", "$r_{\\omega_z}$")]
    powerloop = {"lqr_q_x": None, "lqr_q_y": None, "lqr_q_z": 1.0, "lqr_q_vx": 0.16, "lqr_q_vy": 0.16, "lqr_q_vz": 0.4, "lqr_q_thetax": 0.8,
                 "lqr_q_thetay": 0.8, "lqr_q_thetaz": 0.16, "lqr_r_a_cmd": 0.02, "lqr_r_omega_x": 0.012, "lqr_r_omega_y": 0.012, "lqr_r_omega_z": 0.004}
    rows = []
    for k, sym in wkeys:
        pl = powerloop[k]
        rows.append([sym, "--" if pl is None else f"{pl:g}", f"{cfg[k]:g}" + (r"\,$^\dagger$" if (pl is not None and abs(pl - cfg[k]) > 1e-12) else "")])
    T["lqr_weights"] = table("lcc", ["Weight", "Powerloop (7-D)", "Landing (9-D)"], rows, midrules=(9,))

    K = lqr["K"]
    cols = ["$\\Delta p_x$", "$\\Delta p_y$", "$\\zeta-z_{\\mathrm{des}}$", "$v_x$", "$v_y$", "$v_z$", "$\\phi_x$", "$\\phi_y$", "$\\phi_z$"]
    rn = ["$\\delta a$", "$\\omega_x$", "$\\omega_y$", "$\\omega_z$"]
    T["gain"] = table("l" + "r" * 9, [""] + cols, [[rn[i]] + [f"{K[i, j]:.2f}" if abs(K[i, j]) > 5e-3 else "0" for j in range(9)] for i in range(4)])

    # ------------------------------------------------------------- level bounds
    cu = lqr["c_u_rows"]
    chosen = cert["chosen"]
    pl = d["lqr_powerloop_yaw"]
    T["bounds"] = table("llrr", ["Bound", "Definition", "Landing weights", "Powerloop yaw"], [
        [r"$c_{\mathcal{U}}$", r"$\min_i \bar u_i^2/(K_iP^{-1}K_i^\top)$", f(cert["c_u"], 2), f(pl["c_u"], 2)],
        [r"\quad thrust row", "", f(cu[0], 2), ""],
        [r"\quad $\omega_x$, $\omega_y$ rows", "", f(cu[1], 2), ""],
        [r"\quad $\omega_z$ row", "", f(cu[3], 2), ""],
        [r"$c_{\mathrm{chart}}$", r"$\|\phi\|<2$ on $\mathcal{B}_c$", f(cert["c_chart"], 2), f(pl["c_chart"], 2)],
        [r"$c_{\mathrm{Lyap}}$", r"adversarial one-step decrease", f(cert["c_lyap_adversarial"], 2), r"$\approx 6.7$"],
        [r"$c_{\mathrm{cone}}(z_{\mathrm{des}})$", r"$\min_{\mathcal{B}_c} h_{\mathrm{cone}}\ge 0$ (exact)", f(at_z["c_cone_exact"], 2), ""],
        [r"$c_{\mathrm{ground}}(z_{\mathrm{des}})$", r"$\zeta\ge z_{\mathrm{clear}}$ on $\mathcal{B}_c$", f(chosen["c_ground"], 2), ""],
        [r"$\bar c$", "minimum of the above", f(chosen["c_bar"], 2) + f" ({chosen['binding'].split(' ')[0]})", r"$\le 5.39$"],
        [r"$c_{\mathcal{B}}$", "chosen level", f"{cfg['base_set_c']:.0f} ({cfg['base_set_c'] / cert['c_lyap_adversarial']:.2f}\\,$c_{{\\mathrm{{Lyap}}}}$)", ""],
    ], midrules=(4, 8))

    ext = lqr["extents"]
    T["extents"] = table("lr", ["Quantity (support over $\\mathcal{B}$)", "Value"], [
        [r"horizontal position $\max\|\Delta p\|$", f"{ext['xy']:.3f} m"],
        [r"altitude $\max|\zeta-z_{\mathrm{des}}|$", f"{ext['dz']:.3f} m"],
        [r"lowest point above the pad", f"{cfg['z_des'] - ext['dz']:.3f} m"],
        [r"horizontal velocity $\max|v_x|$", f"{ext['vx']:.3f} m/s"],
        [r"vertical velocity $\max|v_z|$", f"{ext['vz']:.3f} m/s"],
        [r"roll/pitch error $\max|\phi_x|$ (tilt equivalent)", f"{ext['phix']:.3f} ({np.degrees(2 * np.arcsin(min(1.0, ext['phix'] / 2))):.0f}$^\\circ$)"],
        [r"yaw error $\max|\phi_z|$ (angle equivalent)", f"{ext['phiz']:.3f} ({np.degrees(2 * np.arcsin(min(1.0, ext['phiz'] / 2))):.0f}$^\\circ$)"],
        [r"spectral radius $\rho(A_d-B_dK)$", f"{lqr['rho_cl']:.4f}"],
        [r"rank of the controllability matrix", f"{lqr['ctrb_rank']} / 9"],
    ])

    c_rest = min(cert["c_u"], cert["c_chart"], cert["c_lyap_adversarial"])
    rows = []
    sel = [r for r in d["sweep"] if any(abs(r["z_des"] - z) < 1e-6 for z in (0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0))]
    for r in sorted(sel + [at_z], key=lambda q: q["z_des"]):
        if True:
            cb = min(c_rest, r["c_cone_exact"], r["c_ground"])
            binding = "ground" if r["c_ground"] <= min(c_rest, r["c_cone_exact"]) else ("cone" if r["c_cone_exact"] <= c_rest else "Lyapunov")
            rows.append([f"{r['z_des']:.2f}", f(r["c_cone_exact"], 2), f(r["c_cone_note"], 2), f(r["c_ground"], 2), f(cb, 2), binding])
    T["zsweep"] = table("rrrrrl", ["$z_{\\mathrm{des}}$ [m]", "$c_{\\mathrm{cone}}$ exact", "$c_{\\mathrm{cone}}$ corner", "$c_{\\mathrm{ground}}$",
                                   "$\\bar c(z_{\\mathrm{des}})$", "binding"], rows)

    # ----------------------------------------------------------- design region
    T["region"] = table("lll", ["Item", "Symbol", "Value"], [
        ["altitude range of $\\Omega$", "$[0, \\bar\\zeta]$", f"[{rc['zeta_min']:.0f}, {rc['zeta_max']:.0f}] m"],
        ["speed bound", "$\\bar v$", f"{rc['v_max']:.1f} m/s"],
        ["tilt bound", "$\\bar\\vartheta$", f"{rc['tilt_max_deg']:.0f}$^\\circ$"],
        ["yaw bound", "", f"{rc['yaw_max_deg']:.0f}$^\\circ$"],
        ["edge band: margin, altitude cap", "$\\eta$, $\\zeta_{\\mathrm{low}}$", f"{rc['edge_margin']:.2f} m, {rc['edge_zeta_low']:.1f} m"],
        ["edge band: outward-velocity probability", "", f"{rc['edge_outward_prob']:.1f}"],
        ["capture shell", "$c_{\\mathcal{B}}<V\\le\\kappa c_{\\mathcal{B}}$", f"$\\kappa={rc['shell_kappa']:.1f}$"],
        ["mixture at $s=0$ (general/edge/shell)", "", f"{rc['mix_general_low']:.2f} / {rc['mix_edge_low']:.2f} / {rc['mix_shell_low']:.2f}"],
        ["mixture at $s=1$ (general/edge/shell)", "", f"{rc['mix_general_high']:.2f} / {rc['mix_edge_high']:.2f} / {rc['mix_shell_high']:.2f}"],
        ["reference-measure weights $w_r$", "", f"{rc['weight_general']:g} / {rc['weight_edge']:g} / {rc['weight_shell']:g}"],
        ["held-out states per split (general/edge/shell)", "", f"{rc['heldout_general']} / {rc['heldout_edge']} / {rc['heldout_shell']}"],
    ])

    ra = d["configs"][run]["backup_ra"]
    T["hparams"] = table("llll", ["Hyperparameter", "Value", "Hyperparameter", "Value"], [
        ["total env.\\ steps", f"{ra['total_steps'] / 1e6:.0f}M", "parallel environments", f"{ra['num_envs']}"],
        ["replay buffer", f"{ra['replay_size'] / 1e6:.0f}M", "batch size", f"{ra['batch_size']}"],
        ["actor / critic lr", f"{ra['actor_lr']:g} / {ra['critic_lr']:g}", "update every (steps)", f"{ra['update_every']}"],
        ["Polyak $\\tau$", f"{ra['tau']:g}", "policy delay", f"{ra['policy_delay']}"],
        ["networks", f"2$\\times${ra['hidden_size']}, {ra['actor_activation'].upper()} actor", "critic loss", f"Huber ($\\delta={ra['critic_huber_delta']:g}$)"],
        ["action-magnitude penalty", f"{ra['action_smoothness_weight']:g}", "episode cap $k_{\\max}$", f"{ra['episode_max_steps']}"],
        ["TD3 exploration noise", f"$\\sigma={ra['exploration_std']:g}$, clip {ra['exploration_clip']:g}", "SAC target entropy", f"{ra['sac_target_entropy']:g}"],
        ["SAC $\\alpha$ (init, range)", f"{ra['sac_alpha_init']:g}, [{ra['sac_alpha_min']:g}, {ra['sac_alpha_max']:g}]", "curriculum promotion", f"{ra['curriculum_success_threshold']:.1f} over {ra['curriculum_window_episodes']} ep."],
        ["evaluation period", f"{ra['eval_every'] / 1e3:.0f}k steps", "seeds per backbone", f"{sum(1 for r in runs if r.startswith('td3'))}"],
    ])

    # ----------------------------------------------------------------- results
    rows = []
    for r in runs:
        s = S[r]
        t = s["test_at_best"]
        rows.append([pretty(r), f(t["mu_weighted"]), f(t["general"]["m_hat"]), f(t["edge"]["m_hat"]), f(t["shell"]["m_hat"]),
                     f(t["crash_rate"]), f"{s['best_eval_step'] / 1e6:.2f}M", f(s["best_val"]["mu_weighted"]), f(s["final_val"]["mu_weighted"]),
                     f"{s['wall_time_sec'] / 60:.1f}"])
    T["runs"] = table("lrrrrrrrrr", ["Run", "$\\hat\\mu_w$", "general", "edge", "shell", "crash", "best step", "best val", "final val", "min"], rows,
                      midrules=(sum(1 for r in runs if r.startswith("sac")),))

    def mu_w(m: dict[str, float]) -> float:
        w = {"general": rc["weight_general"], "edge": rc["weight_edge"], "shell": rc["weight_shell"]}
        return sum(w[k] * m[k] for k in REGIONS) / sum(w.values())

    unt = S[run]["untrained_val"]
    lq = RT["lqr_only"]
    lq_mu = mu_w({k: lq[k]["m"] for k in REGIONS})
    lq_crash = float(np.mean(np.concatenate([~d["outcomes"]["lqr_only"][k] for k in REGIONS])))
    by = {}
    for bb in ("sac", "td3"):
        rr = [r for r in runs if r.startswith(bb)]
        by[bb] = {k: [S[r]["test_at_best"][k]["m_hat"] for r in rr] for k in REGIONS}
        by[bb]["mu"] = [S[r]["test_at_best"]["mu_weighted"] for r in rr]
        by[bb]["crash"] = [S[r]["test_at_best"]["crash_rate"] for r in rr]
    V = d["viability"]
    oracle = {}
    for reg in REGIONS:
        if reg in V:
            oracle[reg] = 1.0 - V[reg]["failed_by_all"] / V[reg]["n"]
        else:
            oracle[reg] = float(np.mean(np.any(np.stack([d["outcomes"][r][reg] for r in runs]), 0)))
    ceil = {reg: (1.0 - V[reg]["n_nonviable"] / V[reg]["n"]) if reg in V else 1.0 for reg in REGIONS}
    ch = S[run]["test_at_best"]
    T["summary"] = table("lrrrrr", ["Backup policy $\\pi_b$", "$\\hat\\mu_w$", "general", "edge", "shell", "crash"], [
        ["untrained actor (val split)", f(unt["mu_weighted"]), f(unt["general"]["m_hat"]), f(unt["edge"]["m_hat"]), f(unt["shell"]["m_hat"]), f(unt["crash_rate"])],
        ["base controller only ($\\pi_{\\mathcal{B}}$ everywhere)", f(lq_mu), f(lq["general"]["m"]), f(lq["edge"]["m"]), f(lq["shell"]["m"]), f(lq_crash)],
        ["learned, SAC (3 seeds)", pm(by["sac"]["mu"]), pm(by["sac"]["general"]), pm(by["sac"]["edge"]), pm(by["sac"]["shell"]), pm(by["sac"]["crash"])],
        ["learned, TD3 (3 seeds)", pm(by["td3"]["mu"]), pm(by["td3"]["general"]), pm(by["td3"]["edge"]), pm(by["td3"]["shell"]), pm(by["td3"]["crash"])],
        [f"\\textbf{{selected: {pretty(run)}}}", f"\\textbf{{{f(ch['mu_weighted'])}}}", f(ch["general"]["m_hat"]), f(ch["edge"]["m_hat"]), f(ch["shell"]["m_hat"]), f(ch["crash_rate"])],
        ["best of the 6 policies per state (oracle)", f(mu_w(oracle)), f(oracle["general"]), f(oracle["edge"]), f(oracle["shell"]), "--"],
        ["viability ceiling (trajectory optimisation)", f(mu_w(ceil)), f(ceil["general"]), f(ceil["edge"]), f(ceil["shell"]), "--"],
    ], midrules=(2, 5))

    rows = []
    for reg in REGIONS:
        t = RT[run][reg]
        rows.append([reg, f"{t['n']}", f"{f(t['m'])} $\\pm$ {f(t['hw'])}", f(t["crash"]), f(t["timeout"]), f(t["t_mean"], 2), f(t["t_p95"], 2),
                     pct(t["below_pad"], 1)])
    T["chosen"] = table("lrcrrrrr", ["Region", "$M_r$", "$\\hat m_r$ $\\pm$ 95\\% (Hoeffding)", "crash", "timeout", "mean $\\tau_{\\mathcal{B}}$ [s]",
                                     "p95 $\\tau_{\\mathcal{B}}$ [s]", "dips below pad [\\%]"], rows)

    rows = []
    for reg in ("general", "edge"):
        v = V[reg]
        fc = int((~d["outcomes"][run][reg]).sum())
        rows.append([reg, f"{v['n']}", f"{fc}", f"{v['failed_by_all']}", f"{v['failed_by_any']}", f"{v['n_nonviable']}",
                     f"{v['failed_by_all'] - v['n_nonviable']}", f(ceil[reg])])
    T["failures"] = table("lrrrrrrr", ["Region", "$M_r$", f"failed ({pretty(run)})", "failed by all 6", "failed by $\\ge$1",
                                       "no safe traj.\\ found", "shared, viable", "ceiling"], rows)

    A = d["actuator"]
    rows = []
    for name in (cmp_run, run, "lqr_only"):
        a = A[name]
        rows.append([pretty(name) if name != "lqr_only" else "base controller only", pct(a["frac0_thr_sat"]), pct(a["frac0_rate_sat"]),
                     pct(a["frac_thr_max"]), pct(a["frac_thr_zero"]), pct(a["frac_rate_sat"], 2), f(a["mean_du"], 3)])
    T["actuators"] = table("lrrrrrr", ["Policy", "$a$ sat.\\ at $k=0$", "$\\omega$ sat.\\ at $k=0$", "$a\\approx a_{\\max}$", "$a\\approx 0$",
                                        "$\\omega$ sat.", "mean $\\|\\Delta\\tilde u\\|$"], rows)

    rows = []
    for name in (cmp_run, run, "lqr_only"):
        for reg in ("general", "edge"):
            t = RT[name][reg]
            rows.append([(pretty(name) if name != "lqr_only" else "base controller only") if reg == "general" else "", reg,
                         f(t["tilt_p50"], 0), f(t["tilt_p95"], 0), f(t["w_p95"], 1), f(t["a_p95"], 1), f(t["hmin_p5"], 3)])
    T["envelope"] = table("llrrrrr", ["Policy", "region", "peak tilt med [$^\\circ$]", "p95 [$^\\circ$]", "p95 peak $|\\omega|$ [rad/s]",
                                      "p95 peak $a$ [m/s$^2$]", "p5 $\\min h$ [m]"], rows, midrules=(2, 4))
    mac_env = {"tiltMedGen": f(RT[run]["general"]["tilt_p50"], 0), "tiltPninetyfiveGen": f(RT[run]["general"]["tilt_p95"], 0),
               "tiltMedEdge": f(RT[run]["edge"]["tilt_p50"], 0), "tiltPninetyfiveEdge": f(RT[run]["edge"]["tilt_p95"], 0),
               "tiltPninetyfiveGenLQR": f(RT["lqr_only"]["general"]["tilt_p95"], 0), "tiltPninetyfiveEdgeLQR": f(RT["lqr_only"]["edge"]["tilt_p95"], 0)}

    SE = d["sensitivity"]
    rows = []
    for name in [*runs, "lqr_only"]:
        s = SE[name]
        rows.append([pretty(name) if name != "lqr_only" else "base controller only", f(s["jac_med"], 2), f(s["jac_p95"], 2), f(s["phi_med"], 2),
                     f(s["phi_p95"], 2), f(s["phi_max"], 1), f(s["row_p95"], 2)])
    T["sensitivity"] = table("lrrrrrr", ["Policy", "$\\|\\partial\\tilde\\pi/\\partial x\\|$ med", "p95", "$\\max_t\\|\\Phi_t\\|$ med", "p95", "max",
                                         "$\\max_t\\|\\nabla h\\,\\Phi_t\\|$ p95"], rows, midrules=(3, 6))

    # ------------------------------------------------------------------ cases
    rows = []
    case_rows = {}
    for c in d["cases"]:
        fe = c["features"]
        cs = case_summary(d, c["main"])
        case_rows[c["tag"]] = cs
        rows.append([c["tag"], c["region"], f(fe["zeta"], 2), f(fe["r"], 2), f(fe["h"], 3), f(fe["v_out"], 2), f(fe["vz"], 2), f(fe["tilt"], 0),
                     cs["outcome"], f(cs["t"], 2), f(cs["hmin"], 3), f(cs["zmin"], 3), f(cs["amax"], 1), f(cs["wmax"], 1)])
    T["cases"] = table("llrrrrrrlrrrrr", ["", "region", "$\\zeta_0$", "$r_0$", "$h_0$", "$v_{\\mathrm{out}}$", "$v_z$", "tilt$^\\circ$", "outcome",
                                          "$t$ [s]", "$\\min h$", "$\\min\\zeta$", "$\\max a$", "$\\max|\\omega|$"], rows)

    # ---------------------------------------------------------------- macros
    def mac(name: str, val: str) -> None:
        M[name] = val

    mac("muChosen", f(ch["mu_weighted"]))
    mac("mGenChosen", f(ch["general"]["m_hat"]))
    mac("mEdgeChosen", f(ch["edge"]["m_hat"]))
    mac("mShellChosen", f(ch["shell"]["m_hat"]))
    mac("muUntrained", f(unt["mu_weighted"]))
    mac("muLQR", f(lq_mu))
    mac("mEdgeLQR", f(lq["edge"]["m"]))
    mac("mGenLQR", f(lq["general"]["m"]))
    mac("muSAC", pm(by["sac"]["mu"]))
    mac("muTDthree", pm(by["td3"]["mu"]))
    mac("muCeil", f(mu_w(ceil)))
    mac("mEdgeCeil", f(ceil["edge"]))
    mac("mGenCeil", f(ceil["general"]))
    mac("muOracle", f(mu_w(oracle)))
    mac("fracOfCeil", f"{100 * ch['mu_weighted'] / mu_w(ceil):.1f}")
    mac("cU", f(cert["c_u"], 2))
    mac("cChart", f(cert["c_chart"], 2))
    mac("cLyap", f(cert["c_lyap_adversarial"], 2))
    mac("cB", f"{cfg['base_set_c']:.0f}")
    mac("cBratio", f"{cfg['base_set_c'] / cert['c_lyap_adversarial']:.2f}")
    mac("cConeZ", f(at_z["c_cone_exact"], 2))
    mac("cGroundZ", f(chosen["c_ground"], 2))
    mac("zDes", f"{cfg['z_des']:.2f}")
    mac("zSweet", f"{cert['sweet_spot_z_des_at_c_rest']:.2f}")
    mac("cUpl", f(pl["c_u"], 2))
    mac("cChartpl", f(pl["c_chart"], 2))
    mac("rhoCL", f"{lqr['rho_cl']:.4f}")
    mac("containMinH", f(cert["containment_min_h"], 3))
    mac("invMaxV", f(cert["invariance_max_V"], 2))
    mac("invMinH", f(cert["invariance_min_h"], 3))
    mac("extXY", f(ext["xy"], 2))
    mac("extZ", f(ext["dz"], 2))
    mac("extV", f(ext["vx"], 2))
    mac("lowestB", f(cfg["z_des"] - ext["dz"], 2))
    mac("edgeShared", f"{V['edge']['failed_by_all']}")
    mac("edgeAny", f"{V['edge']['failed_by_any']}")
    mac("edgeNonviable", f"{V['edge']['n_nonviable']}")
    mac("genShared", f"{V['general']['failed_by_all']}")
    mac("genNonviable", f"{V['general']['n_nonviable']}")
    mac("edgeFailChosen", f"{int((~d['outcomes'][run]['edge']).sum())}")
    mac("edgeSharedViable", f"{V['edge']['failed_by_all'] - V['edge']['n_nonviable']}")
    mac("genSharedViable", f"{V['general']['failed_by_all'] - V['general']['n_nonviable']}")
    vc = d.get("viability_control", {"n": 0, "n_viable": 0})
    mac("ctrlViable", f"{vc['n_viable']}/{vc['n']}")
    mac("tfMedian", f"{np.median(d['nonviable_tf']):.2f}")
    mac("tfMax", f"{np.max(d['nonviable_tf']):.2f}")
    mac("tMeanGen", f(RT[run]["general"]["t_mean"], 2))
    mac("tMeanEdge", f(RT[run]["edge"]["t_mean"], 2))
    mac("belowPadGen", pct(RT[run]["general"]["below_pad"], 1))
    mac("belowPadEdge", pct(RT[run]["edge"]["below_pad"], 1))
    mac("hwEdge", f(RT[run]["edge"]["hw"], 3))
    for i, (k, nm) in enumerate((("frac0_thr_sat", "ThrSatZero"), ("frac_thr_max", "ThrMax"), ("frac_thr_zero", "ThrZero"), ("frac_rate_sat", "RateSat"))):
        mac(f"td{nm}", pct(A[run][k], 1))
        mac(f"sac{nm}", pct(A[cmp_run][k], 1))
    sl = d["slices"]
    for i, nm in enumerate(("Rest", "Out", "OutDown")):
        mac(f"slice{nm}", pct(sl["learned"][i]["frac_success"], 1))
        mac(f"slice{nm}LQR", pct(sl["lqr_only"][i]["frac_success"], 1))
    for k, v in mac_env.items():
        mac(k, v)
    mac("runChosen", pretty(run))
    mac("runCompare", pretty(cmp_run))
    mac("wallMin", f"{np.mean([S[r]['wall_time_sec'] for r in runs]) / 60:.0f}")
    for tag, cs in case_rows.items():
        mac(f"case{tag}t", f(cs["t"], 2))
        mac(f"case{tag}hmin", f(cs["hmin"], 3))
        mac(f"case{tag}hminabs", f(abs(cs["hmin"]), 3))
        mac(f"case{tag}zmin", f(cs["zmin"], 3))
        mac(f"case{tag}amax", f(cs["amax"], 1))
        mac(f"case{tag}amin", f(cs["amin"], 1))
        mac(f"case{tag}wmax", f(cs["wmax"], 1))
        mac(f"case{tag}tilt", f(cs["tilt_max"], 0))
    for c in d["cases"]:
        for k, v in c["features"].items():
            mac(f"case{c['tag']}f{k.replace('_', '')}", f(v, 2) if k != "tilt" else f(v, 0))
        mac(f"case{c['tag']}fvzabs", f(abs(c["features"]["vz"]), 2))
        if "compare" in c:
            cs2 = case_summary(d, c["compare"])
            mac(f"case{c['tag']}ct", f(cs2["t"], 2))
            mac(f"case{c['tag']}coutcome", cs2["outcome"])
            mac(f"case{c['tag']}camax", f(cs2["amax"], 1))
            mac(f"case{c['tag']}cwmax", f(cs2["wmax"], 1))
        if "opt" in c:
            mac(f"case{c['tag']}opthmin", f(c["opt"]["best_hmin"], 3))
            mac(f"case{c['tag']}opthminabs", f(abs(c["opt"]["best_hmin"]), 3))
    return T, M


def write(d: dict, out: Path) -> None:
    T, M = make_tables(d)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for k, v in T.items():
        (out / "tables" / f"{k}.tex").write_text(v + "\n", encoding="utf-8")
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    (out / "numbers.tex").write_text("% generated by report_tables.py\n" + "\n".join(lines) + "\n", encoding="utf-8")
