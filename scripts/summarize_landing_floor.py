"""Morning report for run_landing_floor_overnight.sh: $ROOT/REPORT.md (+ REPORT_*.png).

Reads what the stages left behind (any subset is fine):
  phase1/<run>/{summary.json, landing_backup_policy_actor.pkl}   Phase-I variants
  checks/<run>/check_landing_bcbf.json                           self-checks
  tests/<run or baseline>/floor_tests.json                       reach tests

The recommendation is the (backup, filter setting) pair that is safe in every test
(no cone exit, no floor penetration beyond --tol, slack <= --slack_tol, no QP fallbacks,
self-check PASS) and then reaches the floor best: touchdown rate on the cone-cutting
tracking task, then on the LQR landings, then the lowest allowed hover height, then Phase-I mu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _f(x, spec=".3f", none="–"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return none
    return format(x, spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/landing_floor")
    ap.add_argument("--phase1_dir", default=None, help="Phase-I runs (default: <root>/phase1)")
    ap.add_argument("--tol", type=float, default=1e-3, help="tolerated negative margin [m]")
    ap.add_argument("--slack_tol", type=float, default=1e-2)
    args = ap.parse_args(argv)
    root = Path(args.root)
    lines = ["# Floor-aware landing: overnight report", ""]

    # ---------------------------------------------------------------- Phase I
    p1 = {}
    for d in sorted((Path(args.phase1_dir) if args.phase1_dir else root / "phase1").glob("*")):
        s = _load(d / "summary.json")
        if not s or not (d / "landing_backup_policy_actor.pkl").exists():
            continue
        t = s.get("test_at_best", {})
        chk = _load(root / "checks" / d.name / "check_landing_bcbf.json")
        p1[d.name] = {
            "mu": t.get("mu_weighted"), **{r: t.get(r, {}).get("m_hat") for r in ("general", "edge", "shell")},
            "crash": t.get("crash_rate"), "minutes": s.get("wall_time_sec", float("nan")) / 60.0,
            "check": chk,
            "cert": _load(d / "certificate.json"),
        }
    if p1:
        lines += ["## Phase I variants (floor in S; certified before training)", "",
                  "| run | test μ_w | general | edge | shell | crash | certificate (min bound) | self-check | cone stress min h | floor stress min height | max slack |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in p1.items():
            chk = r["check"] or {}
            cert = r["cert"] or {}
            cmin = min((v for k, v in cert.items() if k.startswith("c_") and isinstance(v, (int, float))), default=None)
            ok = chk and chk.get("recoverability", {}).get("ok") and chk.get("filter_stress_test", {}).get("ok") \
                and chk.get("floor_stress_test", {"ok": True}).get("ok")
            lines.append(
                f"| {n} | {_f(r['mu'])} | {_f(r['general'])} | {_f(r['edge'])} | {_f(r['shell'])} | {_f(r['crash'])} | "
                f"{_f(cmin, '.2f')} ≥ c_B | {'PASS' if ok else ('FAIL' if chk else '–')} | "
                f"{_f(chk.get('filter_stress_test', {}).get('filtered_min_h'), '+.4f')} | "
                f"{_f(chk.get('floor_stress_test', {}).get('filtered_min_height'), '+.4f')} | "
                f"{_f(chk.get('filter_stress_test', {}).get('max_slack'), '.1e')} |")
        lines += ["", "μ is measured with Phase I's own failure test (S, plus the recovery envelope when on), so variants with the envelope report a smaller μ by construction.", ""]

    # ---------------------------------------------------------------- tests
    rows = []
    for d in sorted((root / "tests").glob("*")):
        rep = _load(d / "floor_tests.json")
        if not rep:
            continue
        for name, e in rep["settings"].items():
            c = e.get("lqr_centre_all", {})
            tr = e.get("tracking", {}).get("filtered_in_CN", {}) or {}
            safe_vals = [c.get("min_h_cone"), c.get("min_height"), tr.get("min_h_cone"), tr.get("min_height")]
            safe = all(v is not None and v >= -args.tol for v in safe_vals if v is not None) and \
                max(c.get("max_slack", 0.0), tr.get("max_slack", 0.0)) <= args.slack_tol and \
                c.get("qp_fallbacks", 0) + tr.get("qp_fallbacks", 0) == 0
            if e.get("floor_constraint") is False:
                safe = False  # no floor in S: floor penetration is not prevented
            if d.name in p1:  # a trained run must also pass its self-check
                chk = p1[d.name].get("check")
                safe = safe and chk is not None and bool(chk.get("filter_stress_test", {}).get("ok")) and \
                    bool(chk.get("floor_stress_test", {"ok": True}).get("ok"))
            rows.append({
                "backup": d.name, "setting": name, "over": e.get("overrides"),
                "hover_low": e.get("hover", {}).get("lowest_feasible_height_on_axis"),
                "hover_frac": e.get("hover", {}).get("feasible_fraction_lower_half"),
                "lqr_td": c.get("touchdown_rate"), "lqr_t": (c.get("touchdown_time") or {}).get("median"),
                "lqr_n": c.get("n"), "lqr_minh": c.get("min_h_cone"), "lqr_minz": c.get("min_height"),
                "trk_td": tr.get("touchdown_rate"), "trk_t": (tr.get("touchdown_time") or {}).get("median"),
                "trk_minh": tr.get("min_h_cone"), "trk_minz": tr.get("min_height"),
                "trk_unf_out": e.get("tracking", {}).get("unfiltered_leaves_cone_rate"),
                "slack": max(c.get("max_slack", 0.0), tr.get("max_slack", 0.0)),
                "fallbacks": c.get("qp_fallbacks", 0) + tr.get("qp_fallbacks", 0),
                "mu": p1.get(d.name, {}).get("mu"), "safe": safe,
                # relative-time rows everywhere = the paper's form: pi_b is always a feasible QP point
                "guaranteed": all(e.get("relative_time_per_constraint") or [True]),
                "safeguard": e.get("discrete_safeguard"),
                "sg_rate": max(v for v in (c.get("safeguard_active_rate"), tr.get("safeguard_active_rate"), 0.0) if v is not None),
                "zt": rep.get("zeta_touch"), "vt": rep.get("v_touch"),
            })
    if rows:
        zt, vt = rows[0]["zt"], rows[0]["vt"]
        lines += ["## Reaching the floor through the filter", "",
                  f"Touchdown means height ≤ {zt} m over the pad disk at speed ≤ {vt} m/s. T1 is the LQR aimed at the pad centre (all starts). "
                  "T3 is an LQR tracking a reference that cuts outside the cone and ends at the pad centre (starts in C_N). "
                  "The hover column is the lowest height over the pad centre where holding still is allowed.", "",
                  "| backup | setting | lowest hover [m] | hover-feasible, lower half | T1 touchdown | T1 median time [s] | T3 touchdown | T3 median time [s] | min h_cone (T1/T3) | min height (T1/T3) | T3 tracker alone leaves cone | max slack | safe |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['backup']} | {r['setting']} | {_f(r['hover_low'])} | {_f(r['hover_frac'], '.2f')} | "
                f"{_f(r['lqr_td'], '.2f')} (n={r['lqr_n']}) | {_f(r['lqr_t'], '.2f')} | {_f(r['trk_td'], '.2f')} | {_f(r['trk_t'], '.2f')} | "
                f"{_f(r['lqr_minh'], '+.3f')} / {_f(r['trk_minh'], '+.3f')} | {_f(r['lqr_minz'], '+.3f')} / {_f(r['trk_minz'], '+.3f')} | "
                f"{_f(r['trk_unf_out'], '.2f')} | {_f(r['slack'], '.1e')} | {'yes' if r['safe'] else 'no'} |")
        lines.append("")
        cands = [r for r in rows if r["safe"]]
        key = lambda r: (r["trk_td"] or 0.0, r["lqr_td"] or 0.0, bool(r["guaranteed"]),  # noqa: E731
                         -(r["hover_low"] if r["hover_low"] is not None else 9.0), r["mu"] or 0.0)
        if cands:
            best = max(cands, key=key)
            lines += ["## Recommendation", "",
                      f"* backup **{best['backup']}**, filter setting **{best['setting']}** `{best['over']}`",
                      f"* T3 touchdown {_f(best['trk_td'], '.2f')}, T1 touchdown {_f(best['lqr_td'], '.2f')}, "
                      f"lowest hover {_f(best['hover_low'])} m, min cone margin {_f(min(v for v in (best['lqr_minh'], best['trk_minh']) if v is not None), '+.3f')} m, "
                      f"min height {_f(min(v for v in (best['lqr_minz'], best['trk_minz']) if v is not None), '+.3f')} m, max slack {_f(best['slack'], '.1e')}",
                      "* Next: train Phase II through the CIL with this backup and setting (LANDING_HANDOFF.md).", ""]
        else:
            lines += ["## Recommendation", "", "No (backup, setting) pair passed every safety check; see the table.", ""]

        # figure: touchdown vs lowest hover per setting
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        backups = list(dict.fromkeys(r["backup"] for r in rows))
        settings = list(dict.fromkeys(r["setting"] for r in rows))
        width = 0.8 / max(1, len(backups))
        for i, b in enumerate(backups):
            vals_td = [next((r["trk_td"] if r["trk_td"] is not None else r["lqr_td"] or 0.0
                             for r in rows if r["backup"] == b and r["setting"] == s), np.nan) for s in settings]
            vals_h = [next((r["hover_low"] if r["hover_low"] is not None else np.nan
                            for r in rows if r["backup"] == b and r["setting"] == s), np.nan) for s in settings]
            xs = np.arange(len(settings)) + (i - (len(backups) - 1) / 2) * width
            axs[0].bar(xs, vals_td, width, label=b)
            axs[1].bar(xs, vals_h, width, label=b)
        for ax, ttl in zip(axs, ("touchdown rate (T3 tracking; T1 if no T3)", "lowest allowed hover height over the pad centre [m]")):
            ax.set_xticks(np.arange(len(settings)))
            ax.set_xticklabels(settings, rotation=20, fontsize=8)
            ax.set_title(ttl, fontsize=10)
            ax.grid(alpha=0.3, axis="y")
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "REPORT_touchdown.png", dpi=130)
        plt.close(fig)
        lines += ["![touchdown](REPORT_touchdown.png)", "",
                  "Per-backup figures: `tests/<backup>/hover_maps.png` and `tests/<backup>/setting_<name>.png`.", ""]

    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
