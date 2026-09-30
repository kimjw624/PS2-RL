"""Compare several evaluate_landing_phase2_filter.py outputs (e.g. BCBF settings) side by side.

    python scripts/compare_landing_filter_evals.py --dirs outputs/landing_phase2_filter/land_v2_miss_s1__* \\
        --out outputs/landing_phase2_filter/compare_land_v2_miss_s1

Writes compare.md (table over starts in C_N) and compare.png: the filtered trajectories near
the pad for each setting, height over time, and cone margin over time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def label_of(s: dict) -> str:
    st = s["setup"]
    parts = [f"alpha={st['alpha']:g}", "rel-time rows " + ("on" if st.get("include_relative_time_term", True) else "off")]
    if st.get("backup_policy_mode", "learned") != "learned":
        parts.append(f"backup={st['backup_policy_mode']}")
    return ", ".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    runs = []
    for d in args.dirs:
        d = Path(d)
        if not (d / "summary.json").exists() or not (d / "episodes.npz").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        runs.append((d, s, np.load(d / "episodes.npz")))
    if not runs:
        sys.exit("no evaluation directories with summary.json + episodes.npz")
    runs.sort(key=lambda r: (r[1]["setup"].get("backup_policy_mode", "learned") != "learned",
                             not r[1]["setup"].get("include_relative_time_term", True), r[1]["setup"]["alpha"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [("episodes (start in C_N)", lambda v: f"{v['episodes']}"),
            ("episodes leaving the cone [%]", lambda v: f"{100 * v['cone_violation_episode_rate']:.1f}"),
            ("worst min h_cone [m]", lambda v: f"{v['h_min']['min']:+.3f}"),
            ("median min h_cone [m]", lambda v: f"{v['h_min']['median']:+.3f}"),
            ("final height, median [m]", lambda v: f"{v['final_height']['median']:.3f}"),
            ("final distance to pad centre, median [m]", lambda v: f"{v['final_pad_dist_xy']['median']:.3f}"),
            ("ends over the pad (≤ r0) [%]", lambda v: f"{100 * v['ends_over_pad_rate']:.1f}"),
            ("mean tracking error, median [m]", lambda v: f"{v['pos_err_mean']['median']:.3f}"),
            ("filter modifies the action [% steps]", lambda v: f"{100 * v['intervention_rate']['mean']:.1f}"),
            ("max QP slack", lambda v: f"{v['slack_max']:.1e}"),
            ("QP fallbacks [steps]", lambda v: f"{v['qp_fallback_steps']}")]
    base = runs[0][1]["variants"]["unfiltered"]["start_in_CN"]
    labels = [label_of(s) for _, s, _ in runs]
    md = ["| metric | policy only | " + " | ".join(f"filter: {l}" for l in labels) + " |",
          "|---|---|" + "---|" * len(runs)]
    for name, fmt in rows:
        md.append(f"| {name} | {fmt(base)} | " + " | ".join(fmt(s["variants"]["filtered"]["start_in_CN"]) for _, s, _ in runs) + " |")
    text = f"# Filter settings compared on `{runs[0][1]['setup']['phase2_run_name']}`\n\nStarts in C_N only.\n\n" + "\n".join(md) + "\n"
    (out / "compare.md").write_text(text, encoding="utf-8")
    print(text)

    lc = runs[0][1]["setup"]["landing_config"]
    pad = np.array([lc["pad_x"], lc["pad_y"], lc["pad_z"]])
    tan_t = np.tan(np.radians(lc["cone_theta_deg"]))
    ref = np.load(runs[0][1]["setup"]["reference_path"])["states"]
    dvec = ref[-1, 0:2] - pad[:2]
    dvec = dvec / np.linalg.norm(dvec) if np.linalg.norm(dvec) > 1e-6 else np.array([1.0, 0.0])
    dt = float(lc["dt"])
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8))
    ax = axs[0]
    z_win = float(lc["z_des"])  # window: from the pad up to the base-set height
    td = float((ref[-1, 0:2] - pad[:2]) @ dvec)
    zz = np.linspace(0.0, z_win, 200)
    rr = np.sqrt(np.maximum((lc["cone_r0"] + tan_t * zz) ** 2 - lc["cone_eps"] ** 2, 0.0))
    ax.fill_betweenx(zz, -rr, rr, color="tab:green", alpha=0.1)
    ax.plot(rr, zz, color="tab:green")
    ax.plot(-rr, zz, color="tab:green")
    ax.plot([-lc["cone_r0"], lc["cone_r0"]], [0, 0], color="tab:green", lw=5, solid_capstyle="butt")
    rel = ref[:, 0:3] - pad
    ax.plot(rel[:, 0:2] @ dvec, rel[:, 2], "--", color="k", lw=1.2, label="reference")
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
    for c, (d, s, ev), lab in zip(cols, runs, labels):
        m = ev["start_recoverable"].astype(bool)
        tr = ev["filtered_traj"][m]
        rel = tr[..., 0:3] - pad
        sx, sz = rel[..., 0:2] @ dvec, rel[..., 2]
        ax.plot(sx[0], sz[0], color=c, lw=1.6, label=lab)
        ax.scatter(sx[:, -1], sz[:, -1], s=10, color=c)
        t = np.arange(tr.shape[1]) * dt
        axs[1].plot(t, np.median(sz, axis=0), color=c, lw=1.8, label=lab)
        axs[1].fill_between(t, np.percentile(sz, 10, axis=0), np.percentile(sz, 90, axis=0), color=c, alpha=0.15)
        h = ev["filtered_h"][m]
        axs[2].plot(t, np.median(h, axis=0), color=c, lw=1.8, label=lab)
        axs[2].plot(t, h.min(axis=0), color=c, lw=0.8, ls=":")
    unf = runs[0][2]
    m = unf["start_recoverable"].astype(bool)
    t = np.arange(unf["unfiltered_traj"].shape[1]) * dt
    axs[1].plot(t, np.median(unf["unfiltered_traj"][m][..., 2] - pad[2], axis=0), color="tab:red", lw=1.4, label="policy only")
    axs[2].plot(t, np.median(unf["unfiltered_h"][m], axis=0), color="tab:red", lw=1.4, label="policy only")
    ax.set_xlim(-float(rr[-1]), max(float(rr[-1]), td + 0.2))
    ax.set_ylim(-0.05, z_win)
    ax.set_aspect("equal")
    ax.set_xlabel("along miss direction [m]")
    ax.set_ylabel("height above pad [m]")
    ax.set_title("near the pad: episode 0 path, final positions of all C_N starts (dots)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("height above pad [m]")
    axs[1].set_title("height (median, 10–90%)", fontsize=10)
    axs[1].legend(fontsize=7)
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_xlabel("t [s]")
    axs[2].set_ylabel("h_cone [m]")
    axs[2].set_title("cone margin (median solid, minimum dotted)", fontsize=10)
    axs[2].legend(fontsize=7)
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(f"Filter settings compared on {runs[0][1]['setup']['phase2_run_name']} (starts in C_N)")
    fig.tight_layout()
    fig.savefig(out / "compare.png", dpi=140)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
