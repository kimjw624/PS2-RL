"""Learning curves of the UE Phase I vs the nominal landing Phase I (recoverability per region).

    python scripts/plot_phase1_landing_ue.py --ue_run outputs/landing_phase1_ue/ue_floor_rec10_td3_h128_seed0 \
        --nominal_history results/landing_phase1/floor_rec10_td3_seed0/training/history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ue_run", required=True)
    p.add_argument("--nominal_history", default="results/landing_phase1/floor_rec10_td3_seed0/training/history.json")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    ue_dir = Path(a.ue_run)
    h = json.loads((ue_dir / "history.json").read_text())
    cfg = json.loads((ue_dir / "configs.json").read_text())
    n_ue = len(h["eval_mu_weighted"])
    s_ue = np.arange(1, n_ue + 1) * cfg["backup_ra"]["eval_every"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    regions = ("general", "edge", "shell")
    for i, r in enumerate(regions):
        ax = axes[i]
        ax.plot(s_ue / 1e6, h[f"eval_m_{r}"], "b-", label="UE Phase I: tightened C_N, d_hat ~ |d| <= delta_d")
        ax.plot(s_ue / 1e6, h[f"eval_nominal_m_{r}"], "b:", label="UE Phase I backup, d_hat = 0, untightened")
        if a.nominal_history and Path(a.nominal_history).exists():
            hn = json.loads(Path(a.nominal_history).read_text())
            sn = np.arange(1, len(hn[f"eval_m_{r}"]) + 1) * 5e4
            ax.plot(sn / 1e6, hn[f"eval_m_{r}"], color="tab:orange", alpha=0.8,
                    label="nominal Phase I, its own (nominal) test")
        cmp_path = ue_dir / "compare_backups.json"
        if cmp_path.exists():
            cmp = json.loads(cmp_path.read_text())
            ax.axhline(cmp["nominal"][r]["tightened_with_d_hat"], color="tab:red", ls="--",
                       label="nominal backup on the UE test (same states, d_hat, tube)")
            ax.axhline(cmp["ue"][r]["tightened_with_d_hat"], color="b", ls="-.", alpha=0.5,
                       label="UE backup (selected) on the UE test")
        ax.set_title(f"recoverability m_{r}")
        ax.set_xlabel("steps [M]")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    ax = axes[3]
    ax.plot(s_ue / 1e6, h["eval_rate_p90"], "k-", label="p90 of max growth rate along rollout")
    ax.set_title("P-metric tube growth rate [1/s] (outside B)")
    ax.set_xlabel("steps [M]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(a.out) if a.out else ue_dir / "phase1_ue_curves.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
