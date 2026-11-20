#!/usr/bin/env python
"""Check the compiled rollouts (numba) against the numpy path of the online filter: the same closed-loop episodes
run with both backends must make the same decisions (blend, backup, mode) at every step.

    python scripts/check_cyl_online_fast.py --backup checkpoints/cyl_online/td3_w64_seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ps2rl.online.filter import FilterConfig, OnlineFilter  # noqa: E402
from ps2rl.online.model import load_backup  # noqa: E402
from ps2rl.online.nominal import Nominal, NominalConfig  # noqa: E402
from ps2rl.online.sim import PlantConfig, SceneConfig, make_scenes, run_episodes  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backup", default=str(ROOT / "checkpoints/cyl_online/td3_w64_seed0"))
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--periods", default="1,2")
    a = p.parse_args(argv)
    m, actor, _ = load_backup(a.backup)
    fam = ("learned", "lqr") if actor is not None else ("lqr",)
    worst = 0
    for P in [int(v) for v in a.periods.split(",")]:
        for kind, nc in (("random", NominalConfig(cruise=1.2, margin=0.45, lookahead=3.0)), ("one", NominalConfig(search=False))):
            sc = SceneConfig(kind=kind, sensing_range=4.0, episode_sec=20.0)
            eps = make_scenes(sc, a.episodes, seed=3)
            nom = Nominal(m, nc)
            out = {}
            for be in ("numpy", "numba"):
                f = OnlineFilter(m, actor, FilterConfig(backups=fam, period=P, backend=be), nom)
                t0 = time.time()
                out[be] = run_episodes(m, eps, sc, PlantConfig(), nom, f, seed=5, log=True)
                out[be]["wall"] = time.time() - t0
            la, lb = out["numpy"]["log"], out["numba"]["log"]
            n = min(len(la["lam"]), len(lb["lam"]))
            dl = np.abs(np.asarray(la["lam"][:n]) - np.asarray(lb["lam"][:n])).max()
            dx = np.abs(np.asarray(la["x"][:n]) - np.asarray(lb["x"][:n])).max()
            mism = sum(int(np.any(ma != mb)) for ma, mb in zip(la["mode"][:n], lb["mode"][:n]) if ma is not None)
            worst = max(worst, mism)
            print(f"[check] period {P} {kind:6}: steps {n}, max |dlam| {dl:.2e}, max |dx| {dx:.2e}, steps with a different "
                  f"mode {mism}; success {out['numpy']['success']:.2f} / {out['numba']['success']:.2f}, wall "
                  f"{out['numpy']['wall']:.1f} s numpy / {out['numba']['wall']:.1f} s numba", flush=True)
    print("[check] " + ("OK: identical decisions" if worst == 0 else f"MISMATCH ({worst} steps)"))


if __name__ == "__main__":
    main()
