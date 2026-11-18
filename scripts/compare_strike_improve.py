"""Paired comparison of docker_batch/run_strike_improve.sh outputs against each striker's reference run.

    python scripts/compare_strike_improve.py outputs/strike_improve [--seeds "0 1"] [--out compare.md] [--contacts contacts.md]

Variants are the folders under s<seed>/ that every seed has; each is compared with the reference of its striker
(--ref_suffix, default "_ref": ps2_* against ps2_ref, residual_* against residual_ref). Seeds are pooled; a pair is the
same seed and scene. Per variant:
- strike %, contacts (count), time-up %;
- scenes won and lost against the reference (strike vs no strike), exact McNemar p;
- strike time over the pairs both strike: mean, difference, Wilcoxon signed-rank p;
- lost-certificate episodes %, filter active share and ||du|| / nominal (step-weighted, as the evaluation reports them).
Contacts: seed, scene, outcome, steps without a certificate, minimum chaser and obstacle distances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, wilcoxon


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))


def load(d: Path, seed: int, v: str) -> dict | None:
    f = d / f"s{seed}" / v / "strike_filter_episodes.json"
    if not f.exists():
        return None
    z = json.loads(f.read_text())
    m = z["modes"]["learned"]
    out = {k: np.asarray(m[k], float) for k in m if k != "outcome"}
    out["code"] = np.array([z["outcomes"][i] for i in m["outcome"]])
    out["t"] = out["end_step"] * z["dt"]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dir")
    p.add_argument("--seeds", default="0 1")
    p.add_argument("--ref_suffix", default="_ref")
    p.add_argument("--out", default="")
    p.add_argument("--contacts", default="")
    a = p.parse_args(argv)
    d, seeds = Path(a.dir), [int(s) for s in a.seeds.split()]
    names = sorted(set.intersection(*[{q.name for q in (d / f"s{s}").iterdir() if (q / "strike_filter_episodes.json").exists()}
                                      for s in seeds]))
    refs = [n for n in names if n.endswith(a.ref_suffix)]
    order = []
    for r in refs:                                  # each reference first, then its variants
        stem = r[: -len(a.ref_suffix)]
        order += [r] + [n for n in names if n.startswith(stem + "_") and n != r]
    rows = ["| variant | episodes | strike % | contact | time-up % | won / lost vs ref (p) | strike time, both strike [s] | "
            "lost-cert episodes % | filter active | ‖Δu‖ / nominal |", "|" + "---|" * 10]
    crow = ["| variant | seed | scene | outcome | steps without certificate | min chaser dist [m] | min obstacle dist [m] |",
            "|" + "---|" * 7]
    for v in order:
        ref = next(r for r in refs if v.startswith(r[: -len(a.ref_suffix)] + "_") or v == r)
        R, F = [load(d, s, v) for s in seeds], [load(d, s, ref) for s in seeds]
        cat = lambda key, X: np.concatenate([x[key] for x in X])
        code, rcode = cat("code", R), cat("code", F)
        st, rst = code == "strike", rcode == "strike"
        con = np.char.startswith(code.astype(str), "crash")
        won, lost = int(np.sum(st & ~rst)), int(np.sum(~st & rst))
        both = st & rst
        dts = cat("t", R)[both] - cat("t", F)[both]
        p_t = wilcoxon(dts).pvalue if np.any(dts != 0) else 1.0
        act = cat("active_steps", R).sum() / cat("end_step", R).sum()
        du = cat("du_sum", R).sum() / max(cat("du_nom_sum", R).sum(), 1e-9)
        is_ref = v == ref
        rows.append(f"| {v} | {len(code)} | {100 * st.mean():.2f} | {int(con.sum())} | {100 * np.mean(code == 'timeout'):.2f} | "
                    f"{'—' if is_ref else f'{won} / {lost} (p {fmt_p(mcnemar(won, lost))})'} | "
                    f"{cat('t', R)[both].mean():.2f}" + ("" if is_ref else f" (Δ {dts.mean():+.2f}, p {fmt_p(p_t)})") + " | "
                    f"{100 * np.mean(cat('no_cert_steps', R) > 0):.2f} | {act:.3f} | {du:.2f} |")
        for s, r in zip(seeds, R):
            for i in np.where(np.char.startswith(r["code"].astype(str), "crash"))[0]:
                crow.append(f"| {v} | {s} | {i} | {r['code'][i]} | {int(r['no_cert_steps'][i])} | "
                            f"{r['min_chaser_dist'][i]:.2f} | {r['min_obstacle_dist'][i]:.2f} |")
    txt = "\n".join(rows)
    print(txt + "\n\n" + "\n".join(crow))
    if a.out:
        Path(a.out).write_text(txt + "\n")
    if a.contacts:
        Path(a.contacts).write_text("\n".join(crow) + "\n")


if __name__ == "__main__":
    main()
