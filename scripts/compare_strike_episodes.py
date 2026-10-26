#!/usr/bin/env python
"""Compare closed-loop strike evaluations on the same scenes, episode by episode (e.g. two methods, two seeds or two
chasers). Each run is an output folder of scripts/eval_strike_filter.py with the same --scene_seed and --episodes.

    python scripts/compare_strike_episodes.py --runs outputs/strike_final/eval/s0/pursuit_final outputs/strike_final/eval/s0/ps2_final \
        --out outputs/strike_final/compare_s0 --pick 6
    # after the runs have GIFs of the same scenes (eval_strike_filter.py --gif_scenes ...): stitch them side by side
    python scripts/compare_strike_episodes.py --runs ... --out ... --scenes 3 17 42 --stitch

Writes to --out:
  compare.md            summary per run and mode; the episodes whose outcome under --mode differs between the runs
  picked_scenes.txt     scenes worth animating: contacts, changed outcomes, chaser hits the filter prevented, close calls
  compare_scene<i>.gif  (--stitch) the runs' GIFs of scene i side by side, the first run on the left
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHORT = {"strike": "strike", "crash_obstacle": "obstacle", "crash_floor": "floor", "crash_chaser": "chaser",
         "out_of_arena": "out", "timeout": "time-up"}


def load(run: Path):
    ep = json.loads((run / "strike_filter_episodes.json").read_text())
    tb = json.loads((run / "strike_filter_table.json").read_text())
    return ep, tb


def chaser_of(tb: dict) -> str:
    ch = tb.get("chaser")
    if not ch:
        return ""
    return f"chaser {ch['actual_radius']:g} m, {ch['actual_speed']:g} m/s" + ("" if ch.get("filter_sees_chasers", True)
                                                                             else ", filter blind")


def stitch(gifs, labels, path: Path, stride: int = 2, header: int = 22):
    """The GIFs side by side (frame by frame; a shorter one holds its last frame), a label above each."""
    from PIL import Image, ImageDraw, ImageSequence
    ims = [Image.open(g) for g in gifs]
    n = [getattr(im, "n_frames", 1) for im in ims]
    dur = int(ims[0].info.get("duration", 40)) * stride
    w = [im.width for im in ims]
    h = max(im.height for im in ims)
    adaptive = Image.Palette.ADAPTIVE if hasattr(Image, "Palette") else Image.ADAPTIVE
    out = []
    for k in range(0, max(n), stride):
        canvas = Image.new("RGB", (sum(w), h + header), "white")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for im, nk, wk, lab in zip(ims, n, w, labels):
            im.seek(min(k, nk - 1))
            canvas.paste(im.convert("RGB"), (x, header))
            draw.text((x + 8, 5), lab, fill=(11, 11, 11))
            x += wk
        out.append(canvas.convert("P", palette=adaptive, colors=128))
    out[0].save(path, save_all=True, append_images=out[1:], duration=dur, loop=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="eval_strike_filter.py output folders (same scenes)")
    p.add_argument("--labels", nargs="*", default=None, help="one per run (default: the chaser of each run)")
    p.add_argument("--mode", default="learned", help="mode compared episode by episode (learned = the deployed filter)")
    p.add_argument("--pick", type=int, default=6, help="number of scenes written to picked_scenes.txt")
    p.add_argument("--scenes", type=int, nargs="*", default=None, help="scenes to stitch (default: the picked ones)")
    p.add_argument("--stitch", action="store_true", help="stitch the runs' GIFs of each scene side by side")
    p.add_argument("--out", default="", help="output folder (default: the parent of the first run)")
    a = p.parse_args(argv)
    runs = [Path(r) for r in a.runs]
    out = Path(a.out) if a.out else runs[0].parent
    out.mkdir(parents=True, exist_ok=True)
    data = [load(r) for r in runs]
    if a.labels:
        labels = a.labels
    else:                                         # the run folders; the chaser too where the runs differ in it
        chs = [chaser_of(tb) for _, tb in data]
        labels = [r.name + (f" ({c})" if len(set(chs)) > 1 and c else "") for r, c in zip(runs, chs)]
    if len(labels) != len(runs):
        raise SystemExit("--labels needs one label per run")
    seeds = {(tb["scene_seed"], tb["episodes"], tuple(ep.get("episode_index") or [])) for ep, tb in data}
    if len(seeds) > 1:
        raise SystemExit("the runs are not on the same episodes (scene seed, episodes, replayed subset differ)")
    ep_idx = np.asarray(data[0][0].get("episode_index") or np.arange(len(data[0][0]["has_chaser"])))
    noms = {tb["nominal"] for _, tb in data}
    names = data[0][0]["outcomes"]
    dt = float(data[0][0]["dt"])
    n_ep = data[0][1]["episodes"]

    lines = [f"Same {n_ep} scenes (seed {data[0][1]['scene_seed']}); nominal: {'; '.join(sorted(noms))}.", "",
             "| run | mode | strike | contact | of which chaser | time-up | median strike time [s] | lost-certificate episodes |",
             "|---|---|---|---|---|---|---|---|"]
    for lab, (ep, _) in zip(labels, data):
        for mode, m in ep["modes"].items():
            oc = np.asarray(m["outcome"])
            contact = np.isin(oc, [names.index(k) for k in ("crash_obstacle", "crash_floor", "crash_chaser")])
            ts = np.asarray(m["strike_step"])[oc == 0] * dt
            lost = (np.asarray(m["no_cert_steps"]) > 0).mean() if mode != "none" else float("nan")
            lines.append(f"| {lab} | {mode} | {(oc == 0).mean():.3f} | {contact.mean():.3f} | "
                         f"{(oc == names.index('crash_chaser')).mean():.3f} | {(oc == names.index('timeout')).mean():.3f} | "
                         f"{np.median(ts) if len(ts) else float('nan'):.2f} | {lost:.3f} |")

    if any(a.mode not in ep["modes"] for ep, _ in data):
        raise SystemExit(f"mode {a.mode!r} is missing in a run (modes: {[list(ep['modes']) for ep, _ in data]})")
    M = [{k: np.asarray(v) for k, v in ep["modes"][a.mode].items()} for ep, _ in data]
    N = [({k: np.asarray(v) for k, v in ep["modes"]["none"].items()} if "none" in ep["modes"] else None) for ep, _ in data]
    oc = np.stack([m["outcome"] for m in M])                      # (runs, episodes)
    changed = np.nonzero((oc != oc[0]).any(axis=0))[0]

    def cell(m, i):
        o = SHORT[names[int(m["outcome"][i])]]
        t = int(m["end_step"][i]) * dt
        return f"{o} {t:.1f} s, min {float(m['min_chaser_dist'][i]):.2f} m, no-cert {int(m['no_cert_steps'][i]) * dt:.1f} s"

    lines += ["", f"Episodes whose outcome under '{a.mode}' differs ({len(changed)} of {len(ep_idx)}); min = closest chaser "
                  "distance, no-cert = time without a certificate. Unfiltered outcome in brackets where the run has it.", "",
              "| scene | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for i in changed:
        row = []
        for m, nm in zip(M, N):
            extra = f" [no filter: {SHORT[names[int(nm['outcome'][i])]]}]" if nm is not None else ""
            row.append(cell(m, i) + extra)
        lines.append(f"| {int(ep_idx[i])} | " + " | ".join(row) + " |")

    # scenes worth animating
    contact_codes = [names.index(k) for k in ("crash_obstacle", "crash_floor", "crash_chaser")]
    pick = []

    def add(idx):                                 # positions -> episode indices
        for i in idx:
            e = int(ep_idx[int(i)])
            if e not in pick and len(pick) < a.pick:
                pick.append(e)

    add([i for r in range(len(runs) - 1, -1, -1) for i in np.nonzero(np.isin(oc[r], contact_codes))[0]])   # contacts
    add(changed)                                                                                        # changed outcomes
    last, last_none = M[-1], N[-1]
    if last_none is not None:                         # the filter prevented a chaser hit (last run)
        add(np.nonzero((last_none["outcome"] == names.index("crash_chaser")) & (last["outcome"] == 0))[0])
    s = np.nonzero(last["outcome"] == 0)[0]
    add(s[np.argsort(last["min_chaser_dist"][s])])     # close calls among the strikes (last run)
    (out / "picked_scenes.txt").write_text(" ".join(map(str, pick)) + "\n")
    lines += ["", f"Picked scenes (contacts, changed outcomes, chaser hits prevented by the filter, close calls): "
                  f"{' '.join(map(str, pick))}"]

    if a.stitch:
        scenes = a.scenes if a.scenes else pick
        made = []
        for i in scenes:
            gifs = [r / f"strike_filter_scene{i}.gif" for r in runs]
            if all(g.exists() for g in gifs):
                stitch(gifs, labels, out / f"compare_scene{i}.gif")
                made.append(i)
            else:
                print(f"[stitch] scene {i}: GIF missing in {[str(g.parent) for g in gifs if not g.exists()]}", flush=True)
        lines += ["", f"Side-by-side GIFs ({labels[0]} on the left): " + ", ".join(f"compare_scene{i}.gif" for i in made)]

    text = "\n".join(lines) + "\n"
    (out / "compare.md").write_text(text)
    print(text)
    print(f"wrote {out}/compare.md, picked_scenes.txt" + (", compare_scene*.gif" if a.stitch else ""), flush=True)


if __name__ == "__main__":
    main()
