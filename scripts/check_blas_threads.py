#!/usr/bin/env python3
"""How much CPU does the slow-track policy's numpy forward pass burn at 50 Hz?

Runs a 34-256-256-8 relu MLP (the size of the slow-track actor) 50 times a second for a few seconds,
once with this machine's default BLAS threading and once with OPENBLAS_NUM_THREADS=1, and prints
the CPU time used per second of wall time.

Older OpenBLAS builds (for example the one in numpy 1.26) split even a 256 x 256 matrix-vector
product over every core and then keep those threads spinning between calls. At 50 Hz they never
sleep, so the bridge takes most of the computer away from Gazebo and PX4. Run this with the same
python3 as the bridge:

    python3 ~/ws_shared/PS2-RL/scripts/check_blas_threads.py
"""
from __future__ import annotations

import os
import subprocess
import sys

CHILD = r"""
import os, resource, time
import numpy as np
rng = np.random.default_rng(0)
layers = [(rng.standard_normal((34, 256)), rng.standard_normal(256)),
          (rng.standard_normal((256, 256)), rng.standard_normal(256)),
          (rng.standard_normal((256, 8)), rng.standard_normal(8))]
def act(o):
    h = o
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < 2:
            h = np.maximum(h, 0.0)
    return h
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime
o = rng.standard_normal(34)
for _ in range(50):
    act(o)
c0, w0, n = cpu(), time.time(), 0
while time.time() - w0 < float(os.environ["CHECK_SECONDS"]):
    t = time.time(); act(o); n += 1
    time.sleep(max(0.0, 0.02 - (time.time() - t)))
w = time.time() - w0
pools = ""
try:
    from threadpoolctl import threadpool_info
    pools = ", ".join(f"{p.get('internal_api')} {p.get('num_threads')} threads" for p in threadpool_info())
except Exception:
    pass
print(f"{100 * (cpu() - c0) / w:7.0f} %   ({n} forward passes; numpy {np.__version__}{'; ' + pools if pools else ''})")
"""


def run(label: str, env_update: dict, drop: tuple = ()) -> None:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_update)
    env["CHECK_SECONDS"] = "4"
    print(f"{label:38s}", end="", flush=True)
    out = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())


def main() -> None:
    print(f"logical CPUs: {os.cpu_count()};  CPU used per second of wall time (100 % = one core):")
    blas_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
    run("default threading (as before the fix)", {}, drop=blas_vars)
    run("OPENBLAS_NUM_THREADS=1 (the fix)", {v: "1" for v in blas_vars})
    print("A default far above the single-threaded value means BLAS threads spin between policy steps.")


if __name__ == "__main__":
    main()
