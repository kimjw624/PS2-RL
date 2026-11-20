"""The obstacle-demo controller in its own process (``CylOnlineRuntimeConfig.process``).

Why: in the ROS node the control step shares Python's GIL with the node's executor threads (PX4 messages, timers), so
a step that takes 8 ms on its own took 23-32 ms on a Raspberry Pi 4B, and 6 ms instead of 1.3 ms on a laptop. In a
worker process the controller has an interpreter (and a CPU core) of its own; the node's control thread only sends the
state and waits for the command - the wait releases the GIL, so the executor runs meanwhile.

``RuntimeProcess`` has the interface of ``CylOnlineRuntime`` that the flight logic and the node use (the constants,
``describe``, ``reset``, ``reset_observer``, ``observe``, ``margins``, ``clearance``, ``recoverable``, ``nominal_goal``,
``act``, the state ``d_hat`` / ``goal_entered_t``, and ``cpu_s``: the worker's CPU time so far). Every call is forwarded to a ``CylOnlineRuntime`` in the worker
and returns the same values (numpy arrays pickled through a pipe). ``act``'s ``latency_s`` is the round trip seen by
the caller; ``info["compute_ms"]`` the worker's own time.

The worker is a fresh interpreter (``python -m ps2rl.deploy.cyl_online_worker``, not a fork of the multithreaded ROS
process) with this process's ``sys.path`` and environment (single-threaded BLAS from the bridge package), pinned to
``worker_cpu`` if given. It exits when the pipe closes (``close``, or this process ends).
"""

from __future__ import annotations

from dataclasses import asdict
import os
from multiprocessing.connection import Connection
import subprocess
import sys
import time

import numpy as np

STATE_KEYS = ("d_hat", "goal_entered_t", "k", "started")


class RuntimeProcess:
    """Proxy to a ``CylOnlineRuntime`` running in a worker process."""

    def __init__(self, cfg, *, ps2rl_path: str | None = None, logger=None, start_timeout: float = 120.0) -> None:
        self.cfg = cfg
        self._log = logger
        p_rd, c_wr = os.pipe()                       # parent reads  <- child writes
        c_rd, p_wr = os.pipe()                       # child reads   <- parent writes
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        self._proc = subprocess.Popen([sys.executable, "-m", "ps2rl.deploy.cyl_online_worker", str(c_rd), str(c_wr)],
                                      pass_fds=(c_rd, c_wr), env=env, close_fds=True)
        os.close(c_rd)
        os.close(c_wr)
        self._conn_in, self._conn_out = Connection(p_rd), Connection(p_wr)
        self._conn_out.send(("init", (asdict(cfg), ps2rl_path)))
        if not self._conn_in.poll(start_timeout):
            self.close()
            raise RuntimeError(f"the controller process did not start within {start_timeout:g} s")
        ok, payload, logs = self._conn_in.recv()
        self._relay(logs)
        if not ok:
            self.close()
            raise RuntimeError(f"the controller process failed to start: {payload}")
        self._apply(payload)

    # ------------------------------------------------------------------ plumbing
    def _relay(self, logs) -> None:
        for level, msg in logs:
            if self._log is None:
                print(msg)
            else:
                getattr(self._log, "warn" if level == "warn" else "info")(msg)

    def _apply(self, payload: dict) -> None:
        for k, v in payload.get("const", {}).items():
            setattr(self, k, v)
        for k, v in payload.get("state", {}).items():
            setattr(self, k, v)

    def _call(self, name: str, *args, **kwargs):
        self._conn_out.send((name, (args, kwargs)))
        ok, payload, logs = self._conn_in.recv()
        self._relay(logs)
        if not ok:
            raise RuntimeError(f"controller process: {name} failed: {payload}")
        self._apply(payload)
        return payload.get("result")

    # ------------------------------------------------------------------ the runtime's interface
    def describe(self) -> dict:
        return self._describe

    def reset(self) -> None:
        self._call("reset")

    def reset_observer(self, x) -> None:
        self._call("reset_observer", np.asarray(x, np.float64))

    def observe(self, x, a_thrust: float, dt: float) -> None:
        self._call("observe", np.asarray(x, np.float64), float(a_thrust), float(dt))

    def clearance(self, x) -> float:
        return self._call("clearance", np.asarray(x, np.float64))

    def margins(self, x):
        return self._call("margins", np.asarray(x, np.float64))

    def recoverable(self, x) -> bool:
        return self._call("recoverable", np.asarray(x, np.float64))

    def nominal_goal(self, t: float):
        return self.goal.copy()

    def act(self, x, t: float, *, a_meas=None, omega_meas=None, dt_obs=None):
        t0 = time.perf_counter()
        out = self._call("act", np.asarray(x, np.float64), float(t))
        out.info["compute_ms"] = 1e3 * out.latency_s
        out.latency_s = time.perf_counter() - t0
        return out

    def close(self) -> None:
        for c in (getattr(self, "_conn_out", None), getattr(self, "_conn_in", None)):
            try:
                if c is not None:
                    c.close()
            except OSError:
                pass
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- the worker
class _Log:
    def __init__(self) -> None:
        self.buf: list = []

    def info(self, msg) -> None:
        self.buf.append(("info", str(msg)))

    def warn(self, msg) -> None:
        self.buf.append(("warn", str(msg)))

    warning = warn

    def take(self) -> list:
        out, self.buf = self.buf, []
        return out


def _consts(rt) -> dict:
    return {"dt": rt.dt, "gravity": rt.gravity, "action_low": rt.action_low, "action_high": rt.action_high, "pad": rt.pad,
            "start_state": rt.start_state, "hover_position": rt.hover_position, "target": rt.target, "goal": rt.goal,
            "obst": rt.obst, "label": rt.label, "backend": rt.backend + " (worker process)", "course_label": rt.course_label,
            "r_drone": rt.r_drone, "rho_xy": rt.rho_xy, "z_floor": rt.z_floor, "hold": rt.hold, "meta": rt.meta,
            "filter_overrides": {}, "_describe": {**rt.describe(), "backend": rt.backend + " (worker process)"}}


def _state(rt) -> dict:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {**{k: getattr(rt, k) for k in STATE_KEYS}, "cpu_s": r.ru_utime + r.ru_stime}   # the worker's CPU time so far


def _serve(fd_in: int, fd_out: int) -> None:
    conn_in, conn_out = Connection(fd_in), Connection(fd_out)
    log = _Log()
    rt = None
    while True:
        try:
            name, payload = conn_in.recv()
        except (EOFError, OSError):
            return
        try:
            if name == "init":
                cfg_dict, ps2rl_path = payload
                from ps2rl.deploy.cyl_online_runtime import CylOnlineRuntime, CylOnlineRuntimeConfig
                cfg = CylOnlineRuntimeConfig(**{**cfg_dict, "process": False})
                if int(cfg.worker_cpu) >= 0 and hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, {int(cfg.worker_cpu)})
                    log.info(f"controller process pinned to CPU {cfg.worker_cpu}")
                rt = CylOnlineRuntime(cfg, ps2rl_path=ps2rl_path, logger=log)
                log.info(f"controller process: pid {os.getpid()}")
                conn_out.send((True, {"const": _consts(rt), "state": _state(rt)}, log.take()))
                continue
            args, kwargs = payload
            result = getattr(rt, name)(*args, **kwargs)
            conn_out.send((True, {"result": result, "state": _state(rt)}, log.take()))
        except Exception as exc:  # report, keep serving (the caller decides)
            import traceback
            conn_out.send((False, f"{exc!r}\n{traceback.format_exc()}", log.take()))


if __name__ == "__main__":
    _serve(int(sys.argv[1]), int(sys.argv[2]))
