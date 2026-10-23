"""Driver for the G3b checkpoint-stall measurement (not part of the repo).

Runs ``alife bench --play ...`` unchanged and, from outside, logs every checkpoint the bench pacer writes:
the step, the stopwatch of ``before_checkpoint`` to ``after_checkpoint`` (the same ``save`` part the bench
reports as a mean), the cumulative ``save_stall_steps``, the file size and whether the call fell inside the
measured part.  Usage: python bench_ckpt.py CKPT_LOG.jsonl -- <alife args>
"""

import json
import os
import sys
import time

from alife import bench, cli

log_path = sys.argv[1]
args = sys.argv[sys.argv.index("--") + 1 :]
orig = bench._bench_pacer_class


def patched():
    base = orig()

    class LoggedPacer(base):
        def before_checkpoint(self, state, t):
            self._ck_t0 = time.perf_counter()
            self._ck_step = t
            super().before_checkpoint(state, t)

        _late_seen = 0

        def after_step(self, state, out, t, social):
            super().after_step(state, out, t, social)
            late = self.sched.late_steps
            if late != self._late_seen:
                with open(log_path, "a") as f:
                    rec = {
                        "event": "late",
                        "run": self.opt.run,
                        "step": t,
                        "late_cum": late,
                        "added": late - self._late_seen,
                        "utc": time.strftime("%H:%M:%S"),
                    }
                    f.write(json.dumps(rec) + "\n")
                self._late_seen = late

        def after_checkpoint(self, t):
            super().after_checkpoint(t)
            ms = self._saved_ms
            path = os.path.join(self.opt.run, "checkpoints", f"step_{t:010d}.safetensors")
            size = os.path.getsize(path) if os.path.exists(path) else None
            rec = {
                "run": self.opt.run,
                "step": t,
                "save_ms": ms,
                "since_before_ms": 1e3 * (time.perf_counter() - self._ck_t0),
                "save_stall_steps_cum": self.sched.save_stall_steps,
                "late_cum": self.sched.late_steps,
                "recording": self.timing.recording,
                "bytes": size,
                "utc": time.strftime("%H:%M:%S"),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    return LoggedPacer


bench._bench_pacer_class = patched
sys.exit(cli.main(args))
