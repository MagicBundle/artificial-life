"""Append-and-fsync latency of a directory: 300 appends of a 0.4 kB line, 20 ms apart (G3b, RESULTS_G3B 6.1).

Usage: python fsync_probe.py DIR.  Leaves DIR/fsync_probe.jsonl (the caller removes it).
"""

import os
import sys
import time

d = sys.argv[1]
p = os.path.join(d, "fsync_probe.jsonl")
ts = []
for i in range(300):
    t0 = time.perf_counter()
    with open(p, "a") as f:
        f.write('{"i": %d, "pad": "%s"}\n' % (i, "x" * 400))  # noqa: UP031
        f.flush()
        os.fsync(f.fileno())
    ts.append(1e3 * (time.perf_counter() - t0))
    time.sleep(0.02)
ts.sort()
print(d, "n", len(ts), f"p50 {ts[150]:.2f} p90 {ts[270]:.2f} p99 {ts[297]:.2f} max {ts[-1]:.2f} ms")
