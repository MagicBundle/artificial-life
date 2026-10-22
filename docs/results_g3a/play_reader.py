"""Attach a strict reader to `alife play` (Metal), decode every record with tests/play_decode.py, keep counts.

usage: play_reader.py OUTJSON STOP_AFTER_STEPS FOLLOW(0|1) -- alife-play-args...
Quits (sends `quit`) once the stream's step is >= first step + STOP_AFTER_STEPS.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time

import numpy as np

WT = os.environ["WT"]
sys.path.insert(0, WT + "/tests")
from play_decode import decode  # noqa: E402
from stream_decode import decode_header  # noqa: E402

out_json, stop_after, follow = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
args = sys.argv[sys.argv.index("--") + 1 :]
cmd = [os.environ["ALIFE_BIN"], "play", *args]
t0 = time.time()
err = open(out_json + ".stderr", "wb")  # noqa: SIM115 (open for the life of the child process)
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=0)

q: queue.Queue = queue.Queue()
raw_bytes = 0
max_q = 0


def reader() -> None:
    global raw_bytes
    buf = bytearray()
    hdr = None
    while True:
        chunk = p.stdout.read(1 << 20)
        if not chunk:
            break
        raw_bytes += len(chunk)
        buf += chunk
        while True:
            if hdr is None:
                if len(buf) < 12:
                    break
                jl = struct.unpack_from("<I", buf, 8)[0]
                if len(buf) < 12 + jl:
                    break
                hdr = bytes(buf[: 12 + jl])
                del buf[: 12 + jl]
                q.put(("hdr", time.time(), hdr))
                continue
            if len(buf) < 8:
                break
            ln = struct.unpack_from("<I", buf, 4)[0]
            if len(buf) < 8 + ln:
                break
            q.put(("rec", time.time(), bytes(buf[: 8 + ln])))
            del buf[: 8 + ln]
    q.put(("eof", time.time(), bytes(buf)))


th = threading.Thread(target=reader, daemon=True)
th.start()

hdr_bytes = b""
tags = collections.Counter()
bytes_by_tag = collections.Counter()
events = []  # (wall, step, fam, kind, imp, text, backlog)
fam_counts = collections.Counter()
storm_ids = set()
named_ids = {}
storm_ended = collections.Counter()
focus_events = []
focs_count = 0
focs_uid_changes = 0
last_focs_uid = None
first_step = None
last_step = 0
clck = []
decode_s = 0.0
max_decode_ms = 0.0
errors = []
followed = False
quit_sent = False
n_ids_seen = 0
first_rec_wall = None
tail_leftover = b""
latest = {}
recent_evnt = collections.deque(maxlen=40)
snap_done = set()


def dump_snapshot(name):
    with open(out_json + "." + name + ".bin", "wb") as f:
        f.write(hdr_bytes)
        for tg in ("IDNT", "INTN", "TERR", "STRM", "MAP1", "ENVS", "FRMA", "FOCS", "CLCK"):
            if tg in latest:
                f.write(latest[tg])
        for e in recent_evnt:
            f.write(e)


while True:
    kind, wall, data = q.get()
    max_q = max(max_q, q.qsize())
    if kind == "hdr":
        hdr_bytes = data
        header, _ = decode_header(data)
        continue
    if kind == "eof":
        tail_leftover = data
        break
    td = time.perf_counter()
    try:
        d = decode(hdr_bytes + data)
    except AssertionError as e:
        errors.append(f"assert at record #{sum(tags.values())}: {e}")
        d = None
    dt = (time.perf_counter() - td) * 1000
    decode_s += dt / 1000
    max_decode_ms = max(max_decode_ms, dt)
    if d is None:
        continue
    if d.skipped:
        errors.append(f"skipped {d.skipped}")
    r = d.records[0]
    tags[r.tag] += 1
    latest[r.tag] = data
    if r.tag == "EVNT":
        recent_evnt.append(data)
    bytes_by_tag[r.tag] += len(data)
    if first_step is None:
        first_step = r.step
        first_rec_wall = wall
    last_step = max(last_step, r.step)
    if r.tag == "IDNT" and follow and not followed and r.f["n_rows"] > 0:
        uid = int(r.a["uid"][r.f["n_rows"] // 2])
        p.stdin.write(f"follow {uid}\n".encode())
        p.stdin.flush()
        followed = True
    if r.tag == "EVNT":
        for e in r.f["json"]:
            events.append((wall, r.step, e["fam"], e["kind"], e["imp"], e["text"], r.f["backlog"]))
            if not r.f["backlog"]:
                fam_counts[e["fam"]] += 1
    if r.tag == "STRM":
        for s in r.f["json"]["storms"]:
            storm_ids.add(s["id"])
            if s["named"]:
                named_ids[s["id"]] = s["name"]
        for e in r.f["json"].get("ended", []) or []:
            storm_ended[e.get("reason", "?")] += 1
    if r.tag == "FOCS":
        focs_count += 1
        if r.f["uid"] != last_focs_uid:
            focs_uid_changes += 1
            last_focs_uid = r.f["uid"]
    if r.tag == "CLCK":
        clck.append(
            (
                wall,
                r.step,
                r.f["rate"],
                r.f["achieved"],
                r.f["step_ms"],
                r.f["late_steps"],
                r.f["save_stall_steps"],
                r.f["saving"],
                r.f["reason"],
            )
        )
    if (
        r.tag == "FRMA"
        and "mid" not in snap_done
        and first_step is not None
        and last_step >= first_step + stop_after // 2
    ):
        dump_snapshot("mid")
        snap_done.add("mid")
    if not quit_sent and first_step is not None and last_step >= first_step + stop_after:
        p.stdin.write(b"quit\n")
        p.stdin.flush()
        quit_sent = True
        quit_wall = time.time()

dump_snapshot("end")
with contextlib.suppress(Exception):
    p.stdin.close()
rc = p.wait()
end = time.time()

wall_total = end - first_rec_wall if first_rec_wall else 0.0
mins = max(wall_total / 60, 1e-9)
evt_nb = [e for e in events if not e[6]]
by_fam_min = {f: round(c / mins, 3) for f, c in fam_counts.items()}
by_kind = collections.Counter((e[2], e[3]) for e in evt_nb)
# per 60 s wall window: peak
win = collections.Counter(int((e[0] - first_rec_wall) // 60) for e in evt_nb)
peak_window = max(win.values()) if win else 0
imp = collections.Counter(e[4] for e in evt_nb)
res = {
    "cmd": cmd,
    "exit_code": rc,
    "wall_s": round(end - t0, 2),
    "stream_wall_s": round(wall_total, 2),
    "first_step": first_step,
    "last_step": last_step,
    "raw_bytes": raw_bytes,
    "records_by_tag": dict(tags),
    "bytes_by_tag": dict(bytes_by_tag),
    "decode_total_s": round(decode_s, 1),
    "decode_max_ms": round(max_decode_ms, 1),
    "max_queue_depth": max_q,
    "decode_errors": errors[:20],
    "n_decode_errors": len(errors),
    "trailing_bytes_after_last_record": len(tail_leftover),
    "events_nonbacklog": len(evt_nb),
    "events_backlog": len(events) - len(evt_nb),
    "events_per_min_by_family": by_fam_min,
    "events_per_min_total": round(len(evt_nb) / mins, 3),
    "peak_events_in_a_wall_minute": peak_window,
    "events_by_kind": {f"{a}/{b}": c for (a, b), c in sorted(by_kind.items())},
    "events_by_importance": dict(imp),
    "storm_ids_seen": len(storm_ids),
    "named_storms": len(named_ids),
    "named_storm_names": sorted(named_ids.values())[:40],
    "storm_ended_reasons": dict(storm_ended),
    "focs_records": focs_count,
    "focs_distinct_uid_changes": focs_uid_changes,
    "clck_n": len(clck),
    "clck_last": clck[-1] if clck else None,
    "late_steps_last": clck[-1][5] if clck else None,
    "save_stall_steps_last": clck[-1][6] if clck else None,
    "saving_flag_records": sum(1 for c in clck if c[7]),
    "clck_reasons": dict(collections.Counter(c[8] for c in clck)),
    "achieved_samples": {
        "min": float(np.min([c[3] for c in clck])) if clck else None,
        "p50": float(np.median([c[3] for c in clck])) if clck else None,
        "max": float(np.max([c[3] for c in clck])) if clck else None,
    },
    "step_ms_samples": {
        "p50": float(np.median([c[4] for c in clck])) if clck else None,
        "p99": float(np.percentile([c[4] for c in clck], 99)) if clck else None,
        "max": float(np.max([c[4] for c in clck])) if clck else None,
    },
}
with open(out_json, "w") as f:
    json.dump(res, f, indent=1, default=str)
with open(out_json + ".events.jsonl", "w") as f:
    for e in events:
        f.write(
            json.dumps(
                {
                    "wall_s": round(e[0] - first_rec_wall, 2) if first_rec_wall else None,
                    "step": e[1],
                    "fam": e[2],
                    "kind": e[3],
                    "imp": e[4],
                    "text": e[5],
                    "backlog": e[6],
                }
            )
            + "\n"
        )
with open(out_json + ".clck.jsonl", "w") as f:
    for c in clck:
        f.write(json.dumps(c) + "\n")
print(json.dumps(res, indent=1, default=str))
