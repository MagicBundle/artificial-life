# ruff: noqa: B007, E501, RUF005, SIM105, SIM115, SIM222
# The reader is kept as it ran on the Mac day (2026-10-23), reformatted by `ruff format` only (no logic change); the style rules
# above are waived for this file on purpose so that the code that produced the numbers of docs/RESULTS_G3B.md is not edited.
"""G3b Mac-day reader: attach a strict reader to `alife play` (Metal), decode every record with tests/play_decode.py,
answer chapter pauses with `run`, acknowledge alerts, send one power of each kind plus one cancel, keep counts.

usage: play_reader_g3b.py OUTJSON STOP_AFTER_STEPS [--no-powers] -- alife-play-args...
env: WT (worktree), ALIFE_BIN, SNAPDIR (stream snapshots for the render step)
Quits (sends `quit`) once the stream's step is >= first step + STOP_AFTER_STEPS.
"""

from __future__ import annotations

import collections
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

out_json, stop_after = sys.argv[1], int(sys.argv[2])
use_powers = "--no-powers" not in sys.argv[3 : sys.argv.index("--")]
args = sys.argv[sys.argv.index("--") + 1 :]
SNAPDIR = os.environ.get("SNAPDIR", os.path.dirname(out_json))
os.makedirs(SNAPDIR, exist_ok=True)
cmd = [os.environ["ALIFE_BIN"], "play", *args]
t0 = time.time()
p = subprocess.Popen(
    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(out_json + ".stderr", "wb"), bufsize=0
)
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


threading.Thread(target=reader, daemon=True).start()

hdr_bytes = b""
header: dict = {}
tags = collections.Counter()
bytes_by_tag = collections.Counter()
events = []  # (wall, step, fam, kind, imp, text, backlog)
fam_counts = collections.Counter()
alerts = []  # (wall, step, class, state, surface, imp, pause, title, text, id, watched, calibrated)
reports = []  # dict
plans = []  # dict
clck = []
sent = []  # (wall, step, line)
errors = []
latest = {}
alrt_all = []  # raw ALRT records, in order
rprt_all = []
plan_recent = []
evnt_recent = []
first_step = None
last_step = 0
first_rec_wall = None
quit_sent = False
decode_s = 0.0
max_decode_ms = 0.0
snaps = []

# --- state of the session ---
paused = False
reason = 0
rate_now = 0.0
chapter_handled = set()  # chapter-end steps already answered
acked = set()
ack_due = {}  # alert id -> due wall
run_due = None  # wall time at which to send `run`
pause_log = []  # (wall, step, reason, flags)
watchdog = []
pause_open = None
power_state = {}  # id -> dict(name, line, t_preview, t_confirm, t_applied, status list)
timeline = []  # (offset, kind, power name, id, params string)
next_item = 0
pending_confirm = []  # (due wall, id, line)
power_ids = {"food": 1, "bloom": 2, "cancel": 3, "blight": 4, "strike": 5}


def now() -> float:
    return time.time()


last_clck_key = None
dbg = open(out_json + ".debug.log", "a", buffering=1)
paused_since = None
last_dump = time.time()
last_tick = 0.0


def last_clck_key_set(f):
    global last_clck_key
    last_clck_key = (f["rate"], f["reason"], f["saving"])


def log(msg: str) -> None:
    dbg.write(f"{time.time() - t0:9.2f} step {last_step} {msg}\n")


def send(line: str) -> None:
    global sent
    p.stdin.write((line + "\n").encode())
    p.stdin.flush()
    sent.append((now(), last_step, line))
    log("SEND " + line)


def dump_snapshot(name: str) -> None:
    path = os.path.join(SNAPDIR, f"snap_{name}.bin")
    with open(path, "wb") as f:
        f.write(hdr_bytes)
        for tg in ("IDNT", "INTN", "TERR", "STRM", "MAP1", "ENVS", "FRMA", "FOCS"):
            if tg in latest:
                f.write(latest[tg])
        for e in evnt_recent:
            f.write(e)
        for a in alrt_all:
            f.write(a)
        for r in rprt_all[-3:]:
            f.write(r)
        for pl in plan_recent[-12:]:
            f.write(pl)
        if "CLCK" in latest:
            f.write(latest["CLCK"])
    snaps.append((name, last_step, round(now() - t0, 1), os.path.getsize(path)))


want_timeline = False
items = []  # the scripted powers: dicts name, kind, at (offset or absolute step), params, tries, state, id
next_id = 1


def build_timeline() -> None:
    lo_hi = header["play"]["powers"]["strike"]["params"]["radius"]
    size = float(header["L"])
    radius = float(lo_hi[0])
    if os.environ.get("READER_TEST_NOW"):
        pass
    items.extend(
        [
            {
                "name": "food",
                "kind": "power",
                "at": first_step + 500,
                "params": "mult=1.3 steps=6000",
                "tries": 0,
                "state": "todo",
            },
            {
                "name": "bloom",
                "kind": "power",
                "at": first_step + 1500,
                "params": "patch=2 gain=1.75 steps=3000",
                "tries": 0,
                "state": "todo",
            },
            {
                "name": "strike",
                "kind": "power",
                "at": first_step + 3000,
                "params": f"x={size / 2:g} y={size / 2:g} radius={radius:g} kill=0.45",
                "tries": 0,
                "state": "todo",
            },
            {
                "name": "blight",
                "kind": "power",
                "at": first_step + 6000,
                "params": "patch=0 steps=3000",
                "tries": 0,
                "state": "todo",
            },
            {
                "name": "food",
                "kind": "cancel",
                "after": "food",
                "delay": 1500,
                "at": None,
                "params": "",
                "tries": 0,
                "state": "todo",
            },
        ]
    )


def _test_now() -> None:
    for it in items:
        if it["at"] is not None:
            it["at"] = first_step


def item_of(pid: int):
    return next((i for i in items if i.get("id") == pid), None)


def drive() -> None:
    """Called on every loop turn: answer pauses, acks, due power lines."""
    global run_due, next_item
    t = now()
    # alerts to acknowledge
    for aid, due in list(ack_due.items()):
        if t >= due:
            send(f"ack {aid}")
            acked.add(aid)
            del ack_due[aid]
    global paused_since, last_dump, last_tick
    if paused and paused_since is not None and t - paused_since > 20 and run_due is None and not quit_sent:
        log(f"WATCHDOG: paused for {t - paused_since:.0f} s with reason {reason}: sending run")
        watchdog.append((t, last_step, reason))
        send("run")
        paused_since = t
    if t - last_dump > 90 and clck:
        last_dump = t
        try:
            write_results(None, t)
        except Exception as e:  # a partial dump must never stop the session
            log(f"partial dump failed: {e!r}")
    if paused and t - last_tick > 5:
        last_tick = t
        log(
            f"TICK paused={paused} reason={reason} run_due={run_due} paused_since={paused_since} now={t} quit_sent={quit_sent} ack_due={sorted(ack_due)}"
        )
    if run_due is not None and t >= run_due and paused:
        send("run")
        run_due = None
    elif run_due is not None and not paused:
        run_due = None
    # confirmations
    for item in list(pending_confirm):
        due, pid, line = item
        if t >= due:
            send(line)
            power_state[pid]["t_confirm"] = t
            pending_confirm.remove(item)
    # the scripted powers
    global next_id
    if want_timeline and first_step is not None and not items:
        build_timeline()
        if os.environ.get("READER_TEST_NOW"):
            _test_now()
    if use_powers and first_step is not None and not quit_sent:
        for it in items:
            if it["state"] != "todo":
                continue
            if it["kind"] == "cancel":
                dep = next((i for i in items if i["kind"] == "power" and i["name"] == it["after"]), None)
                if dep is None or dep["state"] != "done" or last_step < dep["step_applied"] + it["delay"]:
                    continue
            elif last_step < it["at"]:
                continue
            pid = next_id
            next_id += 1
            it["id"], it["state"] = pid, "inflight"
            if it["kind"] == "cancel":
                power_state[pid] = {
                    "name": "cancel:" + it["name"],
                    "t_preview": None,
                    "t_confirm": t,
                    "statuses": [],
                }
                send(f"power cancel {it['name']} id={pid}")
            else:
                power_state[pid] = {"name": it["name"], "t_preview": t, "t_confirm": None, "statuses": []}
                send(f"power preview {it['name']} id={pid} {it['params']}")


# ----------------------------------------------------------------------------------------------------------
def write_results(rc, end):
    wall_total = end - first_rec_wall if first_rec_wall else 0.0
    # run time (the wall time not spent paused) is the right denominator for the per-minute rates of a played world
    paused_wall = 0.0
    for i in range(len(clck) - 1):
        if clck[i][2] == 0.0:
            paused_wall += clck[i + 1][0] - clck[i][0]
    run_wall = max(wall_total - paused_wall, 1e-9)
    mins = max(wall_total / 60, 1e-9)
    run_mins = run_wall / 60
    evt_nb = [e for e in events if not e[6]]
    fam_total = collections.Counter(e[2] for e in evt_nb)
    by_kind = collections.Counter((e[2], e[3]) for e in evt_nb)
    alerts_nb = [a for a in alerts if not a[12]]
    raised = [a for a in alerts_nb if a[3] == "raised"]
    by_class = collections.Counter((a[2], a[4], a[5]) for a in raised)
    alert_surface_imp2 = [a for a in raised if a[4] == "alert" and a[5] >= 2]

    # chapter ends: the report time
    chapter_ends = []
    for w0, s0, rate, ach, sm, late, stall, saving, rsn, fl, ch, inch in clck:
        if rsn == 3 and saving == 1:
            chapter_ends.append({"step": s0, "chapter": ch, "t_card": w0})
    for ce in chapter_ends:
        rep = [
            x for x in reports if not x["backlog"] and x["wall"] >= ce["t_card"] and x["step"] == ce["step"]
        ]
        done = [c for c in clck if c[0] >= ce["t_card"] and c[1] == ce["step"] and c[7] == 0]
        ce["t_report_s"] = round(rep[0]["wall"] - ce["t_card"], 3) if rep else None
        ce["t_save_done_s"] = round(done[0][0] - ce["t_card"], 3) if done else None
        ce["report_inside_stall"] = bool(rep and done and rep[0]["wall"] <= done[0][0])
        ce["report_headline"] = rep[0]["headline"] if rep else None
        ce["report_lead"] = rep[0]["lead"] if rep else None

    # powers
    pw = {}
    for pid, st in power_state.items():
        statuses = [s[0] for s in st["statuses"]]
        pw[str(pid)] = {
            "name": st["name"],
            "statuses": statuses,
            "preview_reply_s": round(
                next(s[1] for s in st["statuses"] if s[0] in ("preview", "refused")) - st["t_preview"], 3
            )
            if st["t_preview"] and statuses
            else None,
            "click_to_applied_s": round(st["t_applied"] - st["t_confirm"], 3)
            if st.get("t_applied") and st["t_confirm"]
            else None,
            "step_applied": st.get("step_applied"),
        }
    plan_counts = collections.Counter((str(x["power"]), x["status"]) for x in plans)
    paused_by = collections.Counter(("chapter" if x[2] == 3 else "alert") for x in pause_log)
    res = {
        "cmd": cmd,
        "exit_code": rc,
        "wall_s": round(end - t0, 2),
        "stream_wall_s": round(wall_total, 2),
        "paused_wall_s": round(paused_wall, 2),
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
        "events_nonbacklog": len(evt_nb),
        "events_backlog": len(events) - len(evt_nb),
        "events_per_wall_min_by_family": {f: round(c / mins, 3) for f, c in fam_total.items()},
        "events_per_wall_min_total": round(len(evt_nb) / mins, 3),
        "events_per_run_min_by_family": {f: round(c / run_mins, 3) for f, c in fam_total.items()},
        "events_per_run_min_total": round(len(evt_nb) / run_mins, 3),
        "events_by_kind": {f"{a}/{b}": c for (a, b), c in sorted(by_kind.items())},
        "alerts_raised": len(raised),
        "alerts_raised_by_class_surface_imp": {
            f"{a}/{b}/{c}": n for (a, b, c), n in sorted(by_class.items())
        },
        "alerts_alert_surface_imp_ge2": len(alert_surface_imp2),
        "alerts_alert_surface_imp_ge2_per_wall_min": round(len(alert_surface_imp2) / mins, 3),
        "alerts_alert_surface_imp_ge2_per_run_min": round(len(alert_surface_imp2) / run_mins, 3),
        "alerts_states": dict(collections.Counter(a[3] for a in alerts_nb)),
        "alerts_backlog": len(alerts) - len(alerts_nb),
        "acks_sent": sum(1 for s in sent if s[2].startswith("ack ")),
        "pauses": [
            {
                "wall_s": round(x[0] - first_rec_wall, 2),
                "step": x[1],
                "reason": x[2],
                "flags": x[3],
                "chapter": x[4],
            }
            for x in pause_log
        ],
        "pauses_by_cause": dict(paused_by),
        "chapter_ends": chapter_ends,
        "reports": [{k: v for k, v in x.items() if k != "wall"} for x in reports],
        "powers": pw,
        "power_items": items,
        "watchdog": [(round(w[0] - first_rec_wall, 1), w[1], w[2]) for w in watchdog],
        "partial": rc is None,
        "plan_counts": {f"{a}/{b}": c for (a, b), c in sorted(plan_counts.items())},
        "commands_sent": [(round(s[0] - first_rec_wall, 2), s[1], s[2]) for s in sent],
        "snapshots": snaps,
        "clck_n": len(clck),
        "clck_reasons": dict(collections.Counter(c[8] for c in clck)),
        "late_steps_last": clck[-1][5] if clck else None,
        "save_stall_steps_last": clck[-1][6] if clck else None,
        "achieved_running": {
            "p50": float(np.median([c[3] for c in clck if c[2] != 0])) if clck else None,
            "min": float(np.min([c[3] for c in clck if c[2] != 0])) if clck else None,
        },
        "step_ms_running": {
            "p50": float(np.median([c[4] for c in clck if c[2] != 0])) if clck else None,
            "p99": float(np.percentile([c[4] for c in clck if c[2] != 0], 99)) if clck else None,
            "max": float(np.max([c[4] for c in clck if c[2] != 0])) if clck else None,
        },
    }
    json.dump(res, open(out_json, "w"), indent=1, default=str)
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
    with open(out_json + ".alerts.jsonl", "w") as f:
        for a in alerts:
            f.write(
                json.dumps(
                    {
                        "wall_s": round(a[0] - first_rec_wall, 2),
                        "step": a[1],
                        "class": a[2],
                        "state": a[3],
                        "surface": a[4],
                        "imp": a[5],
                        "pause": a[6],
                        "title": a[7],
                        "text": a[8],
                        "id": a[9],
                        "watched": a[10],
                        "calibrated": a[11],
                        "backlog": a[12],
                    }
                )
                + "\n"
            )
    with open(out_json + ".plans.jsonl", "w") as f:
        for x in plans:
            f.write(json.dumps({**x, "wall_s": round(x["wall"] - first_rec_wall, 2)}, default=str) + "\n")
    with open(out_json + ".clck.jsonl", "w") as f:
        for c in clck:
            f.write(json.dumps([round(c[0] - first_rec_wall, 3), *c[1:]]) + "\n")
    return res


while True:
    try:
        kind, wall, data = q.get(timeout=0.2)
    except queue.Empty:
        drive()
        if time.time() - t0 > 3 * 3600:
            p.kill()
        continue
    max_q = max(max_q, q.qsize())
    if kind == "hdr":
        hdr_bytes = data
        header, _ = decode_header(data)
        want_timeline = use_powers and "powers" in header.get("play", {})
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
        drive()
        continue
    if d.skipped:
        errors.append(f"skipped {d.skipped}")
    if not d.records:
        drive()
        continue
    r = d.records[0]
    tags[r.tag] += 1
    bytes_by_tag[r.tag] += len(data)
    latest[r.tag] = data
    if first_step is None:
        first_step = r.step
        first_rec_wall = wall
    last_step = max(last_step, r.step)
    if r.tag == "EVNT":
        evnt_recent.append(data)
        del evnt_recent[:-40]
        for e in r.f["json"]:
            events.append((wall, r.step, e["fam"], e["kind"], e["imp"], e["text"], r.f["backlog"]))
            if not r.f["backlog"]:
                fam_counts[e["fam"]] += 1
    elif r.tag == "ALRT":
        alrt_all.append(data)
        new_raise = False
        for a in r.f["json"]:
            alerts.append(
                (wall, r.step, a["class"], a["state"], a["surface"], a["imp"], a["pause"], a["title"], a["text"],
                 a["id"], a["watched"], a["calibrated"], r.f["backlog"])
            )  # fmt: skip
            if a["pause"] and not r.f["backlog"]:
                log(
                    f"ALRT pause {a['class']} {a['state']} imp {a['imp']} id {a['id']} acked {a['id'] in acked}"
                )
            if a["state"] in ("raised", "updated") and a["surface"] == "alert" and not r.f["backlog"]:
                new_raise = True
                if a["id"] not in acked and a["id"] not in ack_due:
                    ack_due[a["id"]] = now() + 2.0
        n_alert_snaps = sum(1 for s in snaps if s[0].startswith("alerts"))
        if new_raise and n_alert_snaps < 6:
            # a snapshot every time the open set grows, at most six
            dump_snapshot(f"alerts_{n_alert_snaps + 1}")
    elif r.tag == "RPRT":
        rprt_all.append(data)
        o = r.f["json"]
        reports.append(
            {"wall": wall, "step": r.step, "chapter": r.f["chapter"], "mode": r.f["mode"], "backlog": r.f["backlog"],
             "final": r.f["final"], "withheld": o["withheld"], "headline": o["headline"]["text"] if o["headline"] else None,
             "lead": o["lead"]["text"] if o["lead"] else None, "n_body": len(o["body"]), "n_interventions": len(o["interventions"]),
             "latency_s": o["latency_s"], "fallback": o["fallback"], "bytes": len(data)}
        )  # fmt: skip
    elif r.tag == "PLAN":
        o = r.f["json"]
        _st = power_state.get(o["id"])
        if _st is not None and _st["name"] == "strike" and o["status"] == "preview" and r.f["reply"]:
            dump_snapshot(
                "strike_preview"
            )  # without this record: the probe re-creates the request and feeds the reply
            with open(os.path.join(SNAPDIR, "snap_strike_preview_plan.json"), "w") as fh:
                json.dump(o, fh)
        plan_recent.append(data)
        del plan_recent[:-40]
        plans.append(
            {
                "wall": wall,
                "step": r.step,
                "reply": r.f["reply"],
                **{
                    k: o[k]
                    for k in (
                        "id",
                        "kind",
                        "status",
                        "power",
                        "step_applied",
                        "ends",
                        "cost",
                        "influence",
                        "refused",
                        "text",
                    )
                },
            }
        )
        st = power_state.get(o["id"])
        if st is not None:
            st["statuses"].append((o["status"], wall, o["step_applied"]))
            if o["status"] == "preview" and st["t_confirm"] is None and not st["name"].startswith("cancel"):
                line = next(
                    l
                    for (_, _, l) in reversed(sent)
                    if l.startswith("power preview") and f"id={o['id']} " in l + " "
                )
                conf = line.replace("power preview ", "power ", 1)
                pending_confirm.append((now() + (1.0 if st["name"] == "strike" else 0.3), o["id"], conf))
            it = item_of(o["id"])
            if it is not None and o["status"] == "refused":
                import re

                m = re.search(r"until step ([\d,]+)", (o["refused"] or "") + " " + o["text"])
                it["tries"] += 1
                it["refusals"] = it.get("refusals", []) + [o["refused"] or o["text"]]
                if it["tries"] >= 4:
                    it["state"] = "failed"
                else:
                    it["at"] = int(m.group(1).replace(",", "")) + 10 if m else last_step + 500
                    it["state"] = "todo"
            if o["status"] == "applied" and st.get("t_applied") is None:
                st["t_applied"] = wall
                st["step_applied"] = o["step_applied"]
                if it is not None:
                    it["state"], it["step_applied"] = "done", o["step_applied"]
            if it is not None and o["status"] == "cancelled":
                it["state"] = "done"
                it["step_applied"] = o["step_applied"] or last_step
                if st["name"] in ("strike", "food") and st["name"] + "_applied" not in [s[0] for s in snaps]:
                    pass
    elif r.tag == "CLCK":
        f = r.f
        clck.append((wall, r.step, f["rate"], f["achieved"], f["step_ms"], f["late_steps"], f["save_stall_steps"],
                     f["saving"], f["reason"], f["flags"], f["chapter"], f["in_chapter"]))  # fmt: skip
        was_paused = paused
        paused = f["rate"] == 0.0
        if paused and not was_paused:
            paused_since = time.time()
        if not paused:
            paused_since = None
        if (f["rate"], f["reason"], f["saving"]) != last_clck_key:
            log(
                f"CLCK rate {f['rate']} reason {f['reason']} saving {f['saving']} flags {f['flags']} chapter {f['chapter']} in {f['in_chapter']} achieved {f['achieved']:.2f}"
            )
            last_clck_key_set(f)
        was_reason = reason
        reason = f["reason"]
        if paused and f["reason"] in (3, 4):
            if pause_open is None or pause_open[2] != f["reason"]:
                pause_log.append((wall, r.step, f["reason"], f["flags"], f["chapter"]))
                pause_open = (wall, r.step, f["reason"])
            if f["saving"] == 0:
                if f["reason"] == 3 and r.step not in chapter_handled:
                    chapter_handled.add(r.step)
                    n = len(chapter_handled)
                    # let the report arrive and the player 'read': snapshot first, then run
                    dump_snapshot(f"chapter_{n}")
                    run_due = now() + 2.0
                elif (
                    f["reason"] == 4 and run_due is None
                ):  # (the engine repeats the CLCK while paused: set it once)
                    run_due = now() + 2.5
        elif not paused:
            pause_open = None
        if f["reason"] == 3 and f["saving"] == 1 and not paused:
            pass
    if (
        not quit_sent
        and first_step is not None
        and last_step >= first_step + stop_after
        and (not paused or True)
    ):
        send("quit")
        quit_sent = True
    drive()

dump_snapshot("end")
try:
    p.stdin.close()
except Exception:
    pass
rc = p.wait()
end = now()


res = write_results(rc, end)
print(json.dumps(res, indent=1, default=str))
