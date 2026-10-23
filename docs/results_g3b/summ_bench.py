"""Summarise `alife bench --play` JSON files (the G3b Mac day's step 4): one block per arm.

usage: python summ_bench.py BENCH.json [BENCH.json ...]
"""

import json
import sys


def short(digests):
    return {k: v[:12] for k, v in digests.items()}


for path in sys.argv[1:]:
    with open(path) as fh:
        d = json.load(fh)["play"]
    print(
        "==",
        path.split("/")[-1],
        "engine_ms",
        round(d["engine_ms"], 2),
        "digests_equal",
        d["digests_equal"],
        "plain",
        short(d["plain_digest"]),
        "replay",
        short(d["replay_digest"]),
    )
    for a in d["arms"]:
        sm = a["step_ms"]
        print(
            f"{a['name']:10s} ach {a['achieved_x']:.5f}x sec {a['seconds']:.1f} "
            f"step_ms p50 {sm['p50']:.1f} p99 {sm['p99']:.1f} max {sm['max']:.1f} "
            f"late {a['late_steps']} stall {a['save_stall_steps']} digest {a['digest'][:12]} "
            f"exp {a['expected']} eq {a['digest_equal']} n_alive_end {a['n_alive_end']}"
        )
        gate = a["gate"] or {}
        print("    gate", gate.get("pass"), gate.get("got"), "checks", gate.get("checks"))
        print("    chapters", a.get("chapters"), "alerts", a.get("alerts"))
        if a.get("powers"):
            p = a["powers"]
            print(
                "    powers", {k: ([round(x, 3) for x in v] if k == "latency_s" else v) for k, v in p.items()}
            )
        print(
            "    components",
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a["components_ms"].items()},
        )
        print("    step kinds typical/boundary", a.get("step_ms_typical"), a.get("step_ms_boundary"))
