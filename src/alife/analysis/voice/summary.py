"""The voice log's summary: what each lineage says, in what situation, what hearers did next, and drift.

Built from the coded tables of one window (``analyse.Prepared``). Everything is JSON-safe (``None`` for an
undefined number). ``to_markdown`` writes the human-readable page; ``compare_windows`` compares the
summaries of
two windows of one world (dialect drift between them).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alife.analysis.voice import analyse, spec, tables
from alife.analysis.voice import measures as ms
from alife.analysis.voice.tables import Tables

SHARE_MIN = 0.01  # a call type with at least this share of the calls is part of the vocabulary


def symbol_name(sym: int) -> str:
    """``silent`` or ``ch<c>+`` / ``ch<c>-``: the loudest channel and whether it is above the median."""
    if sym == 0:
        return "silent"
    c, loud = divmod(sym - 1, 2)
    return f"ch{c}{'+' if loud else '-'}"


def _f(v: Any) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def _shares(sym: np.ndarray, n_sym: int) -> np.ndarray:
    return np.bincount(sym.astype(np.int64), minlength=n_sym) / max(len(sym), 1)


def population(p: analyse.Prepared, tab: Tables) -> dict[str, Any]:
    v = tab.senders["voice"].astype(np.float64) / 255.0
    sh = _shares(p.s_sym, p.n_sym)
    return {
        "n": len(p.s_sym),
        "speaker_frac": _f((p.s_sym > 0).mean()),
        "vocabulary": int((sh[1:] > SHARE_MIN).sum()),
        "entropy_bits": _f(ms.entropy_bits(p.s_sym, p.n_sym)),
        "entropy_max_bits": _f(math.log2(p.n_sym)),
        "mean_voice": [_f(x) for x in v.mean(axis=0)],
        "channel_on": [_f(x) for x in (v > 0).mean(axis=0)],
        "symbols": {symbol_name(i): _f(s) for i, s in enumerate(sh)},
        "intensity_split": p.sym_thr / 255.0,
    }


def _enrichment(sym: np.ndarray, code: np.ndarray, n_sym: int) -> np.ndarray:
    """``E[s] = mean(code / max | S = s) - mean(code / max)`` (NaN for a call type that does not occur)."""
    x = code.astype(np.float64) / max(int(code.max(initial=1)), 1)
    out = np.full(n_sym, np.nan)
    for s in range(n_sym):
        m = sym == s
        if m.any():
            out[s] = x[m].mean() - x.mean()
    return out


def context_code_matrix(p: analyse.Prepared) -> dict[str, list[float | None]]:
    """For each usable context the enrichment of the context level given each call type (population)."""
    return {k: [_f(e) for e in _enrichment(p.s_sym, p.s_ctx[k], p.n_sym)] for k in analyse.usable(p.s_ctx)}


def _lineage_code(
    p: analyse.Prepared, rows: np.ndarray, rng: np.random.Generator, n_perm: int
) -> dict[str, Any] | None:
    """The best context of the call type inside one lineage (shuffle within sample step)."""
    if rows.sum() < 200:
        return None
    strata = ms.compact(p.s_step[rows])[0]
    pairs = [
        ms.Pair(f"S~{k}", p.s_sym[rows], p.n_sym, p.s_ctx[k][rows], tables.n_levels(p.s_ctx[k][rows]), strata)
        for k in analyse.usable({k: v[rows] for k, v in p.s_ctx.items()})
    ]
    if not pairs:
        return None
    r = ms.family_test(pairs, n_perm, rng, spec.MIN_STRATUM)
    return r.best_summary()


def lineages(p: analyse.Prepared, tab: Tables, top: int, rng: np.random.Generator, n_perm: int) -> list[dict]:
    u, c = np.unique(p.s_lineage, return_counts=True)
    order = np.argsort(-c, kind="stable")[:top]
    base = _shares(p.s_sym, p.n_sym)
    ctx_names = analyse.usable(p.s_ctx)
    out = []
    for i in order:
        lineage = int(u[i])
        rows = p.s_lineage == lineage
        sh = _shares(p.s_sym[rows], p.n_sym)
        top_syms = [int(s) for s in np.argsort(-sh)[:3] if sh[s] > 0]
        vm = (tab.senders["voice"][rows].astype(np.float64) / 255.0).mean(axis=0)
        # the context of the lineage's commonest non-silent call against the population at large
        ns = [s for s in top_syms if s != 0]
        assoc = {}
        if ns:
            m = rows & (p.s_sym == ns[0])
            if m.sum() >= 30:
                for k in ctx_names:
                    hi = p.s_ctx[k] == p.s_ctx[k].max()
                    assoc[k] = {"p_given_call": _f(hi[m].mean()), "p_population": _f(hi.mean())}
        heard = p.h_has_src & (p.h_src_lineage == lineage)
        out.append(
            {
                "lineage": lineage,
                "n_senders": int(rows.sum()),
                "share": _f(rows.mean()),
                "speaker_frac": _f((p.s_sym[rows] > 0).mean()),
                "mean_voice": [_f(x) for x in vm],
                "entropy_bits": _f(ms.entropy_bits(p.s_sym[rows], p.n_sym)),
                "top_calls": [
                    {"call": symbol_name(s), "share": _f(sh[s]), "population_share": _f(base[s])}
                    for s in top_syms
                ],
                "top_call_context": {"call": symbol_name(ns[0]), "contexts": assoc} if ns and assoc else None,
                "code": _lineage_code(p, rows, rng, n_perm),
                "heard_by": {"n_hearers": int(heard.sum())},
            }
        )
    return out


def hearers_after(p: analyse.Prepared, top: int) -> dict[str, Any]:
    """What hearers did next, by the call of their loudest audible source: response rates for each call type
    against all hearers that had a source (rows are hearers with a source)."""
    m = p.h_has_src
    resp = {
        "attack": p.h_resp["attack"][m] > 0,
        "bond": p.h_resp["bond"][m] > 0,
        "eat": p.h_resp["eat"][m] > 0,
        "speak": p.h_resp["speak"][m] > 0,
        "toward": p.h_resp["approach"][m] == 2,
        "away": p.h_resp["approach"][m] == 0,
    }
    sym = p.h_src_sym[m]
    base = {k: _f(v.mean()) for k, v in resp.items()}
    rows = []
    for s in np.argsort(-np.bincount(sym.astype(np.int64), minlength=p.n_sym))[:top]:
        sel = sym == s
        if sel.sum() < 100:
            continue
        rows.append(
            {
                "source_call": symbol_name(int(s)),
                "n": int(sel.sum()),
                **{k: _f(v[sel].mean()) for k, v in resp.items()},
            }
        )
    return {"n": int(m.sum()), "baseline": base, "by_source_call": rows}


def _js_pair(a: np.ndarray, b: np.ndarray, n_sym: int, min_n: int = 100) -> float | None:
    if len(a) < min_n or len(b) < min_n:
        return None
    return _f(
        ms.jensen_shannon(
            np.bincount(a.astype(np.int64), minlength=n_sym), np.bincount(b.astype(np.int64), minlength=n_sym)
        )
    )


def drift(p: analyse.Prepared, tab: Tables, top: int) -> dict[str, Any]:
    """Dialect drift inside the window: the call-type distribution of the first quarter of the blocks
    against the
    last quarter (Jensen-Shannon, bits; 0 = same, 1 = disjoint), for the population and each top lineage, the
    shift of the mean call, and the Jensen-Shannon distance of every block to the first."""
    n_blocks = p.n_blocks
    q = max(n_blocks // 4, 1)
    first, last = (p.s_block >= 0) & (p.s_block < q), p.s_block >= n_blocks - q
    v = tab.senders["voice"].astype(np.float64) / 255.0
    out: dict[str, Any] = {
        "first_blocks": q,
        "population_js": _js_pair(p.s_sym[first], p.s_sym[last], p.n_sym),
        "population_mean_voice_shift": _f(np.linalg.norm(v[first].mean(0) - v[last].mean(0)))
        if first.any() and last.any()
        else None,
        "by_block": [
            _js_pair(p.s_sym[p.s_block == 0], p.s_sym[p.s_block == b], p.n_sym) for b in range(n_blocks)
        ],
        "lineages": [],
    }
    u, c = np.unique(p.s_lineage, return_counts=True)
    for i in np.argsort(-c, kind="stable")[:top]:
        rows = p.s_lineage == u[i]
        a, b = rows & first, rows & last
        out["lineages"].append(
            {
                "lineage": int(u[i]),
                "js": _js_pair(p.s_sym[a], p.s_sym[b], p.n_sym),
                "mean_voice_shift": _f(np.linalg.norm(v[a].mean(0) - v[b].mean(0)))
                if a.sum() >= 30 and b.sum() >= 30
                else None,
                "n_first": int(a.sum()),
                "n_last": int(b.sum()),
            }
        )
    # the code itself: the call-type to context association in the two halves
    half = n_blocks // 2
    h1, h2 = (p.s_block >= 0) & (p.s_block < half), p.s_block >= half
    code = {}
    for k in analyse.usable(p.s_ctx):
        n_k = tables.n_levels(p.s_ctx[k])
        code[k] = {
            "mi_first_half": _f(ms.mi_codes(p.s_sym[h1], p.s_ctx[k][h1], p.n_sym, n_k))
            if h1.sum() > 100
            else None,
            "mi_second_half": _f(ms.mi_codes(p.s_sym[h2], p.s_ctx[k][h2], p.n_sym, n_k))
            if h2.sum() > 100
            else None,
        }
    out["code_by_half"] = code
    return out


def counterfactual(tab: Tables) -> dict[str, Any] | None:
    """The counterfactual samples, averaged: for each other control the effect on the audible rows and on the
    silent control group (see ``replay.counterfactual``)."""
    if "stats" not in tab.cf:
        return None
    st = tab.cf["stats"].astype(np.float64)  # (samples, modes, groups, metrics)
    names = tab.meta["cf_metrics"]
    out: dict[str, Any] = {"samples": int(st.shape[0]), "own_control": tab.meta["control"], "modes": {}}
    for i, mode in enumerate(tab.meta["cf_modes"]):
        grp = {}
        for g, gname in enumerate(("audible", "silent")):
            mean = ms.nanmean(st[:, i, g, :], axis=0)
            n_ok = np.isfinite(st[:, i, g, :]).sum(axis=0)
            var = ms.nanmean((st[:, i, g, :] - mean) ** 2, axis=0) * n_ok / np.maximum(n_ok - 1, 1)
            sd = np.where(n_ok > 1, np.sqrt(var), np.nan)
            grp[gname] = {n: {"mean": _f(mean[j]), "sd": _f(sd[j])} for j, n in enumerate(names)}
        out["modes"][mode] = grp
    return out


def build(
    tab: Tables,
    window: tuple[int, int],
    *,
    n_perm: int = spec.N_PERM,
    n_perm_block: int = spec.N_PERM_BLOCK,
    top: int = spec.TOP_LINEAGES,
    seed: int = spec.SEED,
    n_blocks: int = spec.N_BLOCKS,
    with_tests: bool = True,
    log: Any = lambda m: None,
) -> dict[str, Any]:
    """The whole summary of one window (JSON-safe)."""
    p = analyse.prepare(tab, window, n_blocks)
    rng = np.random.default_rng([seed, 1])
    out: dict[str, Any] = {
        "run": tab.meta["run"],
        "window": list(window),
        "control": tab.meta.get("control"),
        "config_hash": tab.meta.get("config_hash"),
        "geometry": tab.meta.get("geometry"),
        "meta": {
            k: tab.meta.get(k)
            for k in (
                "every",
                "rows",
                "seed",
                "samples",
                "rows_verified",
                "end_digest",
                "device",
                "mlx_version",
                "hear_range",
            )
        },
        "population": population(p, tab),
        "context_code": context_code_matrix(p),
        "lineages": lineages(p, tab, top, rng, max(n_perm // 4, 20)),
        "hearers_after": hearers_after(p, 6),
        "drift": drift(p, tab, top),
        "counterfactual": counterfactual(tab),
    }
    if with_tests:
        out["tests"] = analyse.run_all(
            tab,
            window,
            n_perm=n_perm,
            n_perm_block=n_perm_block,
            seed=seed,
            n_blocks=n_blocks,
            prepared=p,
            log=log,
        )
    return out


def _fmt(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _pct(v: Any, nd: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{nd}f} %"


def to_markdown(s: dict[str, Any]) -> str:
    """The human-readable page of a summary."""
    pop, meta = s["population"], s["meta"]
    ed = meta["end_digest"]
    head = (
        f"Window steps {s['window'][0]} to {s['window'][1]}, control `{s['control']}`, "
        f"{meta['samples']} samples of {meta['rows']} rows every {meta['every']} steps, "
        f"{meta['rows_verified']} metrics rows re-verified"
    )
    if ed:
        head += f", end state equal to the checkpoint: {ed['equal']}"
    lines = [
        f"# Voice log: {s['run']}",
        "",
        head + ".",
        "",
        "## Population",
        "",
        f"- {_pct(pop['speaker_frac'])} of the sampled rows speak; vocabulary (call types above 1 %): "
        f"{pop['vocabulary']} of {len(pop['symbols']) - 1}; entropy {_fmt(pop['entropy_bits'])} of "
        f"{_fmt(pop['entropy_max_bits'])} bits.",
        "- Mean intensity per channel: " + ", ".join(_fmt(x) for x in pop["mean_voice"]) + ".",
        "- Call types: " + ", ".join(f"{k} {_pct(v)}" for k, v in pop["symbols"].items()) + ".",
        "",
        "## Tag clusters: top calls, context, response",
        "",
        "A cluster is a k-means slice of the kin tags at the window start (one continuous cloud); "
        "a birth inherits its parent's cluster. Descriptive strata, not lineages.",
        "",
        _row(
            "cluster",
            "share",
            "speaks",
            "top calls (cluster / population share)",
            "best context",
            "excess bits",
            "z",
        ),
        _row(*["---"] * 7),
    ]
    for ln in s["lineages"]:
        calls = ", ".join(
            f"{c['call']} {_pct(c['share'], 0)} / {_pct(c['population_share'], 0)}" for c in ln["top_calls"]
        )
        code = ln["code"]
        best = (
            (code["pair"].split("~")[1], _fmt(code["excess"], 4), _fmt(code["z"], 1))
            if code
            else ("n/a",) * 3
        )
        lines.append(_row(ln["lineage"], _pct(ln["share"]), _pct(ln["speaker_frac"], 0), calls, *best))
    lines += [
        "",
        "Context of each cluster's commonest call (share of its callers at the highest level of a context, "
        "against the population):",
        "",
    ]
    for ln in s["lineages"]:
        tc = ln["top_call_context"]
        if tc:
            ctx = ", ".join(
                f"{k} {_fmt(v['p_given_call'])} vs {_fmt(v['p_population'])}"
                for k, v in tc["contexts"].items()
            )
            lines.append(f"- cluster {ln['lineage']}, call {tc['call']}: {ctx}")
    ha = s["hearers_after"]
    base = ", ".join(f"{k} {_fmt(v)}" for k, v in ha["baseline"].items())
    cols = ("attack", "bond", "eat", "speak", "toward", "away")
    lines += [
        "",
        "## What hearers did next, by the call of their loudest source",
        "",
        f"{ha['n']} hearers with a source. Baseline: {base}. (`eat` is harvest above the lowest tercile.)",
        "",
        _row("source call", "n", *cols),
        _row(*["---"] * (2 + len(cols))),
    ]
    for r in ha["by_source_call"]:
        lines.append(_row(r["source_call"], r["n"], *(_fmt(r[k]) for k in cols)))
    d = s["drift"]
    lines += [
        "",
        "## Dialect drift inside the window",
        "",
        f"Population call types, first {d['first_blocks']} blocks against the last: "
        f"JS {_fmt(d['population_js'], 4)} bits "
        f"(mean-call shift {_fmt(d['population_mean_voice_shift'], 4)}).",
        "",
        _row("cluster", "JS first/last (bits)", "mean-call shift", "n first / last"),
        _row(*["---"] * 4),
    ]
    for ln in d["lineages"]:
        lines.append(
            _row(
                ln["lineage"],
                _fmt(ln["js"], 4),
                _fmt(ln["mean_voice_shift"], 4),
                f"{ln['n_first']} / {ln['n_last']}",
            )
        )
    t = s.get("tests")
    if t:
        lines += [
            "",
            "## Information tests (shuffle null, family p)",
            "",
            _row("test", "best pair", "MI (bits)", "excess", "z", "p (family)"),
            _row(*["---"] * 6),
        ]
        for fam, r in t["families"].items():
            if r is None:
                lines.append(_row(fam, "not enough rows", "", "", "", ""))
                continue
            b = r["best"]
            lines.append(
                _row(
                    fam,
                    b["pair"],
                    _fmt(b["mi"], 4),
                    _fmt(b["excess"], 4),
                    _fmt(b["z"], 1),
                    _fmt(b["p_family"], 4),
                )
            )
    cf = s.get("counterfactual")
    if cf:
        lines += ["", "## Counterfactual: one step under another control (decisions that change)", ""]
        for mode, grp in cf["modes"].items():
            a, q = grp["audible"], grp["silent"]

            def m(k: str, a: dict = a) -> str:
                return _fmt(a[k]["mean"], 4)

            lines.append(
                f"- `{cf['own_control']}` -> `{mode}`, rows with an audible neighbour: "
                f"|thrust| change {m('d_thrust')}, "
                f"attack flips {m('flip_attack')}, bond-target flips {m('flip_bond')}, speech flips "
                f"{m('flip_speak')}, call change {m('d_voice')}; rows with none (control group): call change "
                f"{_fmt(q['d_voice']['mean'], 4)} (exactly 0 expected)."
            )
    return "\n".join(lines) + "\n"


def compare_windows(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Dialect drift between two windows of one world (summaries from ``build``): the Jensen-Shannon distance
    of the population's call-type distributions, the shift of its mean call, and the stability of the code
    (the correlation of the call-type to context enrichment tables). The tag clusters are re-clustered at the
    start of each window, so their labels do not name the same lines of descent: not compared."""
    sa, sb = a["population"]["symbols"], b["population"]["symbols"]
    names = list(sa)
    pa, pb = np.array([sa[n] for n in names]), np.array([sb[n] for n in names])
    ca, cb = a["context_code"], b["context_code"]
    xs, ys = [], []
    for k in ca.keys() & cb.keys():
        for x, y in zip(ca[k], cb[k], strict=True):
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 and np.std(xs) > 0 and np.std(ys) > 0 else None
    va, vb = np.array(a["population"]["mean_voice"], float), np.array(b["population"]["mean_voice"], float)
    return {
        "windows": [a["window"], b["window"]],
        "population_js": _f(ms.jensen_shannon(pa, pb)),
        "population_mean_voice_shift": _f(np.linalg.norm(va - vb)),
        "code_table_correlation": r,
    }
