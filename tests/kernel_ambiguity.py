"""Which MLX CPU kernels are safe to share between processes (DESIGN F21 item 6, §22.0).

Ported from ``tests/jit_cache.py`` of the branch ``fix/mlx-compile-collision`` (the guard against the
compile-cache constant collision; its author's text and tests are the source of everything below), so that
``tests/worker_cache.py`` can use the filter.

MLX 0.32.3 names a fused kernel after its op tape and ``std::hash`` of its scalar constants printed back to
back with no separator, and its CPU backend keeps every kernel on disk (``$TMPDIR/mlx/<version>/cpu``) and
loads it by name in any later process. Two constant lists whose prints concatenate alike, ``(12.0, 1)`` and
``(1.0, 21)`` ("121"), therefore share one kernel: whichever was compiled first on the machine, or restored
from CI's cache, is what the other runs. Only an *ambiguous* constant string, one that another list of the
same dtypes also prints, can be hit that way; any other kernel name pins its constants.

``ambiguous(source)`` tells them apart from the kernel's generated C++ source. ``worker_cache`` never seeds an
ambiguous kernel from the shared cache into a test process and never hands one back, so an ambiguous kernel is
always compiled by the process tree that uses it, from its own constants. Two configurations that collide
inside one process are not covered (MLX keeps no Python handle on its in-memory kernel cache); DESIGN §15.4 of
the guard branch keeps engine processes to one configuration.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache

import numpy as np

# One constant declaration of MLX's generated CPU source: ``  int32_t tmp_D = 21;`` (print_constant's text).
_DECL = re.compile(r"^\s*([\w:]+) tmp_\w+ = (-?INFINITY|NAN|true|false|[-+0-9.eE]+);\s*$")
_KERNEL = re.compile(r"\bvoid (\w+)\(int\* shape, int64_t\*\* strides, void\*\* args\)")  # after any preamble
_LAYOUT = re.compile(r"_(contiguous|strided_\w+)$")
_INT = re.compile(r"-?(0|[1-9][0-9]*)")
_UINT = re.compile(r"0|[1-9][0-9]*")
_NUMBER = re.compile(r"-?([0-9]+\.?[0-9]*|\.[0-9]+)(e[-+][0-9]+)?")
_INT_RANGE = {
    "int8_t": (-(2**7), 2**7 - 1),
    "int16_t": (-(2**15), 2**15 - 1),
    "int32_t": (-(2**31), 2**31 - 1),
    "int64_t": (-(2**63), 2**63 - 1),
    "uint8_t": (0, 2**8 - 1),
    "uint16_t": (0, 2**16 - 1),
    "uint32_t": (0, 2**32 - 1),
    "uint64_t": (0, 2**64 - 1),
}
_FLOATS = {"float": np.float32, "float16_t": np.float16, "double": np.float64, "bfloat16_t": None}
_DIGITS = {"float": 9, "float16_t": 9, "double": 17}  # print_float_constant: max_digits10 of float or double
_CONSERVATIVE = object()  # a dtype whose printed form this module does not model: any token may be one


def _printable(ctype: str, tok: str) -> bool | object:
    """Is ``tok`` exactly what MLX's ``print_constant`` writes for some value of C type ``ctype``?"""
    if ctype == "bool":
        return tok in ("true", "false")
    if ctype in _INT_RANGE:
        lo, hi = _INT_RANGE[ctype]
        return bool((_UINT if lo == 0 else _INT).fullmatch(tok)) and tok != "-0" and lo <= int(tok) <= hi
    if ctype not in _FLOATS:
        return _CONSERVATIVE
    if tok in ("NAN", "INFINITY", "-INFINITY"):
        return True
    if not _NUMBER.fullmatch(tok):
        return False
    np_type = _FLOATS[ctype]
    if np_type is None:  # bfloat16: no numpy type to round through
        return _CONSERVATIVE
    with np.errstate(over="ignore"):
        v = np_type(float(tok))
    return bool(np.isfinite(v)) and f"{float(v):.{_DIGITS[ctype]}g}" == tok  # C++ ostream default format


def _parses(ctypes: tuple[str, ...], text: str) -> int:
    """How many ways (capped at 2) ``text`` splits into printed constants of ``ctypes``, in order."""

    @cache
    def count(i: int, j: int) -> int:
        if j == len(ctypes):
            return int(i == len(text))
        total = 0
        for end in range(i + 1, len(text) + 1):
            ok = _printable(ctypes[j], text[i:end])
            if ok is _CONSERVATIVE:
                return 2
            if ok:
                total += count(end, j + 1)
                if total >= 2:
                    return 2
        return total

    return count(0, 0)


def constants_ambiguous(decls: Sequence[tuple[str, str]]) -> bool:
    """Do the printed constants ``[(C type, text), ...]`` concatenate to a string that another list of the
    same types also prints (``[("int32_t", "12"), ("int32_t", "1")]``: "121", as ``(1, 21)`` does)?"""
    if len(decls) < 2:
        return False
    if any(_printable(c, t) is not True for c, t in decls):
        return True
    return _parses(tuple(c for c, _ in decls), "".join(t for _, t in decls)) != 1


def ambiguous(source: str) -> bool:
    """True unless the kernel's name provably pins its constants: fewer than two constants, or a constant
    string that no other list of the same dtypes prints. Anything this cannot read counts as ambiguous."""
    m = _KERNEL.search(source)
    if m is None:
        return True
    name = _LAYOUT.sub("", m.group(1))
    parts = name.rsplit("_", 3)  # tape, input kinds (C/S/V), non-constant input dtypes, constant-string hash
    if len(parts) != 4 or not set(parts[1]) <= set("CSV") or not parts[3].isdigit():
        return True
    body = source[m.end() :]
    decls = [d.groups() for line in body.splitlines() if (d := _DECL.match(line))]
    if len(decls) != parts[1].count("C"):
        return True
    return constants_ambiguous(decls)


def declarations(source: str) -> list[tuple[str, str]] | None:
    """The ``(C type, printed text)`` of each constant of a kernel's generated source, in order, or ``None``
    when the source has no kernel function (not a MLX kernel)."""
    m = _KERNEL.search(source)
    if m is None:
        return None
    return [d.groups() for line in source[m.end() :].splitlines() if (d := _DECL.match(line))]
