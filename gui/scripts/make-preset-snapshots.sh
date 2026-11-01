#!/usr/bin/env bash
# Makes the preset snapshots of the Lab's preset guide: gui/PresetSnapshots/<preset>.jpg and manifest.json.
#
#   gui/scripts/make-preset-snapshots.sh [STAGE ...]        stages: runs speed streams render manifest (default: all)
#   gui/scripts/make-preset-snapshots.sh runs tiny dev     only the named presets (a preset name ends the stage list)
#
# Every snapshot is a frame of a real run of that exact plain preset: `alife run --preset P --seed 1 --steps N`, no overlay,
# no --set, the preset's own settings. The one difference: a checkpoint every 5 000 steps instead of the default 10 000, so that a run
# stopped early still has a frame (the manifest says so).
#   runs      the simulations, two at a time, into $WORK/<preset> (log: $WORK/logs). Skipped for a finished run.
#   speed     a solo fork of each finished run for a few hundred steps, nothing overridden: steps/s and peak memory at the
#             snapshot's population.
#   streams   `alife stream RUN --checkpoint latest --steps 0`: the frame as the Lab's viewer would receive it.
#   render    the app's own Metal renderers draw each stream offscreen (an opt-in test, ALIFE_PRESET_SNAPSHOTS=1; it is a
#             no-op in the normal `swift test`) into $WORK/render/<preset>.png; JPEG, ~640 x 400, into gui/PresetSnapshots.
#   manifest  gui/PresetSnapshots/manifest.json: preset, seed, step shown, population (whole world and in view), view,
#             steps/s, peak memory, engine git rev, whether the tree was dirty, date.
#
# Environment: WORK (default ~/alife-preset-snapshots), ALIFE (default <repo>/.venv/bin/alife), STEPS_<preset> to override a
# run length (for example STEPS_64k3d=10000), FORCE=1 to redo finished runs. Needs a Mac with Apple silicon and a Swift
# toolchain; the whole script takes about an hour on an M5 Pro. It writes only under $WORK and gui/PresetSnapshots. It removes
# only its own private TMPDIRs and stream files under $WORK, by exact name, when it ends; the run directories stay (rerunning is
# skipped for a finished one). Make the final snapshots from committed code: the manifest records `git_dirty`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$GUI_DIR/.." && pwd)"
WORK="${WORK:-$HOME/alife-preset-snapshots}"
ALIFE="${ALIFE:-$REPO/.venv/bin/alife}"
OUT_DIR="$GUI_DIR/PresetSnapshots"
SEED=1
ALL_PRESETS=(64k3s 64k3d mini3s mini3d 256k 64k dev tiny3s tiny3d tiny)  # the longest first
ALL_PRESETS_ORDER=(tiny dev 64k 256k tiny3d mini3d 64k3d tiny3s mini3s 64k3s)  # the order of the guide

# Steps of the snapshot run. The 256k world is 4 times the 64k one and runs at 17 steps/s, so it is shown after 10 000.
steps_for() {
  local override="STEPS_$1"
  if [[ -n "${!override:-}" ]]; then echo "${!override}"; return; fi
  case "$1" in
    256k) echo 10000 ;;
    *) echo 30000 ;;
  esac
}

# The MLX kernel cache of one process collides with another's when they share a TMPDIR (CLAUDE.md, F21): one each.
private_tmp() { local d="$WORK/tmp/$1"; mkdir -p "$d"; echo "$d"; }

run_one() {  # $1 = preset
  local p="$1" dir="$WORK/$1" n
  n="$(steps_for "$p")"
  mkdir -p "$WORK/logs"
  if [[ "${FORCE:-0}" != 1 && -f "$dir/meta.json" ]] && grep -q '"status": "done"' "$dir/meta.json"; then
    echo "run $p: finished already, kept"; return 0
  fi
  echo "run $p: $n steps"
  TMPDIR="$(private_tmp "$p")" "$ALIFE" run --preset "$p" --seed "$SEED" --steps "$n" --checkpoint-every 5000 \
    --out "$dir" --overwrite --quiet > "$WORK/logs/$p.log" 2>&1
  echo "run $p: done"
}

if [[ "${1:-}" == "--run-one" ]]; then run_one "$2"; exit 0; fi

STAGES=()
PRESETS=()
for a in "$@"; do
  case "$a" in
    runs|speed|streams|render|manifest) STAGES+=("$a") ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) PRESETS+=("$a") ;;
  esac
done
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(runs speed streams render manifest)
[[ ${#PRESETS[@]} -gt 0 ]] || PRESETS=("${ALL_PRESETS[@]}")
case "$WORK" in
  ""|/|"$HOME"|"$REPO"|"$GUI_DIR") echo "error: WORK=$WORK is not a work directory of its own" >&2; exit 2 ;;
esac
mkdir -p "$WORK/logs" "$WORK/render" "$OUT_DIR"
touch "$WORK/.make-preset-snapshots"

has_stage() { local s; for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

if has_stage runs; then
  # At most two simulations at once.
  printf '%s\n' "${PRESETS[@]}" | WORK="$WORK" ALIFE="$ALIFE" FORCE="${FORCE:-0}" xargs -P 2 -I{} "${BASH_SOURCE[0]}" --run-one {}
fi

# The steps/s and the peak memory of a preset at the snapshot's population: a solo fork of the finished run (`run --init-from RUN`,
# nothing overridden: the fork keeps the run's own cadences), so that the step is the one the picture shows, measured with nothing
# else on the GPU. The first metrics row holds the compile and is left out. The peak memory is the process's "peak memory
# footprint" from `/usr/bin/time -l` (GPU buffers included, compile included): approximate, and the manifest says so.
speed_steps() {
  case "$1" in
    256k) echo 600 ;;
    tiny|tiny3d|tiny3s) echo 2000 ;;
    *) echo 1000 ;;
  esac
}

if has_stage speed; then
  mkdir -p "$WORK/speed"
  for p in "${PRESETS[@]}"; do
    [[ -d "$WORK/$p" ]] || { echo "speed $p: no run in $WORK/$p" >&2; exit 1; }
    echo "speed $p: $(speed_steps "$p") steps from the end of the run, alone"
    TMPDIR="$(private_tmp "speed-$p")" /usr/bin/time -l -o "$WORK/speed/$p.time" "$ALIFE" run --init-from "$WORK/$p" \
      --steps "$(speed_steps "$p")" --out "$WORK/speed/$p" --overwrite --quiet > "$WORK/logs/speed-$p.log" 2>&1
  done
fi

if has_stage streams; then
  for p in "${PRESETS[@]}"; do
    echo "stream $p: the latest checkpoint"
    TMPDIR="$(private_tmp "stream-$p")" "$ALIFE" stream "$WORK/$p" --checkpoint latest --steps 0 --no-control > "$WORK/$p.stream.bin"
  done
fi

if has_stage render; then
  only="$(IFS=,; echo "${PRESETS[*]}")"
  ALIFE_PRESET_SNAPSHOTS=1 ALIFE_PRESET_SNAPSHOT_WORK="$WORK" ALIFE_PRESET_SNAPSHOT_OUT="$OUT_DIR" ALIFE_PRESET_SNAPSHOT_ONLY="$only" \
    swift test --package-path "$GUI_DIR" --filter PresetSnapshotTool
fi

if has_stage manifest; then
  machine="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  # Is the working tree (apart from the snapshots being made) different from HEAD? The final snapshots are made from committed code.
  dirty=false
  [[ -z "$(git -C "$REPO" status --porcelain -- . ':!gui/PresetSnapshots' 2>/dev/null)" ]] || dirty=true
  python3 - "$WORK" "$OUT_DIR/manifest.json" "$machine" "$SEED" "$REPO" "$dirty" "${ALL_PRESETS_ORDER[@]}" <<'PY'
import json, re, subprocess, sys
work, out, machine, seed, repo, dirty = sys.argv[1:7]
order = sys.argv[7:]
head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
snapshots = []
for p in order:
    info = json.load(open(f"{work}/render/{p}.json"))
    meta = json.load(open(f"{work}/{p}/meta.json"))
    rows = [json.loads(l) for l in open(f"{work}/speed/{p}/metrics.jsonl")][1:]  # the first row holds the compile
    steps = sum(r["steps"] for r in rows)
    wall = sum(r["wall_s"] for r in rows)
    peak = int(re.search(r"(\d+)\s+peak memory footprint", open(f"{work}/speed/{p}.time").read()).group(1))
    # Did the engine (src, pyproject, uv.lock) change between the run's rev and HEAD? Then the run is not HEAD's.
    changed = subprocess.run(["git", "-C", repo, "diff", "--quiet", meta["git_rev"], "HEAD", "--", "src", "pyproject.toml", "uv.lock"]).returncode != 0
    snapshots.append({
        "preset": p, "seed": int(seed), "step": info["step"], "population": info["alive"],
        "window_population": info.get("window_alive"), "view": info.get("view", "the whole world"),
        "steps_per_s": round(steps / wall, 1), "speed_measured_over_steps": steps,
        "peak_memory_bytes": peak,
        "git_rev": meta["git_rev"], "engine_changed_since_run": changed, "date": meta["created_utc"][:10],
    })
doc = {
    "machine": machine or "unknown",
    "git_dirty": dirty == "true",
    "drawn_with_rev": head,
    "notes": "steps_per_s: the run's own fork, alone on the GPU, at this population (first metrics row, with the compile, left out). "
             "peak_memory_bytes: peak memory footprint of that fork from /usr/bin/time -l, approximate (GPU buffers and the compile included). "
             "The runs wrote a checkpoint every 5000 steps instead of the default 10000; nothing else is changed from the preset.",
    "snapshots": snapshots,
}
json.dump(doc, open(out, "w"), indent=1)
open(out, "a").write("\n")
print(f"wrote {out}")
PY
fi

# Housekeeping: the private MLX caches are of no use afterwards, and the stream files are 0.02 to 6 MB each. Only paths this script made
# are removed, each by its exact name, and only when $WORK carries the marker this script put there.
if [[ -f "$WORK/.make-preset-snapshots" ]]; then
  for p in "${PRESETS[@]}"; do
    for d in "$WORK/tmp/$p" "$WORK/tmp/speed-$p" "$WORK/tmp/stream-$p"; do [[ -d "$d" ]] && rm -rf "$d"; done
    if has_stage render || has_stage manifest; then [[ -f "$WORK/$p.stream.bin" ]] && rm -f "$WORK/$p.stream.bin"; fi
  done
  rmdir "$WORK/tmp" 2>/dev/null || true
fi
