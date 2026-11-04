#!/bin/zsh
# Repeats the full Swift test run and, when a run lasts STALL_SECONDS longer than the normal ~175 s, records what the test process was
# doing: two `sample`s 10 s apart, the power assertions, the thread count and the load (DESIGN section 22.0, Review-log row I91).
#
#   gui/scripts/stall-hunt.sh [RUNS] [OUT_DIR]      RUNS defaults to 12, OUT_DIR to a fresh directory under $TMPDIR
#
# Env: STALL_SECONDS (default 205, the run time after which a run counts as stalled), STOP_ON_STALL=1 (default: stop after the first
# stalled run). It starts nothing but `swift test` and `sample`, only samples descendants of the `swift test` it started, and
# never writes outside OUT_DIR. It opens no window of its own; the tests make theirs off screen (QuietWindow).
set -u
# zsh lowers the priority of background jobs (BG_NICE); a niced test process is what stalled (run of 2026-11-04).
setopt NO_BG_NICE
RUNS=${1:-12}
OUT=${2:-$(mktemp -d "${TMPDIR:-/tmp}/stall-hunt.XXXXXX")}
THR=${STALL_SECONDS:-260}
STOP=${STOP_ON_STALL:-1}
GUI=${0:A:h:h}
mkdir -p "$OUT"
cd "$GUI" || exit 1

descendants() { local p; for p in $(pgrep -P "$1"); do echo "$p"; descendants "$p"; done }

for i in $(seq 1 "$RUNS"); do
  R=$OUT/run$i; mkdir -p "$R"
  start=$(date +%s)
  swift test > "$R/test.log" 2>&1 &
  PID=$!
  stalled=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 5
    el=$(( $(date +%s) - start ))
    if (( el > THR && stalled == 0 )); then
      stalled=1
      T=$(for d in $(descendants "$PID"); do
            case $(ps -o command= -p "$d" | cut -c1-300) in *swiftpm-testing-helper*|*xctest*) echo "$d";; esac
          done | head -1)
      { echo "run $i past ${THR} s (at $el s), test process $T"; date; pmset -g assertions; uptime
        echo "threads: $(ps -M -p "$T" | wc -l)"; ps -o pid,%cpu,state,etime,rss -p "$T"
        top -l 1 -n 10 -o cpu | head -20; } > "$R/watch.txt" 2>&1
      [[ -n "$T" ]] && sample "$T" 10 -file "$R/sample1.txt" > /dev/null 2>&1
      sleep 10
      { date; pmset -g assertions | head -30; uptime; echo "threads: $(ps -M -p "$T" | wc -l)"; } >> "$R/watch.txt" 2>&1
      [[ -n "$T" ]] && sample "$T" 10 -file "$R/sample2.txt" > /dev/null 2>&1
    fi
  done
  wait "$PID"; rc=$?
  el=$(( $(date +%s) - start ))
  echo "run$i rc=$rc elapsed=${el}s stalled=$stalled $(grep -E 'Test run with' "$R/test.log" | tail -1 | cut -c1-90)" | tee -a "$OUT/summary.txt"
  if (( stalled == 1 && STOP == 1 )); then break; fi
done
