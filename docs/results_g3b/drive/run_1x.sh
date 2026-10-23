#!/bin/zsh
# 1x arms: where a long save is owed in slots (save_stall_steps) and could make steps late
F=/private/tmp/claude-501/-Users-jeromeverony-artificial-life--claude-worktrees-dazzling-noether-500875/79bca148-0e17-4727-a473-3cea7b988e81/scratchpad/g3b/finish/drive
S=${F:h:h:h}
DRIVE="/Volumes/M5 PRO external/alife/coordinator_scratch_2026-10-21/g3b_ckpt_stall_1"
export TMPDIR=$F/tmp PYTHONDONTWRITEBYTECODE=1
cd $S/g3b-wt
for p in "internal 1x1" "drive 1x1"; do
  set -- ${=p}; disk=$1; n=$2
  if [[ $disk == drive ]]; then ROOT=$DRIVE; else ROOT=$F/internal; fi
  { echo "== $disk $n start $(date +%T)"; df -k "$ROOT" | tail -1; } >> $F/logs/space_marks.txt
  $S/g3b-wt/.venv/bin/python $F/bench_ckpt.py $F/logs/ckpt_${disk}_$n.jsonl -- bench --play --preset 64k3d --overlay storm --set director.enabled=true \
    --play-resume $ROOT/planet:472000 --steps 2500 --play-warmup 300 --play-rates 1 --play-directors rules \
    --play-out $ROOT/bench_$n --json $F/logs/bench_${disk}_$n.json > $F/logs/bench_${disk}_$n.log 2>&1
  { echo "== $disk $n end rc=$? $(date +%T)"; df -k "$ROOT" | tail -1; du -sk $ROOT/bench_$n; } >> $F/logs/space_marks.txt
done
echo ALLDONE1X >> $F/logs/space_marks.txt
