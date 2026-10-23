#!/bin/zsh
F=/private/tmp/claude-501/-Users-jeromeverony-artificial-life--claude-worktrees-dazzling-noether-500875/79bca148-0e17-4727-a473-3cea7b988e81/scratchpad/g3b/finish/drive
for p in "internal 1" "drive 1" "internal 2" "drive 2"; do $F/run_pair.sh ${=p}; done
echo ALLDONE >> $F/logs/space_marks.txt
