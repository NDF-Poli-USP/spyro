#!/bin/zsh
# usage: sweep.sh probe|timing [jobs] [extra bench.py args...]
#   probe  - peak checkpoint bytes (walks the tape after every schedule
#            action; deterministic, can run in parallel, timings distorted)
#   timing - wall-clock and peak RSS; run with jobs=1 on an idle machine
# Needs ./setup_trees.sh first and the Firedrake venv active.
DIR=${0:A:h}
REPO=$(git -C $DIR rev-parse --show-toplevel)
MODE=$1; JOBS=${2:-1}; shift 2 2>/dev/null; EXTRA="$*"
export OMP_NUM_THREADS=1
cd $DIR
mkdir -p results/sweep
typeset -A TREE
TREE=(orig $DIR/trees/baseline new $REPO inplace $DIR/trees/inplace)
cmds=()
for v in orig new inplace; do
  cmds+=("PYTHONPATH=${TREE[$v]} python bench.py --schedule single --label $v $EXTRA --out results/sweep/${MODE}_${v}_single_0.npz")
  for sch in mixed revolve; do
    for s in 4 8 16 32 64; do
      flag=""; [[ $MODE == probe ]] && flag="--probe-bytes"
      cmds+=("PYTHONPATH=${TREE[$v]} python bench.py --schedule $sch --snapshots $s $flag --label $v $EXTRA --out results/sweep/${MODE}_${v}_${sch}_${s}.npz")
    done
  done
done
printf '%s\0' "${cmds[@]}" | xargs -0 -P $JOBS -n 1 zsh -c 'eval "$0" > /dev/null 2>&1 || echo FAILED: "$0"'
echo done
