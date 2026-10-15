#!/bin/zsh
# Build the spyro trees compared by the sweep, under ./trees:
#   baseline - the original u^{n+1} solve (main at the base commit);
#   inplace  - the acceleration solved into the u^{n+1} buffer (a variant
#              measured only, not proposed).
# The reformulated solver is the working tree itself.
set -e
DIR=${0:A:h}
REPO=$(git -C $DIR rev-parse --show-toplevel)
BASE=${1:-d5458bef}
rm -rf $DIR/trees && mkdir -p $DIR/trees/baseline $DIR/trees/inplace
git -C $REPO archive $BASE spyro | tar -x -C $DIR/trees/baseline
cp -R $REPO/spyro $DIR/trees/inplace/
python3 - $DIR/trees/inplace/spyro/solvers/acoustic_solver_construction_no_pml.py <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("wave.acceleration, bcs=bcs, constant_jacobian=True)",
              "wave.u_np1, bcs=bcs, constant_jacobian=True)")
s = s.replace("acceleration_solver, wave.u_np1, wave.u_n, wave.u_nm1,\n        wave.acceleration, wave.dt,",
              "acceleration_solver, wave.u_np1, wave.u_n, wave.u_nm1,\n        wave.u_np1, wave.dt,")
open(p, "w").write(s)
PY
grep -q "wave.u_np1, bcs=bcs" $DIR/trees/inplace/spyro/solvers/acoustic_solver_construction_no_pml.py
echo "trees ready in $DIR/trees"
