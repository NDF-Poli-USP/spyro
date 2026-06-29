# Possible pyadjoint CheckpointManager bug with Revolve WORK replay

## Summary

There appears to be a bug in `pyadjoint.checkpointing.CheckpointManager` when a
checkpoint schedule replays a timestep with
`Forward(..., write_adj_deps=True, storage=StorageType.WORK)` immediately before
the reverse sweep of that same timestep.

In this situation, the reverse sweep still needs the checkpointable state from
the start of the current timestep. The current upstream cleanup logic keeps the
checkpointable state required by the next timestep, but it may clear the
checkpointable state of the current timestep before `Reverse` evaluates the
adjoint.

This breaks tapes where a timestep reads a state variable and later overwrites
that same state variable in the same timestep. Spyro's central-difference wave
propagator has exactly this pattern:

1. solve using `u_n` and `u_nm1`;
2. assign `u_nm1 <- u_n`;
3. assign `u_n <- u_np1`;
4. evaluate receiver/functional terms.

The forward replay can still reproduce the functional value, but the adjoint
uses the wrong within-timestep state and the gradient is incorrect.

## Upstream location checked

Checked on 2026-06-29 against:

<https://raw.githubusercontent.com/dolfin-adjoint/pyadjoint/master/pyadjoint/checkpointing.py>

Relevant upstream logic in `CheckpointManager.process_operation(Forward)`:

```python
to_keep = set()
if step < (self.total_timesteps - 1):
    next_step = self.tape.timesteps[step + 1]
    # The checkpointable state set of the current step.
    to_keep = next_step.checkpointable_state
if functional:
    to_keep = to_keep.union([functional.block_variable])

for var in current_step.checkpointable_state - to_keep.union(self._global_deps):
    ...
    var.checkpoint = None
```

For `Forward(..., write_adj_deps=True, storage=StorageType.WORK)`, this can
discard `current_step.checkpointable_state` even though the following
`Reverse(...)` action needs those start-of-step values.

## Reproducer in Spyro

Worktree:

```bash
cd /Users/ddolci/dev_code/spyro/.wrokthrees/exact_checkpointing_auto_adjoint
source /Users/ddolci/dev_code/venv-firedrake/bin/activate
mpiexec -np 2 python3 tests/parallel/test_gradient_auto_adjoint.py
```

The test builds an automated-adjoint tape for the Spyro acoustic wave forward
solve and enables exact checkpointing with:

```python
Revolve(total_timesteps, steps_store=10)
```

Before the local fix, the checkpointed Taylor test failed:

```text
Computed convergence rates: [1.794151625..., 1.660884051..., 1.495484800...]
Automated-adjoint Taylor convergence rate: 1.495484800...
```

The same test without checkpointing passed:

```text
Computed convergence rates: [1.998815459..., 1.999518438..., 1.999787091...]
Automated-adjoint Taylor convergence rate: 1.998815459...
```

Replacing Spyro's adapted manager with pyadjoint's native
`pyadjoint.checkpointing.CheckpointManager` produced the same failing
checkpointed rate, which indicates that the issue is not introduced by Spyro's
manager wrapper.

## Additional diagnostics

The checkpointed forward replay reproduced the functional value at the base
control:

```text
checkpointing False rf_at_c_global 9.422035746651049e-06
checkpointing True  rf_at_c_global 9.422035746651049e-06
```

However, the checkpointed adjoint gradient was wrong:

```text
checkpointing False grad_norm 0.0006363388669809097
checkpointing True  grad_norm 6.504558192701857

checkpointing False directional -4.9635266502611905e-05
checkpointing True  directional -7.999662140454819e-05
```

So the failure is specifically in the reverse/adjoint use of replayed state, not
in recomputing the scalar functional value.

## Proposed fix

When the schedule asks for adjoint dependencies in `WORK` storage, keep the
current timestep checkpointable state until the following reverse sweep has
consumed it:

```python
to_keep = set()
if step < (self.total_timesteps - 1):
    next_step = self.tape.timesteps[step + 1]
    to_keep = next_step.checkpointable_state

if cp_action.write_adj_deps and cp_action.storage == StorageType.WORK:
    to_keep = to_keep.union(current_step.checkpointable_state)

if functional:
    to_keep = to_keep.union([functional.block_variable])
```

This keeps start-of-step values alive for the reverse sweep of that same step,
while preserving the existing cleanup behavior for other schedule actions.

## Validation after local fix

With the fix applied in Spyro's `SpyroCheckpointManager`:

```bash
source /Users/ddolci/dev_code/venv-firedrake/bin/activate
mpiexec -np 2 python3 tests/parallel/test_gradient_auto_adjoint.py
```

Result:

```text
Computed convergence rates: [1.998815459..., 1.999518438..., 1.999787091...]
Automated-adjoint Taylor convergence rate: 1.998815459...
```

Full pytest validation:

```bash
source /Users/ddolci/dev_code/venv-firedrake/bin/activate
pytest tests/parallel/test_gradient_auto_adjoint.py -m newer_firedrake
```

Result:

```text
4 passed, 2 warnings in 248.54s
```

Lint:

```bash
source /Users/ddolci/dev_code/venv-firedrake/bin/activate
python3 -m flake8 tests/parallel/test_gradient_auto_adjoint.py spyro/tools/checkpointing.py
```

Result: no output.

## Notes for an upstream issue

This is easiest to reproduce with a timestep tape where:

- a variable is part of `current_step.checkpointable_state`;
- that variable is read early in the timestep;
- that same variable is overwritten later in the timestep;
- a `Revolve` schedule emits `Forward(..., write_adj_deps=True, StorageType.WORK)`
  followed by `Reverse(...)` for that same step.

The bug is not specific to Spyro's `Revolve(nt, 10)` choice. That schedule only
exposes the problem because it forces recomputation and same-step WORK replay.
