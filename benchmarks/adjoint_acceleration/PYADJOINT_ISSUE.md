# pyadjoint: wrong first adjoint after replaying a checkpointed tape

Draft of an upstream issue for [dolfin-adjoint/pyadjoint](https://github.com/dolfin-adjoint/pyadjoint), plus the plan to fix it there and to drop the spyro workaround afterwards. The text from "Summary" to "Proposed fix" is meant to be posted as the issue.

Files next to this one:

- `repro_replay_before_first_adjoint.py`: Firedrake-only reproducer (no spyro).
- `pyadjoint_replay_fix.diff`: proposed patch to `pyadjoint/checkpointing.py`.

---

## Summary

With a checkpoint schedule that recomputes, such as `MixedCheckpointSchedule` or `Revolve`, the **first** adjoint of a tape is wrong if the tape was replayed before it. Replaying means calling the `ReducedFunctional`, and the result is wrong even when the replay is at the recorded control value. The functional value is correct. Once one adjoint has run, later replays and adjoints are exact.

Every optimizer evaluates the functional at its starting point before it asks for the first gradient. So every optimization over a checkpointed tape starts from a wrong gradient. The Taylor test is affected the same way, because `taylor_test` calls `rf(m)` before `rf.derivative()`.

## Environment

- Firedrake 2026.10.0, pyadjoint-ad 2026.10.0, checkpoint_schedules 1.0.4, petsc4py 3.26.0 (`firedrakeproject/firedrake:latest` Docker image).
- `pyadjoint/checkpointing.py` in that release is identical to `master` as of 2026-10-16.

## Reproducer

`repro_replay_before_first_adjoint.py` tapes a central-difference wave equation, $M(c)(u^{n+1}-2u^n+u^{n-1})/\Delta t^2 + K u^n = f^n$, solved for $u^{n+1}$ with a `LinearVariationalSolver`. It uses 40 steps and closes each time step right after the solve with `tape.end_timestep()`. It then compares gradients with those of a tape without checkpointing. Output with `MixedCheckpointSchedule(40, 4)`:

| sequence after recording at `c0` | relative gradient error |
|---|---|
| `derivative()` | 0 |
| `rf(c0)`, `derivative()` | 3.4e-3 |
| `rf(c1)`, `derivative()` | 2.5e-3 |
| `derivative()`, `rf(c1)`, `derivative()` | 4.9e-15 |

The functional value of every replay is exact.

| schedule | `rf(m)` before the first `derivative()` |
|---|---|
| no checkpointing | exact |
| `SingleMemoryStorageSchedule` | exact |
| `MixedCheckpointSchedule` | wrong (2.5e-3 to 3.4e-3) |
| `Revolve` | wrong (2.5e-3) |

In an application (spyro FWI), a TAO inversion with `MixedCheckpointSchedule` drifts from the same inversion without checkpointing from its first step. With the configuration used, the functional history differed by 1.8e-5 after 4 iterations.

## What goes wrong

Instrumenting the reverse sweep shows that **only the last time step** is wrong. The adjoint of its solve reads $u^n$ and $u^{n-1}$ with wrong values. Every other step reads the same values as with a tape without checkpointing.

`CheckpointManager.recompute` replays the forward schedule. In `process_operation` for `Forward` (`checkpointing.py`, lines 319–411 on `master`), each step frees what the next step does not need:

```python
            to_keep = set()
            if step < (self.total_timesteps - 1):
                next_step = self.tape.timesteps[step + 1]
                # The checkpointable state set of the current step.
                to_keep = next_step.checkpointable_state
            ...
            for var in current_step.checkpointable_state - to_keep.union(self._global_deps):
                ...
                # Clear the checkpoint for other cases
                var.checkpoint = None
```

- **The last step has no next step**, so `to_keep` is empty and its whole checkpointable state is cleared.
- With these schedules the last step runs as `Forward(..., write_adj_deps=True, storage=StorageType.WORK)`, and the `Reverse` of the same step follows immediately and reads that state from working memory. That `Reverse` then finds the checkpoints it needs set to `None`.
- `BlockVariable.saved_output` then silently falls back to the live object:

  ```python
  @property
  def saved_output(self):
      if self.checkpoint is not None:
          return self.output._ad_restore_at_checkpoint(self.checkpoint)
      else:
          return self.output
  ```

- The live `Function`s no longer hold the values at the time of the solve: the state rotation (`u_nm1.assign(u_n)`, `u_n.assign(u_np1)`) ran after it. The adjoint of the last solve is therefore linearized about the wrong state. This also happens when the replay is at the recorded control.

Observed, not fully traced:
- The first adjoint taken right after recording is exact, because the recording itself does not go through this clearing.
- After the first reverse sweep (`_revised_adj_deps`), replays are exact. I did not trace why the revised path keeps the needed values.
- `SingleMemoryStorageSchedule` has its own branch in the same loop that only clears variables that are not adjoint dependencies, so it is not affected.

## Proposed fix

When the step runs in working memory and writes adjoint dependencies, its own `Reverse` is next and needs its state. Keep it, by adding it to `to_keep` where `to_keep` is computed (`pyadjoint_replay_fix.diff`):

```diff
@@ -349,6 +349,10 @@
                 next_step = self.tape.timesteps[step + 1]
                 # The checkpointable state set of the current step.
                 to_keep = next_step.checkpointable_state
+            if cp_action.write_adj_deps and cp_action.storage == StorageType.WORK:
+                # The Reverse of this step follows and reads its state from
+                # working memory.
+                to_keep = to_keep.union(current_step.checkpointable_state)
             if functional:
                 to_keep = to_keep.union([functional.block_variable])
```

Validation with the patch applied to pyadjoint-ad 2026.10.0:

- **Reproducer:** all four sequences exact (largest error 4.9e-15).
- **spyro (unmodified `main`):** a replay before the first gradient matches a fresh recording (< 1e-10). TAO inversions with `MixedCheckpointSchedule` follow the run without checkpointing (< 1e-10). The checkpointing, NRBC and FWI tests pass: 16 tests.
- **Peak checkpoint memory during the reverse sweep** (2D wave, 2481 dofs, 801 steps, in fields of 19 848 B): a constant +2 fields, the state of one step held until its reverse. Recomputed steps and gradients unchanged.

  | schedule | before | with the fix |
  |---|---|---|
  | Mixed, 16 snapshots | 156.0 | 158.0 |
  | Mixed, 4 snapshots | 43.8 | 45.8 |
  | Revolve, 16 snapshots | 40.4 | 42.4 |

Alternatives considered:

- **Special-casing the last step** (`step == self.total_timesteps - 1`): fixes the observed case, but encodes a consequence ("no next step") rather than the cause ("this step's reverse comes next"). It would break again for any schedule that reverses another step straight from working memory.
- **Checking `_revised_adj_deps` / `adjoint_dependencies` inside the clearing loop:** also works, but it adds a third special case to that loop, while the fix above corrects `to_keep` itself.

Separately, it would help if `saved_output` raised, instead of returning the live object, when a checkpoint that a reverse sweep needs has been cleared. This bug would then have surfaced as an exception instead of a 0.3% gradient error. That is a larger behavioural change and is better discussed on its own.

---

## Plan

### In pyadjoint

1. Open the issue with the sections above and attach `repro_replay_before_first_adjoint.py`. Per spyro's AGENTS.md, a person files it and declares that the diagnosis was AI-assisted (Claude Code).
2. Open a PR with `pyadjoint_replay_fix.diff` and a regression test. The test turns the reproducer into a test that, for `MixedCheckpointSchedule` and `Revolve`, asserts that the gradient after `rf(m)` matches the gradient of a tape without checkpointing:
   - at the recorded control;
   - at a different control.
   It should go next to pyadjoint's existing checkpointing tests (to be located; pyadjoint has Firedrake-based checkpointing tests).
3. Points to raise with the maintainers:
   - whether any schedule runs `Forward(write_adj_deps=True, storage=WORK)` over more than one step before its `Reverse`; the fix then keeps the state of every step of that action, not just one, so the memory cost would grow;
   - why the path after `_revised_adj_deps` already works;
   - the optional `saved_output` change above.

### In spyro

1. Keep [NDF-Poli-USP/spyro#435](https://github.com/NDF-Poli-USP/spyro/pull/435) (`AutomatedAdjoint.ensure_first_adjoint`) until a pyadjoint release contains the fix. It costs one extra adjoint sweep per recording, and only with `snapshots`.
2. When that release is out:
   - require it in spyro's version check;
   - remove `ensure_first_adjoint` and its calls;
   - keep `tests/on_one_core/test_automated_adjoint_replay.py`, which then tests pyadjoint's behaviour directly.
3. Link the upstream issue from #435 once it exists.

### Unrelated incompatibility found on the way

In the same Docker image, `pyadjoint.optimization.tao_solver.TAOSolver._tao_reasons` uses `PETSc.TAO.Reason`, which petsc4py 3.26 renamed `PETSc.TAO.ConvergedReason`. Any TAO run that stops without converging (for example on `tao_max_it`) raises `AttributeError` instead of `TAOConvergenceError`. This makes spyro's `test_fwi_automated_adjoint` fail in that image on `main` too. It is worth a separate pyadjoint issue.
