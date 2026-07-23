"""Unit tests for Spyro checkpointing helpers."""

import pytest

from checkpoint_schedules import PeriodicDiskRevolve

from spyro.tools.checkpointing import (
    CheckpointError,
    DecimatedRecompute,
    ExactRecompute,
    Mode,
    RecomputeStrategy,
    SpyroCheckpointManager,
)


class FakeVariable:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint


class FakeBlock:
    def __init__(self, dependencies=(), outputs=()):
        self._dependencies = list(dependencies)
        self._outputs = list(outputs)
        self.recompute_count = 0

    def get_dependencies(self):
        return self._dependencies

    def get_outputs(self):
        return self._outputs

    def recompute(self):
        self.recompute_count += 1
        for output in self._outputs:
            output.checkpoint = f"recomputed-{self.recompute_count}"


class FakeStep(list):
    def __init__(self, blocks=(), checkpointable_state=(), adjoint_dependencies=()):
        super().__init__(blocks)
        self.checkpointable_state = set(checkpointable_state)
        self.adjoint_dependencies = set(adjoint_dependencies)


class FakeManager:
    def __init__(self, mode):
        self.mode = mode


@pytest.mark.newer_firedrake
def test_exact_recompute_runs_every_block():
    output = FakeVariable()
    block = FakeBlock(outputs=[output])
    step = FakeStep(blocks=[block])

    ExactRecompute().Recompute_step(
        FakeManager(Mode.EVALUATED),
        step=3,
        current_step=step,
        cp_action=None,
    )

    assert block.recompute_count == 1
    assert output.checkpoint == "recomputed-1"


@pytest.mark.newer_firedrake
def test_decimated_recompute_exact_functional_recomputes_in_recompute_mode():
    dependency = FakeVariable("dependency-state")
    output = FakeVariable()
    block = FakeBlock(dependencies=[dependency], outputs=[output])
    step = FakeStep(
        blocks=[block],
        checkpointable_state=[dependency],
        adjoint_dependencies=[dependency],
    )
    strategy = DecimatedRecompute(period=10, exact_functional=True)

    strategy.Recompute_step(
        FakeManager(Mode.RECOMPUTE),
        step=1,
        current_step=step,
        cp_action=None,
    )

    assert block.recompute_count == 1
    assert strategy._checkpoints[dependency] == "dependency-state"
    assert strategy._checkpoints[output] == "recomputed-1"


@pytest.mark.newer_firedrake
def test_decimated_recompute_periodic_step_refreshes_saved_values():
    state = FakeVariable("old-state")
    output = FakeVariable()
    block = FakeBlock(outputs=[output])
    step = FakeStep(blocks=[block], checkpointable_state=[state])
    strategy = DecimatedRecompute(period=2, exact_functional=False)

    strategy.Recompute_step(
        FakeManager(Mode.EVALUATED),
        step=4,
        current_step=step,
        cp_action=None,
    )

    assert block.recompute_count == 1
    assert strategy._checkpoints[state] == "old-state"
    assert strategy._checkpoints[output] == "recomputed-1"


@pytest.mark.newer_firedrake
def test_decimated_recompute_restores_saved_outputs_between_periods():
    state = FakeVariable("state-0")
    dependency = FakeVariable("dependency-0")
    output = FakeVariable("output-0")
    block = FakeBlock(dependencies=[dependency], outputs=[output])
    step = FakeStep(
        blocks=[block],
        checkpointable_state=[state],
        adjoint_dependencies=[dependency],
    )
    strategy = DecimatedRecompute(period=3, exact_functional=False)

    strategy.Recompute_step(
        FakeManager(Mode.EVALUATED),
        step=0,
        current_step=step,
        cp_action=None,
    )
    state.checkpoint = None
    dependency.checkpoint = None
    output.checkpoint = None

    strategy.Recompute_step(
        FakeManager(Mode.EVALUATED),
        step=1,
        current_step=step,
        cp_action=None,
    )

    assert block.recompute_count == 1
    assert state.checkpoint == "state-0"
    assert dependency.checkpoint == "dependency-0"
    assert output.checkpoint == "recomputed-1"


@pytest.mark.newer_firedrake
def test_decimated_recompute_falls_back_when_output_was_not_saved():
    output = FakeVariable()
    block = FakeBlock(outputs=[output])
    step = FakeStep(blocks=[block])
    strategy = DecimatedRecompute(period=3, exact_functional=False)

    strategy.Recompute_step(
        FakeManager(Mode.EVALUATED),
        step=1,
        current_step=step,
        cp_action=None,
    )

    assert block.recompute_count == 1
    assert output.checkpoint == "recomputed-1"
    assert strategy._checkpoints[output] == "recomputed-1"


@pytest.mark.newer_firedrake
def test_disk_storage_detection_falls_back_for_periodic_disk_revolve_bug():
    disk_schedule = PeriodicDiskRevolve(12, 3)
    ram_schedule = PeriodicDiskRevolve(12, 3, wd=1000, rd=1000)

    assert SpyroCheckpointManager._uses_disk_storage(disk_schedule) is True
    assert SpyroCheckpointManager._uses_disk_storage(ram_schedule) is False


@pytest.mark.newer_firedrake
def test_disk_storage_detection_uses_schedule_contract_when_available():
    class FakeSchedule:
        def uses_storage_type(self, storage_type):
            return True

    assert SpyroCheckpointManager._uses_disk_storage(FakeSchedule()) is True


@pytest.mark.newer_firedrake
def test_recompute_strategy_base_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        RecomputeStrategy().Recompute_step(
            FakeManager(Mode.EVALUATED),
            step=0,
            current_step=FakeStep(),
            cp_action=None,
        )


@pytest.mark.newer_firedrake
def test_checkpoint_error_is_runtime_error():
    assert issubclass(CheckpointError, RuntimeError)
