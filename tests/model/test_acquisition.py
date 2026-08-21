from unittest.mock import Mock
import pytest
import firedrake as fire
import numpy as np
from spyro.model.acquisition import DataSource, Receiver, RickerSource, Acquisition
from spyro.model.position import Position
from spyro.model.time_axis import TimeAxis


@pytest.fixture
def mesh():
    return fire.UnitSquareMesh(1, 1)


@pytest.fixture
def space(mesh):
    return fire.FunctionSpace(mesh, "CG", 1)


@pytest.fixture
def position():
    return Position(0.25, 0.5)


@pytest.fixture
def time_axis():
    return TimeAxis(0, 5, dt=1)


class TestReceiver:

    def test_sample(self, position, space):
        receiver = Receiver(
            p=position,
        )

        f = fire.Function(space).interpolate(fire.Constant(1.0))

        assert np.isclose(receiver.sample(f), 1.0)


class TestDataSource:

    def test_amplitude(self, position, time_axis):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        s = DataSource(p=position, data=data)

        assert s.amplitude(time_axis) == 1.0
        time_axis.update()
        assert s.amplitude(time_axis) == 2.0


class TestRickerSource:

    def test_amplitude(self, position, time_axis):
        s = RickerSource(p=position, delay=0.0, frequency=1.0)

        assert s.amplitude(time_axis) == 1.0
        time_axis.update()
        assert np.isclose(s.amplitude(time_axis), 0.0, atol=1e-2)


class TestAcquisition:
    def test_get_source_positions(self, position):
        source1 = Mock()
        source2 = Mock()

        source1.p = position
        source2.p = position

        acquisition = Acquisition(
            sources=[source1, source2],
            receivers=[],
        )

        positions = acquisition.get_source_positions()

        assert len(positions) == 2

    def test_get_source_amplitudes(self, time_axis):
        source1 = Mock()
        source2 = Mock()

        source1.amplitude.return_value = 1.0
        source2.amplitude.return_value = 2.0

        acquisition = Acquisition(
            sources=[source1, source2],
            receivers=[],
        )

        np.testing.assert_array_equal(
            acquisition.get_source_amplitudes(time_axis),
            np.array([1.0, 2.0])
        )

    def test_sample(self):
        f = Mock()
        receiver1 = Mock()
        receiver2 = Mock()

        receiver1.sample.return_value = 2.0
        receiver2.sample.return_value = 3.0

        acquisition = Acquisition(
            sources=[],
            receivers=[receiver1, receiver2]
        )

        np.testing.assert_array_equal(
            acquisition.sample(f),
            np.array([2.0, 3.0])
        )
