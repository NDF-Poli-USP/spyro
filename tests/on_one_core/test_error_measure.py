"""Unit tests for the error measures implemented in spyro.tools.error_measure.

These tests cover the MeasureError class and its methods, ensuring
that error calculations and file operations work as expected.
"""

from pytest import fixture, mark, raises, warns
from numpy import arange, array, inf, newaxis, pi, sin, tile, zeros_like
from numpy.random import randn
from numpy.testing import assert_array_equal
from os import makedirs, path
from unittest.mock import patch
from spyro.tools.error_measure import MeasureError


class TestMeasureError:
    """Test suite for MeasureError class."""

    @fixture
    def measure_error(self):
        """Create a MeasureError instance for testing."""
        output_folder = "/output/test_output"
        output_case = "test_case"
        measure_error = MeasureError()
        measure_error.initialize_paths_for_error(output_folder=output_folder,
                                                 output_case=output_case)
        return measure_error

    @fixture
    def sample_signals(self):
        """Create sample signals for testing."""
        dt = 0.01
        t = arange(0, 1, dt)
        # Create a simple sine wave as reference
        reference = sin(2 * pi * 5 * t)
        # Create a slightly different signal as model
        model = sin(2 * pi * 5 * t + 0.1)
        return model, reference, dt

    @fixture
    def receiver_data(self):
        """Create sample receiver data for testing."""
        n_time = 100
        n_receivers = 3
        receivers = randn(n_time, n_receivers)
        return receivers

    def test_initialization_default(self):
        """Test default initialization."""
        error = MeasureError()
        error.initialize_paths_for_error()
        assert error.path_save_error.endswith("/output/")
        assert error.path_save_err_case == error.path_save_error
        assert error.path_reference.endswith("/output/preamble/")
        assert error.comm is None

    def test_initialization_custom_paths(self):
        """Test initialization with custom paths."""
        output_folder = "/custom/path"
        output_case = "case123"
        error = MeasureError()
        error.initialize_paths_for_error(output_folder=output_folder,
                                         output_case=output_case)
        assert error.path_save_error == "/custom/path/"
        assert error.path_save_err_case == "/custom/path/case123/"
        assert error.path_reference == "/custom/path/preamble/"

    def test_check_signal_lengths_equal(self):
        """Test when signals have equal lengths."""
        signal1 = array([1, 2, 3, 4])
        signal2 = array([5, 6, 7, 8])
        result1, result2 = MeasureError.check_signal_lengths(signal1, signal2)
        assert len(result1) == len(result2) == 4
        assert_array_equal(result1, signal1)
        assert_array_equal(result2, signal2)

    def test_check_signal_lengths_different(self):
        """Test when signals have different lengths."""
        signal1 = array([1, 2, 3])
        signal2 = array([4, 5, 6, 7, 8])
        result1, result2 = MeasureError.check_signal_lengths(signal1, signal2)
        assert len(result1) == len(result2) == 5
        assert_array_equal(result2, signal2)
        assert_array_equal(result1[:3], signal1)
        assert result1[3] == 0 and result1[4] == 0

        result2, result1 = MeasureError.check_signal_lengths(signal2, signal1)
        assert len(result1) == len(result2) == 5
        assert_array_equal(result2, signal2)
        assert_array_equal(result1[:3], signal1)
        assert result1[3] == 0 and result1[4] == 0

    def test_peak_error_identical_signals(self):
        """Test peak error with identical signals."""
        signal = array([0, 1, 2, 3, 2, 1, 0])
        peak_error, peak_reference = MeasureError.peak_error(signal, signal)
        assert peak_error == 0.0
        assert peak_reference == 3.0

    def test_peak_error_different_signals(self):
        """Test peak error with different signals."""
        signal1 = array([0, 1, 2, 3, 2, 1, 0])
        signal2 = array([0, 0.5, 1, 1.5, 1, 0.5, 0])
        peak_error, peak_reference = MeasureError.peak_error(signal1, signal2)
        assert peak_error == 1.0  # (3/1.5 - 1) = 1
        assert peak_reference == 1.5

    def test_peak_error_empty_signal_warning(self):
        """Test peak error with signal without peaks."""
        signal = array([0, 0, 0, 0])
        with warns(UserWarning, match="No peak observed"):
            MeasureError.peak_error(signal, signal)

    def test_integral_error_identical_signals(self, sample_signals):
        """Test integral error with identical signals."""
        signal, _, dt = sample_signals
        integral_error = MeasureError().integral_error(signal, signal, dt)
        assert integral_error == 0.0

    def test_integral_error_different_signals(self, sample_signals):
        """Test integral error with slightly different signals."""
        model, reference, dt = sample_signals
        error = MeasureError()
        integral_error = error.integral_error(model, reference, dt)
        assert 0.0 < integral_error < 1.0

    def test_integral_error_zero_reference(self, sample_signals):
        """Test integral error when reference signal is zero."""
        signal, _, dt = sample_signals
        zero_signal = zeros_like(signal)
        error = MeasureError()
        integral_error = error.integral_error(signal, zero_signal, dt)
        assert integral_error == inf

    def test_normalized_root_mean_square_error_identical(self, sample_signals):
        """Test NRMSE with identical signals."""
        signal, _, _ = sample_signals
        error = MeasureError()
        nrms_error = error.normalized_root_mean_square_error(signal, signal)
        assert nrms_error == 0.0

    def test_normalized_root_mean_square_error_different(self, sample_signals):
        """Test NRMSE with different signals."""
        model, reference, _ = sample_signals
        error = MeasureError()
        nrms_error = error.normalized_root_mean_square_error(model, reference)
        assert nrms_error > 0.0
        assert nrms_error < 1.0

    def test_save_reference_signal(self, measure_error, receiver_data, tmp_path):
        """Test saving reference signal."""
        with patch('spyro.tools.error_measure.getcwd', return_value=str(tmp_path)):
            # Mock save function to avoid actual file I/O
            with patch('spyro.tools.error_measure.save') as mock_save:
                receiver_locations = [(1, 2), (3, 4), (5, 6)]
                measure_error.save_reference_signal(
                    receiver_locations=receiver_locations,
                    forward_solution_receivers=receiver_data,
                    number_of_receivers=3, freq_Nyquist=50.0, output_file="test_ref")
                # Check that save was called twice (time and fft)
                assert mock_save.call_count == 2

    def test_error_measures_basic(self, measure_error, receiver_data):
        """Test basic error measures computation."""
        # Create simple test data
        n_time = 100
        n_rec = 2
        dt = 0.01

        # Create reference signal (sine wave)
        t = arange(0, n_time * dt, dt)
        ref = sin(2 * pi * 5 * t)[:, newaxis]
        ref = tile(ref, (1, n_rec))

        # Create model signal (slightly different)
        model = sin(2 * pi * 5 * t + 0.1)[:, newaxis]
        model = tile(model, (1, n_rec))

        error_measures = measure_error.error_measures(forward_solution_receivers=model,
                                                      receivers_reference=ref, dt=dt,
                                                      number_of_receivers=n_rec,
                                                      save_file=False)

        # Check structure of output
        assert len(error_measures) == 5  # [errIt, errPk, pkMax, max_errIt, max_errPK]
        assert len(error_measures[0]) == n_rec  # errIt list
        assert len(error_measures[1]) == n_rec  # errPk list
        assert len(error_measures[2]) == n_rec  # pkMax list
        assert error_measures[3] >= 0.0  # max_errIt
        assert error_measures[4] >= 0.0  # max_errPK

    def test_error_measures_with_energy(self, measure_error, receiver_data):
        """Test error measures with energy values."""
        # Create simple test data
        n_time = 100
        n_rec = 1
        dt = 0.01
        t = arange(0, n_time * dt, dt)
        ref = sin(2 * pi * 5 * t)[:, newaxis]
        model = sin(2 * pi * 5 * t + 0.1)[:, newaxis]

        error_measures = measure_error.error_measures(forward_solution_receivers=model,
                                                      receivers_reference=ref, dt=dt,
                                                      number_of_receivers=n_rec,
                                                      final_energy=0.5,
                                                      final_energy_reference=1.0,
                                                      save_file=False)

        # Check that energy values were added
        assert len(error_measures) == 7  # Added final_energy and dsspt_ener
        assert error_measures[5] == 0.5  # final_energy
        assert error_measures[6] == 0.5  # dissipated energy (1 - 0.5/1.0)

    def test_error_measures_save_file(self, receiver_data, tmp_path):
        """Test saving error measures to file."""
        # Create MeasureError with tmp_path
        measure_error = MeasureError()
        measure_error.initialize_paths_for_error(
            output_folder=str(tmp_path / "output" / "test_output"), output_case="test_case")

        # Create the directory structure
        output_dir = path.join(str(tmp_path), "output", "test_output", "test_case")
        makedirs(output_dir, exist_ok=True)

        with patch('spyro.tools.error_measure.getcwd', return_value=str(tmp_path)):
            n_time = 100
            n_rec = 1
            dt = 0.01
            t = arange(0, n_time * dt, dt)
            ref = sin(2 * pi * 5 * t)[:, newaxis]
            model = sin(2 * pi * 5 * t + 0.1)[:, newaxis]

            # Mock savetxt to avoid actual file writing
            with patch('spyro.tools.error_measure.savetxt') as mock_savetxt:
                measure_error.error_measures(forward_solution_receivers=model,
                                             receivers_reference=ref, dt=dt,
                                             number_of_receivers=n_rec,
                                             save_file=True, save_in_case_folder=True)
                # Check that savetxt was called at least once
                mock_savetxt.assert_called()

    @mark.parametrize("invalid_value", [-1, -0.5, "dt"])
    def test_error_measures_invalid_dt(self, measure_error, receiver_data, invalid_value):
        """Test error measures with invalid dt values."""
        if isinstance(invalid_value, str):
            with raises(TypeError):  # Strings raise TypeError
                measure_error.error_measures(forward_solution_receivers=receiver_data,
                                             receivers_reference=receiver_data,
                                             dt=invalid_value, number_of_receivers=3,
                                             save_file=False)
        else:
            with raises(ValueError):  # Negative numbers raise ValueError
                measure_error.error_measures(forward_solution_receivers=receiver_data,
                                             receivers_reference=receiver_data,
                                             dt=invalid_value, number_of_receivers=3,
                                             save_file=False)

    def test_get_reference_signal(self, receiver_data, tmp_path):
        """Test loading reference signal."""
        with patch('spyro.tools.error_measure.getcwd', return_value=str(tmp_path)):
            # Create MeasureError instance inside the patch context
            measure_error = MeasureError()
            measure_error.initialize_paths_for_error(output_folder="test_output",
                                                     output_case="preamble")
            # Create mock reference files
            with patch('spyro.tools.error_measure.load') as mock_load:
                mock_load.return_value = receiver_data
                with patch('spyro.tools.error_measure.save'):
                    # Mock path.exists where it's actually used
                    with patch('spyro.utils.error_management.path.exists', return_value=True):
                        # First save the reference
                        n_rec = receiver_data.shape[1]  # Get the actual number of receivers
                        measure_error.save_reference_signal(
                            receiver_locations=[(1, 2)] * n_rec,
                            forward_solution_receivers=receiver_data,
                            number_of_receivers=n_rec, freq_Nyquist=50.0,
                            output_file="test_ref")

                        # Then load it
                        ref, ref_fft = measure_error.get_reference_signal()

                        # Check that load was called twice (time and fft)
                        assert mock_load.call_count == 2
                        assert_array_equal(ref, receiver_data)
