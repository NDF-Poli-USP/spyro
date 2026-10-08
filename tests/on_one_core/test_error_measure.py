"""Unit tests for the error measures implemented in spyro.tools.error_measure.

These tests cover the MeasureError class and its methods, ensuring
that error calculations and file operations work as expected.
"""

import numpy as np
import pytest
from unittest.mock import patch
from spyro.tools.error_measure import MeasureError


class TestMeasureError:
    """Test suite for MeasureError class."""

    @pytest.fixture
    def measure_error(self):
        """Create a MeasureError instance for testing."""
        output_folder = "/output/test_output"
        output_case = "test_case"
        return MeasureError(output_folder=output_folder, output_case=output_case)

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for testing."""
        dt = 0.01
        t = np.arange(0, 1, dt)
        # Create a simple sine wave as reference
        reference = np.sin(2 * np.pi * 5 * t)
        # Create a slightly different signal as model
        model = np.sin(2 * np.pi * 5 * t + 0.1)
        return model, reference, dt

    @pytest.fixture
    def acoustic_receiver_data(self):
        """Create synthetic acoustic seismic receiver data."""
        dt = 0.001
        n_time = 20001
        n_receivers = 40

        t = np.arange(n_time) * dt

        # Ricker wavelet
        frequency = 10.0
        t0 = 2.0
        tau = np.pi * frequency * (t - t0)
        reference = (1 - 2 * tau**2) * np.exp(-tau**2)

        # Receiver-dependent amplitudes
        amplitudes = np.linspace(1.0, 0.7, n_receivers)

        reference = reference[:, np.newaxis] * amplitudes[np.newaxis, :]

        # Numerical signal = reference + reproducible random noise
        rng = np.random.default_rng(42)
        noise = 0.01 * rng.standard_normal(reference.shape)

        numerical = reference + noise

        return numerical, reference, dt

    @pytest.fixture
    def elastic_receiver_data(self):
        """Create synthetic elastic seismic receiver data."""
        dt = 0.001
        n_time = 20001
        n_receivers = 40

        t = np.arange(n_time) * dt

        # Ricker wavelet
        frequency = 10.0
        t0 = 2.0
        tau = np.pi * frequency * (t - t0)
        ricker = (1 - 2 * tau**2) * np.exp(-tau**2)

        # Receiver-dependent amplitudes
        receiver_amplitudes = np.linspace(1.0, 0.7, n_receivers)

        # Different amplitudes for the two displacement directions
        direction_amplitudes = np.array([1.0, 0.6])

        reference = (
            ricker[:, np.newaxis, np.newaxis]
            * receiver_amplitudes[np.newaxis, :, np.newaxis]
            * direction_amplitudes[np.newaxis, np.newaxis, :]
        )

        # Numerical signal = reference + reproducible random noise
        rng = np.random.default_rng(42)
        noise = 0.01 * rng.standard_normal(reference.shape)

        numerical = reference + noise

        return numerical, reference, dt

    def test_initialization_default(self):
        """Test default initialization."""
        error = MeasureError()
        assert error.path_save_error.name == "output"
        assert error.path_save_err_case.name == "output"
        assert error.path_reference.name == "preamble"
        assert error.path_reference.parent.name == "output"
        assert error.comm is None

    def test_initialization_custom_paths(self):
        """Test initialization with custom paths."""
        output_folder = "/custom/path"
        output_case = "case123"
        error = MeasureError(output_folder=output_folder, output_case=output_case)
        assert str(error.path_save_error) == "/custom/path"
        assert str(error.path_save_err_case) == "case123"

    def test_pad_signal_lengths_equal(self):
        """Test when signals have equal lengths."""
        signal1 = np.array([1, 2, 3, 4])
        signal2 = np.array([5, 6, 7, 8])
        result1, result2 = MeasureError.pad_signal_lengths(signal1, signal2)
        assert len(result1) == len(result2) == 4
        np.testing.assert_array_equal(result1, signal1)
        np.testing.assert_array_equal(result2, signal2)

    def test_pad_signal_lengths_different_default(self):
        """Test when signals have different lengths with default behavior."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6, 7, 8])
        # Default is error_if_different_length=True
        with pytest.raises(
            ValueError, match="The lengths of the model and reference signals"
        ):
            MeasureError.pad_signal_lengths(signal1, signal2)

    def test_pad_signal_lengths_both_padding_error(self):
        """Test that both start_padding and end_padding cannot be equal simultaneously."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6, 7, 8])
        with pytest.raises(ValueError, match="are mutually exclusive."):
            MeasureError.pad_signal_lengths(
                signal1,
                signal2,
                error_if_different_length=False,
                start_padding=True,
                end_padding=True,
            )
        with pytest.raises(ValueError, match="are mutually exclusive"):
            MeasureError.pad_signal_lengths(
                signal1,
                signal2,
                error_if_different_length=False,
                start_padding=False,
                end_padding=False,
            )

    def test_pad_signal_lengths_start_padding(self):
        """Test padding at the start of the shorter signal."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6, 7, 8])
        result1, result2 = MeasureError.pad_signal_lengths(
            signal1,
            signal2,
            error_if_different_length=False,
            start_padding=True,
            end_padding=False,
        )
        assert len(result1) == len(result2) == 5
        np.testing.assert_array_equal(result2, signal2)
        assert result1[0] == 0 and result1[1] == 0
        np.testing.assert_array_equal(result1[2:], signal1)
        result2, result1 = MeasureError.pad_signal_lengths(
            signal2,
            signal1,
            error_if_different_length=False,
            start_padding=True,
            end_padding=False,
        )
        assert len(result1) == len(result2) == 5
        np.testing.assert_array_equal(result2, signal2)
        assert result1[0] == 0 and result1[1] == 0
        np.testing.assert_array_equal(result1[2:], signal1)

    def test_pad_signal_lengths_end_padding(self):
        """Test padding at the end of the shorter signal."""
        signal1 = np.array([1, 2, 3])
        signal2 = np.array([4, 5, 6, 7, 8])
        result1, result2 = MeasureError.pad_signal_lengths(
            signal1,
            signal2,
            error_if_different_length=False,
            start_padding=False,
            end_padding=True,
        )
        assert len(result1) == len(result2) == 5
        np.testing.assert_array_equal(result2, signal2)
        np.testing.assert_array_equal(result1[:3], signal1)
        assert result1[3] == 0 and result1[4] == 0
        result2, result1 = MeasureError.pad_signal_lengths(
            signal2,
            signal1,
            error_if_different_length=False,
            start_padding=False,
            end_padding=True,
        )
        assert len(result1) == len(result2) == 5
        np.testing.assert_array_equal(result2, signal2)
        np.testing.assert_array_equal(result1[:3], signal1)
        assert result1[3] == 0 and result1[4] == 0

    def test_pad_signal_lengths_equal_lengths_with_padding_options(self):
        """Test that padding options are ignored when signals have equal length."""
        signal1 = np.array([1, 2, 3, 4, 5])
        signal2 = np.array([6, 7, 8, 9, 10])
        # The padding options check only runs when lengths are different
        # Both start_padding and end_padding are False but signals are equal length
        result1, result2 = MeasureError.pad_signal_lengths(
            signal1,
            signal2,
            error_if_different_length=False,
            start_padding=False,
            end_padding=False,
        )
        assert len(result1) == len(result2) == 5
        np.testing.assert_array_equal(result1, signal1)
        np.testing.assert_array_equal(result2, signal2)
        # Both start_padding and end_padding are True but signals are equal length
        result1, result2 = MeasureError.pad_signal_lengths(
            signal1,
            signal2,
            error_if_different_length=False,
            start_padding=True,
            end_padding=True,
        )
        assert len(result1) == len(result2) == 5
        np.testing.assert_array_equal(result1, signal1)
        np.testing.assert_array_equal(result2, signal2)

    def test_peak_error_identical_signals(self):
        """Test peak error with identical signals."""
        signal = np.array([0, 1, 2, 3, 2, 1, 0])
        peak_error, peak_reference = MeasureError.calculate_peak_error(signal, signal)
        assert peak_error == 0.0
        assert peak_reference == 3.0

    def test_peak_error_different_signals(self):
        """Test peak error with different signals."""
        signal1 = np.array([0, 1, 2, 3, 2, 1, 0])
        signal2 = np.array([0, 0.5, 1, 1.5, 1, 0.5, 0])
        peak_error, peak_reference = MeasureError.calculate_peak_error(signal1, signal2)
        assert peak_error == 1.0  # (3/1.5 - 1) = 1
        assert peak_reference == 1.5

    def test_peak_error_empty_signal_warning(self):
        """Test peak error with signal without peaks."""
        signal = np.array([0, 0, 0, 0])
        with pytest.warns(UserWarning, match="No peak observed"):
            MeasureError.calculate_peak_error(signal, signal)

    def test_integral_error_identical_signals(self, sample_signals):
        """Test integral error with identical signals."""
        signal, _, dt = sample_signals
        integral_error = MeasureError.calculate_integral_error(signal, signal, dt)
        assert integral_error == 0.0

    def test_integral_error_different_signals(self, sample_signals):
        """Test integral error with slightly different signals."""
        model, reference, dt = sample_signals
        integral_error = MeasureError.calculate_integral_error(model, reference, dt)
        assert 0.0 < integral_error < 1.0

    def test_integral_error_zero_reference(self, sample_signals):
        """Test integral error when reference signal is zero."""
        signal, _, dt = sample_signals
        zero_signal = np.zeros_like(signal)
        integral_error = MeasureError.calculate_integral_error(signal, zero_signal, dt)
        assert integral_error == np.inf

    def test_normalized_L2_error_identical(self, sample_signals):
        """Test NRMSE with identical signals."""
        signal, _, _ = sample_signals
        nrms_error = MeasureError.calculate_normalized_L2_error(signal, signal)
        assert nrms_error == 0.0

    def test_normalized_L2_error_different(self, sample_signals):
        """Test NRMSE with different signals."""
        model, reference, _ = sample_signals
        nrms_error = MeasureError.calculate_normalized_L2_error(model, reference)
        assert nrms_error > 0.0
        assert nrms_error < 1.0

    def test_save_reference_signal(
        self,
        measure_error,
        acoustic_receiver_data,
        tmp_path,
    ):
        """Test saving reference signal."""
        numerical, _, _ = acoustic_receiver_data
        n_receivers = numerical.shape[1]

        receiver_locations = [
            (i, i + 1)
            for i in range(n_receivers)
        ]

        with patch(
            "spyro.tools.error_measure.getcwd",
            return_value=str(tmp_path),
        ):
            with patch("numpy.save") as mock_save:
                measure_error.save_reference_signal(
                    receiver_locations=receiver_locations,
                    forward_solution_receivers=numerical,
                    number_of_receivers=n_receivers,
                    nyquist_frequency=50.0,
                    output_file_prefix="test_ref",
                )

        # One file for the time-domain signal and one for its FFT.
        assert mock_save.call_count == 2

    def test_error_measures_basic(
        self,
        measure_error,
        acoustic_receiver_data,
    ):
        """Test basic error measures computation."""
        numerical, reference, dt = acoustic_receiver_data
        n_rec = reference.shape[1]

        error_measures = measure_error.calculate_error_measures(
            forward_solution_receivers=numerical,
            receivers_reference=reference,
            dt=dt,
            number_of_receivers=n_rec,
            save_file=False,
        )

        err_it, err_pk, pk_max, max_err_it, max_err_pk = error_measures

        assert len(err_it) == n_rec
        assert len(err_pk) == n_rec
        assert len(pk_max) == n_rec

        assert all(np.isfinite(err_it))
        assert all(np.isfinite(err_pk))
        assert all(np.isfinite(pk_max))

        assert max_err_it == max(err_it)
        assert max_err_pk == max(err_pk)

        # The numerical signal differs from the reference only by
        # low-amplitude random noise.
        assert max_err_it < 0.14
        assert max_err_pk < 0.14

    def test_error_measures_with_energy(self, measure_error):
        """Test error measures with energy values."""
        # Create simple test data
        n_time = 100
        n_rec = 1
        dt = 0.01
        t = np.arange(0, n_time * dt, dt)
        ref = np.sin(2 * np.pi * 5 * t)[:, np.newaxis]
        model = np.sin(2 * np.pi * 5 * t + 0.1)[:, np.newaxis]

        error_measures = measure_error.calculate_error_measures(
            forward_solution_receivers=model,
            receivers_reference=ref,
            dt=dt,
            number_of_receivers=n_rec,
            final_energy=0.5,
            final_energy_reference=1.0,
            save_file=False,
        )

        # Check that energy values were added
        assert len(error_measures) == 7  # Added final_energy and dsspt_ener
        assert error_measures[5] == 0.5  # final_energy
        assert error_measures[6] == 0.5  # dissipated energy (1 - 0.5/1.0)

    def test_error_measures_save_file(self, measure_error, tmp_path):
        """Test saving error measures to file."""
        with patch("spyro.tools.error_measure.getcwd", return_value=str(tmp_path)):
            n_time = 100
            n_rec = 1
            dt = 0.01
            t = np.arange(0, n_time * dt, dt)
            ref = np.sin(2 * np.pi * 5 * t)[:, np.newaxis]
            model = np.sin(2 * np.pi * 5 * t + 0.1)[:, np.newaxis]

            # Mock savetxt to avoid actual file writing
            with patch("numpy.savetxt") as mock_savetxt:
                measure_error.calculate_error_measures(
                    forward_solution_receivers=model,
                    receivers_reference=ref,
                    dt=dt,
                    number_of_receivers=n_rec,
                    save_file=True,
                    save_in_case_folder=True,
                )
                # Check that savetxt was called at least once
                mock_savetxt.assert_called()

    @pytest.mark.parametrize("invalid_value", [-1, -0.5, "dt"])
    def test_error_measures_invalid_dt(
        self,
        measure_error,
        acoustic_receiver_data,
        invalid_value,
    ):
        """Test error measures with invalid dt values."""
        numerical, reference, _ = acoustic_receiver_data
        n_receivers = numerical.shape[1]

        expected_exception = TypeError if isinstance(invalid_value, str) else ValueError

        with pytest.raises(expected_exception):
            measure_error.calculate_error_measures(
                forward_solution_receivers=numerical,
                receivers_reference=reference,
                dt=invalid_value,
                number_of_receivers=n_receivers,
                save_file=False,
            )

    def test_get_reference_signal(
        self,
        acoustic_receiver_data,
        tmp_path,
    ):
        """Test loading reference signal."""
        numerical, _, _ = acoustic_receiver_data
        n_receivers = numerical.shape[1]

        with patch(
            "spyro.tools.error_measure.getcwd",
            return_value=str(tmp_path),
        ):
            measure_error = MeasureError(
                output_folder="test_output",
                output_case="preamble",
            )

            with patch("numpy.load") as mock_load:
                mock_load.return_value = numerical

                with patch("numpy.save"):
                    with patch(
                        "spyro.utils.error_management.path.exists",
                        return_value=True,
                    ):
                        measure_error.save_reference_signal(
                            receiver_locations=[(i, i + 1) for i in range(n_receivers)],
                            forward_solution_receivers=numerical,
                            number_of_receivers=n_receivers,
                            nyquist_frequency=50.0,
                            output_file_prefix="test_ref",
                        )

                        ref, ref_fft = measure_error.get_reference_signal()

                        # One file for the time-domain signal and one for its FFT.
                        assert mock_load.call_count == 2

                        np.testing.assert_array_equal(ref, numerical)

    @pytest.mark.parametrize(
        "physics",
        ["acoustic", "elastic"],
    )
    def test_calculate_receiver_error(
        self,
        request,
        physics,
    ):
        """Test receiver error for acoustic and elastic data."""
        if physics == "acoustic":
            numerical, reference, dt = request.getfixturevalue(
                "acoustic_receiver_data"
            )
            has_displacement = False
        elif physics == "elastic":
            numerical, reference, dt = request.getfixturevalue(
                "elastic_receiver_data"
            )
            has_displacement = True

        error = MeasureError.calculate_receiver_error(
            numerical,
            reference,
            dt,
            has_displacement=has_displacement,
        )

        # Numerical data differs from the reference only by low-amplitude noise.
        assert 0.0 < error < 1.0
