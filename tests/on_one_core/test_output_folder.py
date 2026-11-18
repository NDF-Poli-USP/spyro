"""Tests that spyro writes its outputs inside the configured output folder."""

from pathlib import Path

import firedrake as fire
import pytest
from scipy.optimize import OptimizeResult

import spyro
from spyro.io.dictionaryio import Read_outputs
from spyro.io.field_logger import FieldLogger
from spyro.mpi.spyro_mpi import AutomaticParallelism, SpyroEnsemble


def build_acoustic_dictionary(visualization: dict | None = None) -> dict:
    """Build a small acoustic model dictionary.

    Parameters
    ----------
    visualization : dict, optional
        Visualization section of the dictionary. Default is a section that only
        asks for the forward output.

    Returns
    -------
    dict
        Model dictionary of a short two-dimensional acoustic simulation.
    """
    if visualization is None:
        visualization = {"forward_output": True}
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.5)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.02,
            "dt": 0.002,
            "amplitude": 1.0,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": visualization,
    }


@pytest.fixture
def field() -> fire.Function:
    """Initialize the ensemble and build a field for the logger to write.

    Returns
    -------
    firedrake.Function
        A DG0 field on a unit interval mesh.
    """
    SpyroEnsemble.initialize(AutomaticParallelism(number_of_sources=1))
    mesh = fire.UnitIntervalMesh(2)
    return fire.Function(fire.FunctionSpace(mesh, "DG", 0))


def test_default_output_filenames_are_relative_to_output_folder() -> None:
    """The default output filenames name subfolders of `output/`.

    Returns
    -------
    None
    """
    outputs = Read_outputs()

    assert outputs.output_folder == "output/"
    assert outputs.forward_output_filename == "fields/forward_output.pvd"
    assert outputs.acoustic_energy_filename == "functionals/acoustic_energy.npy"
    assert outputs.mechanical_energy_filename == "functionals/mechanical_energy.npy"


@pytest.mark.parametrize("output_folder", ["run", "./run.1/"])
def test_field_filename_is_relative_to_output_folder(
    output_folder: str,
    field: fire.Function,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative field filename is placed inside the output folder.

    Parameters
    ----------
    output_folder : str
        Output folder of the visualization dictionary, with and without a dot
        in its name.
    field : firedrake.Function
        Field the logger writes.
    tmp_path : pathlib.Path
        Working directory of the run.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory to `tmp_path`.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)
    logger = FieldLogger(
        {
            "output_folder": output_folder,
            "u_output": True,
            "u_output_filename": "fields/u.pvd",
        }
    )
    logger.add_field("u", "u", lambda: field)

    logger.start_logging(0)
    logger.log(0.0)
    logger.stop_logging()

    assert (tmp_path / output_folder / "fields" / "usn0.pvd").is_file()


def test_absolute_field_filename_is_kept(
    field: fire.Function, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute field filename is used as given.

    Parameters
    ----------
    field : firedrake.Function
        Field the logger writes.
    tmp_path : pathlib.Path
        Working directory of the run.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory to `tmp_path`.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)
    absolute_filename = tmp_path / "elsewhere" / "u.pvd"
    logger = FieldLogger(
        {
            "output_folder": "run",
            "u_output": True,
            "u_output_filename": str(absolute_filename),
        }
    )
    logger.add_field("u", "u", lambda: field)

    logger.start_logging(0)
    logger.log(0.0)
    logger.stop_logging()

    assert (tmp_path / "elsewhere" / "usn0.pvd").is_file()
    assert not (tmp_path / "run").exists()


def test_functionals_are_saved_inside_output_folder(
    field: fire.Function, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Functionals and times are saved in the output folder, default name included.

    The acoustic energy filename is left at its default, and the folders of
    both files are created on save.

    Parameters
    ----------
    field : firedrake.Function
        Unused field; the fixture initializes the ensemble.
    tmp_path : pathlib.Path
        Working directory of the run.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory to `tmp_path`.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)
    logger = FieldLogger(
        Read_outputs(
            output_folder="run",
            forward_output=False,
            acoustic_energy=True,
            time_filename="functionals/time",
        )
    )
    logger.add_functional("acoustic_energy", lambda: 1.0)

    logger.start_logging(0)
    logger.log(0.0)
    logger.stop_logging()

    assert (tmp_path / "run" / "functionals" / "acoustic_energy.npy").is_file()
    assert (tmp_path / "run" / "functionals" / "time.npy").is_file()


def test_wave_outputs_are_written_inside_output_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forward run with default settings writes only inside `output/`.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the run.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory to `tmp_path`.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.set_initial_velocity_model(constant=1.5, output=True)
    wave.set_material_property("density", "scalar", constant=1.0, output=True)
    wave.forward_solve()

    output = tmp_path / "output"
    assert list((output / "fields").glob("forward_outputsn*.pvd"))
    assert (output / "velocity_models" / "initial_velocity_model.pvd").is_file()
    assert (output / "property_fields" / "density.pvd").is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output"]


def test_fwi_outputs_are_written_inside_output_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inversion writes its results in the `fwi` subfolder.

    Its dictionary output files are relative to the output folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the run.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory to `tmp_path` and replaces the scipy
        optimizer, so no wave is propagated.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)

    def fake_minimize(fun, x0, **kwargs) -> OptimizeResult:
        """Stop at the starting point.

        Parameters
        ----------
        fun : callable
            Objective, never evaluated.
        x0 : numpy.ndarray
            Starting point, returned as the solution.
        **kwargs : dict
            Remaining arguments of :func:`scipy.optimize.minimize`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            A result whose solution is `x0`.
        """
        return OptimizeResult(x=x0)

    monkeypatch.setattr("spyro.solvers.inversion.scipy_minimize", fake_minimize)
    dictionary = build_acoustic_dictionary({"output_folder": "run"})
    dictionary["inversion"] = {"control_output_file": "custom/control.pvd"}
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=2.0)

    fwi.run_fwi(vmin=1.5, vmax=3.0, maxiter=1)

    run = tmp_path / "run"
    assert fwi.fwi_output_folder == str(Path("run") / "fwi")
    assert (run / "fwi" / "control_end.pvd").is_file()
    assert (run / "fwi" / "result.npy").is_file()
    assert (run / "custom" / "control").is_dir()
    assert (run / "fwi" / "gradient").is_dir()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run"]


@pytest.mark.parametrize(
    "visualization, folder", [({}, "output"), ({"output_folder": "run"}, "run")]
)
def test_acquisition_files_are_saved_inside_output_folder(
    visualization: dict, folder: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source and receiver locations are saved in the `acquisition` subfolder.

    Parameters
    ----------
    visualization : dict
        Visualization section of the model dictionary.
    folder : str
        Output folder the section selects.
    tmp_path : pathlib.Path
        Working directory of the run.
    monkeypatch : pytest.MonkeyPatch
        Changes the working directory to `tmp_path`.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)
    model = {
        "acquisition": {
            "source_locations": [(-0.1, 0.5)],
            "receiver_locations": [(-0.2, 0.25)],
        },
        "visualization": visualization,
    }

    spyro.io.saving_source_and_receiver_location_in_csv(model)

    assert (tmp_path / folder / "acquisition" / "sources.txt").is_file()
    assert (tmp_path / folder / "acquisition" / "receivers.txt").is_file()
