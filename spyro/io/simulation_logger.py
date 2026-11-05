"""Utilities for logging the fields, functionals and time of a simulation.

This module provides lightweight wrappers around Firedrake VTK output and NumPy array
persistence to support time-stepped logging during simulations.
"""

from collections.abc import Callable
from pathlib import Path
import warnings

import numpy as np

from firedrake import Function, VTKFile
from spyro.mpi.spyro_mpi import SpyroEnsemble


def _shot_filename(filename: str, source_id: int | list[int], extension: str) -> str:
    """Append the shot suffix of a propagation to an output filename.

    Parameters
    ----------
    filename : str
        Output filename, with or without `extension`. Dots elsewhere in the path,
        such as in `./results/`, are kept.
    source_id : int or list of int
        Source ids of the propagation, written into the suffix with `str`.
    extension : str
        File extension, including the leading dot, such as `.pvd`.

    Returns
    -------
    str
        `filename` without `extension`, followed by `sn`, `str(source_id)` and
        `extension`. For example, `results/forward.pvd` with source ids `[0]` becomes
        `results/forwardsn[0].pvd`.
    """
    stem = filename.removesuffix(extension)
    return f"{stem}sn{source_id}{extension}"


def _save_array(filename: str, values: list) -> None:
    """Save values to a NumPy file, creating its parent directory if needed.

    Parameters
    ----------
    filename : str
        Destination filename, passed to `numpy.save`.
    values : list
        Values to save.

    Returns
    -------
    None
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, values)


class Field:
    """Represent a named field writer with a deferred data callback.

    Parameters
    ----------
    name : str
        Name used when writing the field to the VTK output.
    file : firedrake.output.vtk_output.VTKFile
        Open VTK file handle used for output writes.
    callback : callable
        Zero-argument callable that returns the field object to be written.
    """

    def __init__(self, name: str, file: VTKFile, callback: Callable[[], Function]):
        """Initialize Field class.

        Parameters
        ----------
        name : str
            Name used when writing the field to the VTK output.
        file : firedrake.output.vtk_output.VTKFile
            Open VTK file handle used for output writes.
        callback : callable
            Zero-argument callable that returns the field object to be written.
        """
        self.name = name
        self.file = file
        self.callback = callback

    def write(self, t: float):
        """Write the current field value at a given time.

        Parameters
        ----------
        t : float
            Simulation time associated with the current field sample.
        """
        self.file.write(self.callback(), time=t, name=self.name)


class Functional:
    """Collect scalar samples from a callback and persist them to disk.

    Parameters
    ----------
    filename : str
        Destination `.npy` filename used to save sampled values.
    callback : callable
        Zero-argument callable that returns the current scalar sample.
    """

    def __init__(self, filename: str, callback: Callable[[], float]):
        """Initialize Functional class.

        Parameters
        ----------
        filename : str
            Destination `.npy` filename used to save sampled values.
        callback : callable
            Zero-argument callable that returns the current scalar sample.
        """
        self.filename = filename
        self.callback = callback
        self.list = []

    def sample(self):
        """Append the current callback value to the internal sample list."""
        self.list.append(self.callback())

    def save(self) -> None:
        """Save all collected samples to the configured NumPy file.

        The parent directory of the file is created if it does not exist.

        Returns
        -------
        None
        """
        _save_array(self.filename, self.list)


class SimulationLogger:
    """Log the fields, scalar functionals and time of a simulation.

    Fields are written to VTK files at every logging step. Functionals and the time
    are sampled at every logging step and saved to NumPy files when the propagation
    stops.

    Parameters
    ----------
    visualization_dict : dict or spyro.io.dictionaryio.Read_outputs
        Visualization and logging configuration. Keys control which outputs are
        enabled and the associated output filenames.

    Notes
    -----
    Every field and functional filename carries the suffix `sn` followed by the
    source ids of the propagation, such as `forward_outputsn[0].pvd`, so that each
    shot, whether propagated in serial or by its own ensemble member, has its own
    files. Each ensemble member saves its functionals from its first spatial rank.
    The time is the same for every propagation and is saved, without a suffix, only
    by the global rank 0.
    """

    def __init__(self, visualization_dict):
        """Initialize SimulationLogger class.

        Parameters
        ----------
        visualization_dict : dict or spyro.io.dictionaryio.Read_outputs
            Visualization and logging configuration. Keys control which outputs are
            enabled and the associated output filenames.
        """
        self.visualization_dict = visualization_dict

        self.__source_id = None
        self.__enabled_fields = []
        self.__wave_data = []
        self.__func_data = {}
        self.__enabled_functionals = {}

        self.__time = []
        self.__time_filename = self.visualization_dict.get("time_filename", None)
        self.__time_enabled = self.__time_filename is not None
        if self.__time_enabled:
            SpyroEnsemble.print(f"Saving time in: {self.__time_filename}")

    def add_field(self, key: str, name: str, callback):
        """Register a field candidate for logging.

        Parameters
        ----------
        key : str
            Configuration prefix used to resolve enable and filename options.
        name : str
            Variable name written into VTK output.
        callback : callable
            Zero-argument callable returning the current field to write.
        """
        self.__wave_data.append((key, name, callback))

    def add_functional(self, key: str, callback: Callable[[], float]) -> None:
        """Register a scalar functional callback.

        Parameters
        ----------
        key : str
            Configuration key controlling whether this functional is enabled.
        callback : callable
            Zero-argument callable returning the current functional value.

        Returns
        -------
        None

        Notes
        -----
        The callback is evaluated on every rank, since it is usually a collective
        operation, such as `firedrake.assemble`, over the spatial communicator.
        """
        self.__func_data[key] = callback

    def start_logging(self, source_id):
        """Initialize logging targets for a given source id.

        Parameters
        ----------
        source_id : int or list of int
            Source ids of the propagation, used to disambiguate output filenames.

        Warns
        -----
        UserWarning
            If a new logging session starts before the previous one is stopped.
        """
        if self.__source_id is not None:
            warnings.warn("Started a new record without stopping the previous one")

        self.__source_id = source_id

        self._init_fields(source_id)
        self._init_functionals(source_id)

    def _init_fields(self, source_id: int | list[int]) -> None:
        """Open the VTK files of the enabled fields for a propagation.

        Parameters
        ----------
        source_id : int or list of int
            Source ids of the propagation, used to disambiguate output filenames.

        Returns
        -------
        None
        """
        self.__enabled_fields = []
        for key, name, callback in self.__wave_data:
            if not self.visualization_dict.get(key + "_output", False):
                continue

            fullname = (
                self.visualization_dict.get(key + "_output_filename") or key + ".pvd"
            )
            filename = _shot_filename(fullname, source_id, ".pvd")

            SpyroEnsemble.print(f"Saving {name} in: {filename}")

            file = VTKFile(filename, comm=SpyroEnsemble.ensemble.comm)
            self.__enabled_fields.append(Field(name, file, callback))

    def _init_functionals(self, source_id: int | list[int]) -> None:
        """Reset the time and the enabled functionals for a propagation.

        Parameters
        ----------
        source_id : int or list of int
            Source ids of the propagation, used to disambiguate output filenames.

        Returns
        -------
        None
        """
        self.__time = []

        self.__enabled_functionals = {}
        for key, callback in self.__func_data.items():
            if self.visualization_dict.get(key, False):
                fullname = self.visualization_dict.get(key + "_filename") or key
                filename = _shot_filename(fullname, source_id, ".npy")
                SpyroEnsemble.print(f"Saving {key} in: {filename}")
                self.__enabled_functionals[key] = Functional(filename, callback)

    def stop_logging(self):
        """Finalize the current logging session and persist accumulated data."""
        self.__source_id = None
        self._save_logs()

    def _save_logs(self) -> None:
        """Save the time and the functional samples of the propagation.

        Returns
        -------
        None
        """
        if self.__time_enabled and SpyroEnsemble.get_global_rank() == 0:
            _save_array(self.__time_filename, self.__time)

        if SpyroEnsemble.get_local_rank() == 0:
            for functional in self.__enabled_functionals.values():
                functional.save()

    def log(self, t: float):
        """Write enabled fields and sample enabled functionals.

        Parameters
        ----------
        t : float
            Simulation time associated with the current logging step.
        """
        for field in self.__enabled_fields:
            field.write(t)

        self._sample_functionals(t)

    def _sample_functionals(self, t: float) -> None:
        """Record the time and sample the enabled functionals on every rank.

        Parameters
        ----------
        t : float
            Simulation time associated with the current logging step.

        Returns
        -------
        None
        """
        if self.__time_enabled:
            self.__time.append(t)

        for functional in self.__enabled_functionals.values():
            functional.sample()

    def get(self, key: str):
        """Return the latest sampled value for an enabled functional.

        Parameters
        ----------
        key : str
            Functional key used during registration.

        Returns
        -------
        object
            Most recently sampled value for the requested functional. It is
            available on every rank.

        Raises
        ------
        KeyError
            If the functional key is not enabled in the current session.
        IndexError
            If no samples have been recorded yet for the requested functional.
        """
        return self.__enabled_functionals[key].list[-1]
