"""Utilities for logging vector and scalar fields and functionals.

This module provides lightweight wrappers around Firedrake VTK output and
NumPy array persistence to support time-stepped logging during simulations.
"""

import numpy as np
import warnings

from firedrake import VTKFile

from .basicio import parallel_print


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

    def __init__(self, name, file, callback):
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

    def write(self, t):
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
        Destination ``.npy`` filename used to save sampled values.
    callback : callable
        Zero-argument callable that returns the current scalar sample.
    """

    def __init__(self, filename, callback):
        """Initialize Functional class.

        Parameters
        ----------
        filename : str
            Destination ``.npy`` filename used to save sampled values.
        callback : callable
            Zero-argument callable that returns the current scalar sample.
        """
        
        self.filename = filename
        self.callback = callback
        self.list = []

    def sample(self):
        """Append the current callback value to the internal sample list."""
        
        self.list.append(self.callback())

    def save(self):
        """Save all collected samples to the configured NumPy file."""
        
        np.save(self.filename, self.list)


class FieldLogger:
    """Coordinate field and functional logging across MPI ranks.

    Parameters
    ----------
    comm : object
        Communication wrapper with ``comm`` MPI communicator and rank support.
    vis_dict : dict
        Visualization and logging configuration dictionary. Keys control which
        outputs are enabled and the associated output filenames.
    """

    def __init__(self, comm, vis_dict):
        """Initialize FieldLogger class.
        
        Parameters
        ----------
        comm : object
            Communication wrapper with ``comm`` MPI communicator and rank support.
        vis_dict : dict
            Visualization and logging configuration dictionary. Keys control which
            outputs are enabled and the associated output filenames.
        """
        self.comm = comm
        self.vis_dict = vis_dict

        self.__source_id = None
        self.__enabled_fields = []
        self.__wave_data = []

        self.__rank = comm.comm.Get_rank()
        if self.__rank == 0:
            self.__func_data = {}
            self.__enabled_functionals = {}

            self.__time_enabled = self.vis_dict.get("time", False)
            if self.__time_enabled:
                self.__time = []
                self.__time_filename = self.vis_dict.get("time_filename", "time.npy")
                print(f"Saving time in: {self.__time_filename}")

    def add_field(self, key, name, callback):
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

    def add_functional(self, key, callback):
        """Register a scalar functional callback on rank zero.

        Parameters
        ----------
        key : str
            Configuration key controlling whether this functional is enabled.
        callback : callable
            Zero-argument callable returning the current functional value.
        """
        if self.__rank == 0:
            self.__func_data[key] = callback

    def start_logging(self, source_id):
        """Initialize logging targets for a given source id.

        Parameters
        ----------
        source_id : int
            Source index used to disambiguate output filenames.

        Warns
        -----
        UserWarning
            If a new logging session starts before the previous one is stopped.
        """
        if self.__source_id is not None:
            warnings.warn("Started a new record without stopping the previous one")

        self.__source_id = source_id
        self.__enabled_fields = []
        for key, name, callback in self.__wave_data:
            enabled = self.vis_dict.get(key + "_output", False)
            if enabled:
                fullname = self.vis_dict.get(key + "_output_filename", key + ".pvd")
                prefix, extension = fullname.split(".")
                filename = prefix + "sn" + str(source_id) + "." + extension

                parallel_print(f"Saving {name} in: {filename}", self.comm)

                file = VTKFile(filename, comm=self.comm.comm)
                self.__enabled_fields.append(Field(name, file, callback))

        if self.__rank == 0:
            if self.__time_enabled:
                self.__time = []

            for key, callback in self.__func_data.items():
                enabled = self.vis_dict.get(key, False)
                if enabled:
                    filename = self.vis_dict.get(key + "_filename", key + ".npy")
                    print(f"Saving {key} in: {filename}")
                    self.__enabled_functionals[key] = Functional(filename, callback)

    def stop_logging(self):
        """Finalize the current logging session and persist accumulated data."""
        
        self.__source_id = None

        if self.__rank == 0:
            if self.__time_enabled:
                np.save(self.__time_filename, self.__time)

            for functional in self.__enabled_functionals.values():
                functional.save()

    def log(self, t):
        """Write enabled fields and sample enabled functionals.

        Parameters
        ----------
        t : float
            Simulation time associated with the current logging step.
        """
        for field in self.__enabled_fields:
            field.write(t)

        if self.__rank == 0:
            if self.__time_enabled:
                self.__time.append(t)

            for functional in self.__enabled_functionals.values():
                functional.sample()

    def get(self, key):
        """Return the latest sampled value for an enabled functional.

        Parameters
        ----------
        key : str
            Functional key used during registration.

        Returns
        -------
        object
            Most recently sampled value for the requested functional.

        Raises
        ------
        KeyError
            If the functional key is not enabled in the current session.
        IndexError
            If no samples have been recorded yet for the requested functional.
        """
        
        return self.__enabled_functionals[key].list[-1]
