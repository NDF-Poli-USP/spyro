import numpy as np
import warnings
import os

from firedrake import VTKFile

from .basicio import parallel_print


class Field:
    def __init__(self, name, file, callback):
        self.name = name
        self.file = file
        self.callback = callback

    def write(self, t):
        self.file.write(self.callback(), time=t, name=self.name)


class Functional:
    def __init__(self, filename, callback):
        self.filename = filename
        self.callback = callback
        self.list = []

    def sample(self):
        self.list.append(self.callback())

    def save(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        np.save(self.filename, self.list)


class FieldLogger:
    def __init__(self, comm, vis_dict):
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
        self.__wave_data.append((key, name, callback))

    def add_functional(self, key, callback):
        if self.__rank == 0:
            self.__func_data[key] = callback

    def start_logging(self, source_id):
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
        self.__source_id = None

        if self.__rank == 0:
            if self.__time_enabled:
                np.save(self.__time_filename, self.__time)

            for functional in self.__enabled_functionals.values():
                functional.save()

    def log(self, t):
        for field in self.__enabled_fields:
            field.write(t)

        if self.__rank == 0:
            if self.__time_enabled:
                self.__time.append(t)

            for functional in self.__enabled_functionals.values():
                functional.sample()

    def get(self, key):
        return self.__enabled_functionals[key].list[-1]
