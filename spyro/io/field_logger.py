import warnings

from firedrake import VTKFile

from .basicio import parallel_print

class Field:
    def __init__(self, name, file, callback):
        self.name = name
        self.file = file
        self.callback = callback
    
    def write(self, t):
        self.file.write(self.callback(), time=t, name=self.name)

class FieldLogger:
    def __init__(self, comm, vis_dict):
        self.comm = comm
        self.vis_dict = vis_dict

        self.__source_id = None
        self.__enabled_fields = []
        self.__wave_data = []
    
    def add_field(self, key, name, callback):
        self.__wave_data.append((key, name, callback))
    
    def start_logging(self, source_id):
        if self.__source_id != None:
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
    
    def stop_logging(self):
        self.__source_id = None
    
    def log(self, t):
        for field in self.__enabled_fields:
            field.write(t)