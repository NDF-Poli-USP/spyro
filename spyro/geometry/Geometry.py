from firedrake import *

import spyro

class Geometry:

    def __init__(self, model, mesh, V, comm):
        """Class describing acquisition geometry.
    
        Parameters
        ----------
        model: `dictionary`
            Contains simulation parameters and options.
        mesh: a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V: Firedrake.FunctionSpace object
            The space of the finite elements
        comm: Firedrake.ensemble_communicator
            An ensemble communicator
        """

        self.V = V
        self.mesh = mesh
        self.my_ensemble = comm
        self.src_depth = model["acquisition"]["src_depth"]
        self.num_sources = model["acquisition"]["num_sources"]
        self.src_XMIN = model["acquisition"]["src_XMIN"]
        self.src_XMAX = model["acquisition"]["src_XMAX"]
        self.rec_depth = model["acquisition"]["rec_depth"]
        self.num_receivers = model["acquisition"]["num_receivers"]
        self.rec_XMIN = model["acquisition"]["rec_XMIN"]
        self.rec_XMAX = model["acquisition"]["rec_XMAX"]
        self.model = model

    def create_sources(self):
        """Create sources transect"""
    
        # self.model["acquisition"]["source_pos"] = spyro.create_receiver_transect(
        #     (self.src_depth, self.src_XMIN), (self.src_depth, self.src_XMAX), self.num_sources
        # )

        self.model["acquisition"]["source_pos"] = []
        for depth in self.src_depth:
            self.model["acquisition"]["source_pos"] += spyro.create_receiver_transect(
                (depth, self.src_XMIN), (depth, self.src_XMAX), self.num_sources
            ).tolist()
        self.model["acquisition"]["source_pos"] = np.array(self.model["acquisition"]["source_pos"])
        
        sources = spyro.Sources(self.model, self.mesh, self.V, self.my_ensemble).create()
    
        return sources
    
    def create_receivers(self):
        "Create receivers transect"""

        self.model["acquisition"]["receiver_locations"] = []
        for depth in self.rec_depth:
            self.model["acquisition"]["receiver_locations"] += spyro.create_receiver_transect(
                (depth, self.rec_XMIN), (depth, self.rec_XMAX), self.num_receivers
            ).tolist()
        self.model["acquisition"]["receiver_locations"] = np.array(self.model["acquisition"]["receiver_locations"])
    
        receivers = spyro.Receivers(self.model, self.mesh, self.V, self.my_ensemble).create()
    
        return receivers

    def create(self):
        """Create both sources and receivers"""
    
        sources = self.create_sources()
        receivers = self.create_receivers()
    
        return sources, receivers
