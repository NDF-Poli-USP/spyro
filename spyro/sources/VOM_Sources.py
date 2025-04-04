import firedrake as fire
from firedrake import dx
from spyro.sources import full_ricker_wavelet


class VOMSources():
    def __init__(self, wave_object):
        self.mesh = wave_object.mesh
        self.space = wave_object.function_space.sub(0)
        self.my_ensemble = wave_object.comm
        self.dimension = wave_object.dimension
        self.degree = wave_object.degree
        self.point_locations = wave_object.source_locations
        self.number_of_points = wave_object.number_of_sources
        self.amplitude = wave_object.amplitude

        self.current_source = None
        self.update_wavelet(wave_object)

        self.reate_vom_sources_cofunction()
    
    def create_vom_sources_cofunction(self):
        source_mesh = fire.VertexOnlyMesh(self.mesh, self.point_locations)
        source_function_space = fire.FunctionSpace(source_mesh, "DG", 0)
        delta_function = fire.Function(source_function_space).interpolate(1.0)
        sources_cofunction = fire.Cofunction(self.space.dual()).interpolate(
            fire.assemble(delta_function) * fire.TestFunction(source_function_space) * dx
        )
        self.sources_cofunction = sources_cofunction
    
    def update_wavelet(self, wave_object):
        self.wavelet = full_ricker_wavelet(
            dt=wave_object.dt,
            final_time=wave_object.final_time,
            frequency=wave_object.frequency,
            delay=wave_object.delay,
            delay_type=wave_object.delay_type,
        )
    
    def apply_source(self, rhs_forcing, step):
        pass