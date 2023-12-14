import firedrake as fire
import warnings

from .acoustic_wave import AcousticWave


class FullWaveformInversion(AcousticWave):
    def __init__(self, dictionary=None, comm=None):
        super().__init__(dictionary=dictionary, comm=comm)
        if self.running_fwi is False:
            warnings.warn("Dictionary FWI options set to not run FWI.")
        self.real_velocity_model = None
        self.real_velocity_model_file = None
        self.real_shot_record = None
        self.guess_shot_record = None
        self.gradient = None
        self.current_iteration = 0
        self.mesh_iteration = 0
        self.iteration_limit = 100
        self.inner_product = 'L2'

    def generate_real_shot_record(self):
        self.initial_velocity_model = self.real_velocity_model
        super().forward_solve()
        self.real_shot_record = self.forward_solution_receivers

    def set_smooth_guess_velocity_model(self, real_velocity_model_file=None):
        if real_velocity_model_file is not None:
            real_velocity_model_file = real_velocity_model_file
        else:
            real_velocity_model_file = self.real_velocity_model_file
        
        
        



