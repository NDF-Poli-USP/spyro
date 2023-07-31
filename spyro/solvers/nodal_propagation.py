# from .CG_acoustic import AcousticWave
# from ..receivers.Receivers import Receivers
# from ..sources.NodalSources import NodalSources
# from ..io import Model_parameters


# class AcousticNodalPropagation(AcousticWave):
#     def __init__(
#         self, model_parameters=None, comm=None, model_dictionary=None
#     ):
#         """Wave object solver. Contains both the forward solver
#         and gradient calculator methods.

#         Parameters:
#         -----------
#         comm: MPI communicator

#         model_parameters: Python object
#             Contains model parameters
#         """
#         if comm != None:
#             self.comm = comm
#         if model_parameters == None:
#             model_parameters = Model_parameters(
#                 dictionary=model_dictionary, comm=comm
#             )
#         if model_parameters.comm != None:
#             self.comm = model_parameters.comm
#         if self.comm != None:
#             self.comm.comm.barrier()
#         self.model_parameters = model_parameters
#         self.initial_velocity_model = None
#         self._unpack_parameters(model_parameters)
#         self.mesh = model_parameters.get_mesh()
#         self.function_space = None
#         self.current_time = 0.0
#         self.set_solver_parameters()

#         self._build_function_space()
#         self.sources = NodalSources(self)
#         self.receivers = Receivers(self)
#         self.wavelet = model_parameters.get_wavelet()
