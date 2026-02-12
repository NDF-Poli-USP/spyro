# import firedrake as fire
# import firedrake.adjoint as fire_ad
# from .time_integration_ad import central_difference_acoustic
# from firedrake.__future__ import interpolate
# # Note this turns off non-fatal warnings
# fire.set_log_level(fire.ERROR)


# class ForwardSolver:
#     """Wave equation forward solver.

#     This forward solver is prepared to work with the automatic
#     differentiation. Only the acoustic wave equation is implemented.

#     Parameters
#     ----------
#     model : dict
#         Dictionary containing the model parameters.
#     mesh : Mesh
#         Firedrake mesh object.
#     """

#     def __init__(self, model, mesh, function_space):
#         self.model = model
#         self.mesh = mesh
#         self.V = function_space
#         self.receiver_mesh = fire.VertexOnlyMesh(
#             self.mesh, self.model["acquisition"]["receiver_locations"])
#         self.solution = None

#     def execute_acoustic(
#             self, c, source_number, wavelet, compute_functional=False,
#             true_data_receivers=None
#     ):
#         """Time-stepping acoustic forward solver.

#         The time integration is done using a central difference scheme.

#         Parameters
#         ----------
#         c : firedrake.Function
#             Velocity field.
#         source_number : int
#             Number of the source. This is used to select the source
#             location.
#         wavelet : list
#             Time-dependent wavelet.
#         compute_functional : bool, optional
#             Whether to compute the functional. If True, the true receiver
#             data must be provided.
#         true_data_receivers : list, optional
#             True receiver data. This is used to compute the functional.

#         Returns
#         -------
#         (receiver_data : list, J_val : float)
#             Receiver data and functional value.

#         Raises
#         ------
#         ValueError
#             If true_data_receivers is not provided when computing the
#             functional.
#         """
#         self.solution = None
#         # RHS
#         source_function = fire.Cofunction(self.V.dual())
#         solver, u_np1, u_n, u_nm1 = central_difference_acoustic(
#             self, c, source_function)
#         # Sources.
#         source_mesh = fire.VertexOnlyMesh(
#             self.mesh,
#             [self.model["acquisition"]["source_locations"][source_number]]
#         )
#         # Source function space.
#         V_s = fire.FunctionSpace(source_mesh, "DG", 0)
#         d_s = fire.Function(V_s)
#         d_s.assign(1.0)
#         source_d_s = fire.assemble(d_s * fire.TestFunction(V_s) * fire.dx)
#         # Interpolate from the source function space to the velocity function space.
#         q_s = fire.Cofunction(self.V.dual()).interpolate(source_d_s)

#         # Receivers
#         V_r = fire.FunctionSpace(self.receiver_mesh, "DG", 0)
#         # Interpolate object.
#         interpolate_receivers = interpolate(u_np1, V_r)

#         # Time execution.
#         J_val = 0.0
#         receiver_data = []
#         total_steps = int(self.model["time_axis"]["final_time"] / self.model["time_axis"]["dt"]) + 1
#         if (
#             fire_ad.get_working_tape()._checkpoint_manager
#             and self.model["aut_dif"]["checkpointing"]
#         ):
#             time_range = fire_ad.get_working_tape().timestepper(
#                 iter(range(total_steps)))
#         else:
#             time_range = range(total_steps)

#         for step in time_range:
#             source_function.assign(wavelet[step] * q_s)
#             solver.solve()
#             u_nm1.assign(u_n)
#             u_n.assign(u_np1)
#             rec_data = fire.assemble(interpolate_receivers)
#             receiver_data.append(rec_data)
#             if compute_functional:
#                 if not true_data_receivers:
#                     raise ValueError("True receiver data is required for"
#                                      "computing the functional.")
#                 misfit = rec_data - true_data_receivers[step]
#                 J_val += fire.assemble(0.5 * fire.inner(misfit, misfit) * fire.dx)
#         self.solution = u_np1
#         return receiver_data, J_val

#     def execute_elastic(self):
#         raise NotImplementedError("Elastic wave equation is not yet implemented"
#                                   "for the automatic differentiation based FWI.")

#     def _solver_parameters(self):
#         if self.model["options"]["variant"] == "lumped":
#             params = {"ksp_type": "preonly", "pc_type": "jacobi"}
#         elif self.model["options"]["variant"] == "equispaced":
#             params = {"ksp_type": "cg", "pc_type": "jacobi"}
#         else:
#             raise ValueError("method is not yet supported")

#         return params


# Class inherited from the Wave class
import firedrake as fire
import warnings
import os

from .acoustic_solver_construction_no_pml import construct_solver_or_matrix_no_pml
from .acoustic_wave import AcousticWave


class DifferentiableForwardSolver(AcousticWave):

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)

    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        self.current_time = 0.0

        abc_type = self.abc_boundary_layer_type

        # Just to document variables that will be overwritten
        self.trial_function = None
        self.u_nm1 = None
        self.u_n = None
        self.u_np1 = fire.Function(self.function_space)
        self.lhs = None
        self.solver = None
        self.rhs = None
        self.B = None

        if abc_type is None or abc_type == "local" or abc_type == "hybrid":
            construct_solver_or_matrix_no_pml(self)
        elif abc_type == "PML":
            raise NotImplementedError(
                "PML is not yet implemented for the automatic differentiation."
            )
        self.acoustic_energy = acoustic_energy(self)
