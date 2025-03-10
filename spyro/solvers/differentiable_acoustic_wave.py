import logging

import firedrake as fire
import firedrake.adjoint as fire_ad
from .time_integration_ad import central_difference_acoustic
from firedrake.__future__ import interpolate
from ..sources import full_ricker_wavelet
# Note this turns off non-fatal warnings
fire.set_log_level(fire.ERROR)


class DifferentiableWaveEquation:
    """Wave equation forward solver.

    This forward solver is prepared to work with the automatic
    differentiation. Only the acoustic wave equation is implemented.

    Parameters
    ----------
    model : dict
        Dictionary containing the model parameters.
    mesh : Mesh
        Firedrake mesh object used for the simulation. 
    """

    def __init__(self, model, mesh):
        self.model = model
        self.mesh = mesh
        self._function_space = self._build_function_space()
        self._solution = None
        self._receiver_data = None
        self._functional_value = None
        # Receiver mesh.
        self._rec_mesh = self._set_receiver_mesh(self.mesh, self.model)

    def acoustic_solver(
            self, c, source_number, wavelet=None,
            compute_functional=False, true_data_receivers=None
    ):
        """Time-stepping acoustic forward solver.

        The time integration is done using a central difference scheme.

        Parameters
        ----------
        c : firedrake.Function
            Velocity field.
        source_number : int
            Number of the source. This is used to select the source
            location.
        wavelet : list
            Time-dependent wavelet.
        compute_functional : bool, optional
            Whether to compute the functional. If True, the true receiver
            data must be provided.
        true_data_receivers : list, optional
            True receiver data. This is used to compute the functional.

        Returns
        -------
        (receiver_data : list, J_val : float)
            Receiver data and functional value.

        Raises
        ------
        ValueError
            If true_data_receivers is not provided when computing the
            functional.
        """
        if wavelet is None:
            # Log information.
            logging.info("Wavelet not provided. Using Ricker wavelet.")
            wavelet = full_ricker_wavelet(
                self.model["timeaxis"]["dt"],
                self.model["timeaxis"]["tf"],
                self.model["acquisition"]["frequency_peak"]
            )
        # RHS
        source_function = fire.Cofunction(self._function_space.dual())
        solver, u_np1, u_n, u_nm1 = central_difference_acoustic(
            self, c, source_function)

        q_s = self._source(source_number)
        interpolate_receivers = self._interpolate_receivers(u_np1)
        
        # Time execution.
        if compute_functional:
            self._functional_value = 0.0
        total_steps = int(self.model["timeaxis"]["tf"] / self.model["timeaxis"]["dt"])
        if (
            fire_ad.get_working_tape()._checkpoint_manager
            and self.model["aut_dif"]["checkpointing"]
        ):
            time_range = fire_ad.get_working_tape().timestepper(
                iter(range(total_steps)))
        else:
            time_range = range(total_steps)

        self._receiver_data = []
        for step in time_range:
            source_function.assign(wavelet[step] * q_s)
            solver.solve()
            u_nm1.assign(u_n)
            u_n.assign(u_np1)
            rec_data = fire.assemble(interpolate_receivers)
            self._receiver_data.append(rec_data)
            if compute_functional:
                if not true_data_receivers:
                    raise ValueError("True receiver data is required for"
                                     "computing the functional.")
                misfit = rec_data - true_data_receivers[step]
                self._functional_value += fire.assemble(
                    0.5 * fire.inner(misfit, misfit) * fire.dx)

        self._solution = u_np1

    def elastic_solver(self):
        raise NotImplementedError("Elastic wave equation is not yet implemented"
                                  "for the automatic differentiation based FWI.")

    @property
    def receiver_data(self):
        return self._receiver_data

    @property
    def solution(self):
        return self._solution

    @property
    def function_space(self):
        return self._function_space

    @property
    def functional_value(self):
        return self._functional_value

    def set_mesh(self, mesh):
        self.mesh = mesh
        self._function_space = fire.FunctionSpace(
            self.mesh, self.model["opts"]["method"],
            degree=self.model["opts"]["degree"],
            variant=self.model["opts"]["quadrature"]
        )

    def _source(self, source_number):
        # Sources.
        source_mesh = self._set_source_mesh(source_number)

        # Source function space.
        V_s = fire.FunctionSpace(source_mesh, "DG", 0)
        d_s = fire.Function(V_s)
        d_s.assign(1.0)
        source_d_s = fire.assemble(d_s * fire.TestFunction(V_s) * fire.dx)
        # Interpolate from the source function space to the velocity function space.
        return fire.Cofunction(
            self._function_space.dual()).interpolate(source_d_s)

    def _interpolate_receivers(self, u_field):
        return interpolate(u_field,
                           fire.FunctionSpace(self._rec_mesh, "DG", 0))

    def _set_receiver_mesh(self, mesh, model):
        return fire.VertexOnlyMesh(
            mesh, model["acquisition"]["receiver_locations"]
        )

    def _set_source_mesh(self, source_number):
        return fire.VertexOnlyMesh(
            self.mesh,
            [self.model["acquisition"]["source_pos"][source_number]]
        )

    def _build_function_space(self):
        return fire.FunctionSpace(
            self.mesh, self.model["opts"]["method"],
            self.model["opts"]["degree"]
            )


    def _solver_parameters(self):
        if self.model["opts"]["method"] == "KMV":
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        elif (
            self.model["opts"]["method"] == "CG"
            and self.mesh.ufl_cell() != quadrilateral  # noqa: F821
            and self.mesh.ufl_cell() != hexahedron  # noqa: F821
        ):
            params = {"ksp_type": "cg", "pc_type": "jacobi"}
        elif self.model["opts"]["method"] == "CG" and (
            self.mesh.ufl_cell() == quadrilateral  # noqa: F821
            or self.mesh.ufl_cell() == hexahedron  # noqa: F821
        ):
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        else:
            raise ValueError("method is not yet supported")

        return params
