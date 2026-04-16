"""Isotropic elastic wave propagator solver.

This module implements the isotropic elastic wave equation solver,
handling density and Lame parameter conversions, displacement fields,
and wave decomposition into P-wave and S-wave components.
"""

import numpy as np

from firedrake import (
    assemble,
    Constant,
    curl,
    DirichletBC,
    div,
    Function,
    FunctionSpace,
    project,
    VectorFunctionSpace,
)

from .elastic_wave import ElasticWave
from .forms import isotropic_elastic_without_pml, isotropic_elastic_with_pml
from .functionals import mechanical_energy_form
from ...utils.typing import override, WaveType
from ...domains.space import create_function_space


class IsotropicWave(ElasticWave):
    """Isotropic elastic wave propagator."""

    def __init__(self, dictionary, comm=None):
        """Initialize the IsotropicWave."""
        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.ISOTROPIC_ELASTIC
        self.rho = None  # Density
        self.lmbda = None  # First Lame parameter
        self.mu = None  # Second Lame parameter
        self.c_s = None  # Secondary wave velocity

        self.u_n = None  # Current displacement field
        self.u_nm1 = None  # Displacement field in previous iteration
        self.u_nm2 = None  # Displacement field at iteration n-2
        self.u_np1 = None  # Displacement field in next iteration

        # Volumetric sources (defined through UFL)
        self.body_forces = None

        # Boundary conditions
        self.bcs = []

        # Variables for logging the P-wave
        self.p_wave = None
        self.D_h = None
        self.field_logger.add_field("p-wave", "P-wave", lambda: self.update_p_wave())

        # Variables for logging the S-wave
        self.s_wave = None
        self.C_h = None
        self.field_logger.add_field("s-wave", "S-wave", lambda: self.update_s_wave())

        self.mechanical_energy = None
        self.field_logger.add_functional(
            "mechanical_energy", lambda: assemble(self.mechanical_energy)
        )

    @override
    def initialize_model_parameters_from_object(self, synthetic_data_dict: dict):
        """Initialize model parameters from a dictionary.

        Parameters
        ----------
        synthetic_data_dict : dict
            Dictionary containing model parameters such as density,
            Lame parameters, or wave velocities.

        Raises
        ------
        Exception
            If the selection of parameters is inconsistent. Valid options are
            {Density, Lame first, Lame second} or {Density, P-wave velocity,
            S-wave velocity}.
        """

        def constant_wrapper(value):
            if np.isscalar(value):
                return Constant(value)
            else:
                return value

        def get_value(key, default=None):
            return constant_wrapper(synthetic_data_dict.get(key, default))

        self.rho = get_value("density")
        self.lmbda = get_value("lambda", default=get_value("lame_first"))
        self.mu = get_value("mu", get_value("lame_second"))
        self.c = get_value("p_wave_velocity")
        self.c_s = get_value("s_wave_velocity")

        # Check if {rho, lambda, mu} is set and {c, c_s} are not
        option_1 = (
            bool(self.rho)
            and bool(self.lmbda)
            and bool(self.mu)
            and not bool(self.c)
            and not bool(self.c_s)
        )
        # Check if {rho, c, c_s} is set and {lambda, mu} are not
        option_2 = (
            bool(self.rho)
            and bool(self.c)
            and bool(self.c_s)
            and not bool(self.lmbda)
            and not bool(self.mu)
        )

        if option_1:
            self.c = ((self.lmbda + 2 * self.mu) / self.rho) ** 0.5
            self.c_s = (self.mu / self.rho) ** 0.5
        elif option_2:
            self.mu = self.rho * self.c_s**2
            self.lmbda = self.rho * self.c**2 - 2 * self.mu
        else:
            raise Exception(
                f"Inconsistent selection of isotropic elastic wave parameters:\n"
                f"    Density        : {bool(self.rho)}\n"
                f"    Lame first     : {bool(self.lmbda)}\n"
                f"    Lame second    : {bool(self.mu)}\n"
                f"    P-wave velocity: {bool(self.c)}\n"
                f"    S-wave velocity: {bool(self.c_s)}\n"
                "The valid options are {Density, Lame first, Lame second} "
                "or (exclusive) {Density, P-wave velocity, S-wave velocity}"
            )

    @override
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        """Initialize model parameters from file.

        Parameters
        ----------
        synthetic_data_dict : dict
            Dictionary with file paths to model parameters.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError

    @override
    def _create_function_space(self):
        return create_function_space(
            self.mesh, self.method, self.degree, dim=self.dimension
        )

    @override
    def _set_vstate(self, vstate):
        self.u_n.assign(vstate)

    @override
    def _get_vstate(self):
        return self.u_n

    @override
    def _set_prev_vstate(self, vstate):
        if self.u_nm2 is not None:
            self.u_nm2.assign(self.u_nm1)
        self.u_nm1.assign(vstate)

    @override
    def _get_prev_vstate(self):
        return self.u_nm1

    @override
    def _set_next_vstate(self, vstate):
        self.u_np1.assign(vstate)

    @override
    def _get_next_vstate(self):
        return self.u_np1

    @override
    def get_receivers_output(self):
        """Obtain displacement data at receiver locations.

        Returns
        -------
        array
            Interpolated displacement values at receiver coordinates.

        Raises
        ------
        NotImplementedError
            If PML absorbing boundary conditions are active.
        """
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        """Return the displacement field.

        Returns
        -------
        Function
            The current displacement field.
        """
        return self.u_n

    @override
    def get_function_name(self):
        """Return the name of the primary solution field.

        Returns
        -------
        str
            The name "Displacement".
        """
        return "Displacement"

    @override
    def matrix_building(self):
        """Build the time marching matrix operator and initialize fields.

        Create displacement fields, set up absorbing boundary conditions,
        parse initial conditions, boundary conditions, and volumetric forces.
        Additionally, set up either standard or PML-based elastic wave equations.
        """
        self.current_time = 0.0

        self.u_n = Function(self.function_space, name=self.get_function_name())
        self.u_nm1 = Function(self.function_space, name=self.get_function_name())
        self.u_np1 = Function(self.function_space, name=self.get_function_name())

        abc_dict = self.input_dictionary.get("absorving_boundary_conditions", None)
        if abc_dict is not None:
            abc_active = abc_dict.get("status", False)
            if abc_active:
                dt_scheme = abc_dict.get("local", {}).get("dt_scheme", None)
                if dt_scheme == "backward_2nd":
                    self.u_nm2 = Function(
                        self.function_space, name=self.get_function_name()
                    )

        self.mechanical_energy = mechanical_energy_form(self)

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        if (
            self.abc_boundary_layer_type is None
            or self.abc_boundary_layer_type == "local"
        ):
            isotropic_elastic_without_pml(self)
        elif self.abc_boundary_layer_type == "PML":
            isotropic_elastic_with_pml(self)

    @override
    def rhs_no_pml(self):
        """Return the right-hand side for time integration.

        Returns
        -------
        UFL.CoFunction
            The right-hand side B for time stepping.

        Raises
        ------
        NotImplementedError
            If PML absorbing boundary conditions are active.
        """
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.B

    def rhs_no_pml_source(self):
        """Return the source term for the elastic wave equation.

        Returns
        -------
        Function
            The source function for the wave equation.

        Raises
        ------
        NotImplementedError
            If PML absorbing boundary conditions are active.
        """
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.source_function

    def parse_initial_conditions(self):
        """Parse and set initial displacement conditions from input dictionary.

        Interpolates initial conditions onto displacement fields u_n and u_nm1
        based on the specified initial_condition function.
        """
        time_dict = self.input_dictionary["time_axis"]
        initial_condition = time_dict.get("initial_condition", None)
        if initial_condition is not None:
            x_vec = self.get_spatial_coordinates()
            self.u_n.interpolate(initial_condition(x_vec, 0 - self.dt))
            self.u_nm1.interpolate(initial_condition(x_vec, 0 - 2 * self.dt))

    def parse_boundary_conditions(self):
        """Parse and apply Dirichlet boundary conditions.

        Processes boundary condition specifications from the input dictionary,
        supporting individual component constraints (ux, uy, uz) or full
        displacement vector constraints (u).

        Raises
        ------
        Exception
            If an unsupported boundary condition tag is encountered.
        """
        bc_list = self.input_dictionary.get("boundary_conditions", [])
        for tag, idbc, value in bc_list:
            if tag == "u":
                subspace = self.function_space
            elif tag == "uz":
                subspace = self.function_space.sub(0)
            elif tag == "ux":
                subspace = self.function_space.sub(1)
            elif tag == "uy":
                subspace = self.function_space.sub(2)
            else:
                raise Exception(f"Unsupported boundary condition with tag: {tag}")
            self.bcs.append(DirichletBC(subspace, value, idbc))

    def parse_volumetric_forces(self):
        """Parse body forces from acquisition dictionary.

        Extract body force data and initialize the body_forces field
        with the specified spatial and temporal variation.
        """
        acquisition_dict = self.input_dictionary["acquisition"]
        body_forces_data = acquisition_dict.get("body_forces", None)
        if body_forces_data is not None:
            x_vec = self.get_spatial_coordinates()
            self.body_forces = body_forces_data(x_vec, self.time)

    def update_p_wave(self):
        """Compute and return the P-wave component.

        Project the divergence of the displacement field onto a DG-0 function
        space to extract the P-wave component.

        Returns
        -------
        Firedrake.function
            The P-wave field as the divergence of displacement.
        """
        if self.p_wave is None:
            self.D_h = FunctionSpace(self.mesh, "DG", 0)
            self.p_wave = Function(self.D_h)

        self.p_wave.assign(project(div(self.get_function()), self.D_h))

        return self.p_wave

    def update_s_wave(self):
        """Compute and return the S-wave (curl) component.

        Project the curl of the displacement field onto a DG-0 function space
        to extract the S-wave component. In 2D, this returns a scalar field;
        in 3D, a vector field.

        Returns
        -------
        Firedrake.function
            The S-wave field as the curl of displacement.
        """
        if self.s_wave is None:
            if self.dimension == 2:
                self.C_h = FunctionSpace(self.mesh, "DG", 0)
            else:
                self.C_h = VectorFunctionSpace(self.mesh, "DG", 0)
            self.s_wave = Function(self.C_h)

        self.s_wave.assign(project(curl(self.get_function()), self.C_h))

        return self.s_wave
