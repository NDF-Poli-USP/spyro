from firedrake import (assemble, Constant, curl, DirichletBC, div, Function,
                       FunctionSpace, project, VectorFunctionSpace)
from ...domains.space import create_function_space
from .forms import (isotropic_elastic_without_pml, isotropic_elastic_with_pml,
                    anisotropic_elastic_without_pml)
from .functionals import mechanical_energy_form, mechanical_energy_form_elastic
from ..wave import Wave
from ...utils.typing import override, WaveType
from .anisotropy import AnisotropyTensor


class ElasticWave(Wave):
    """Base class for elastic wave propagators."""

    def __init__(self, dictionary, anisotropy="ISO", comm=None):
        """Wave Elastic object solver.

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the Wave class.
            Default is None
        anisotropy : `str`, optional
            The type of anisotropy in the medium. Oprions: "ISO, "VTI" or "TTI"  
        comm : `object`, optional
            MPI communicator for parallel execution. Default is None

        Returns
        -------
        None
        """

        # Type of wave equation to solve
        if anisotropy == "ISO":
            wave_type = WaveType.ISOTROPIC_ELASTIC
        elif anisotropy == "VTI":
            wave_type = WaveType.ANISOTROPIC_VTI_ELASTIC
        elif anisotropy == "TTI":
            wave_type = WaveType.ANISOTROPIC_TTI_ELASTIC

        super().__init__(dictionary, wave_type=wave_type, comm=comm)
        self.time = Constant(0)  # Time variable

        # State variables
        self.u_n = None   # Current displacement field
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
        self.field_logger.add_field("p-wave", "P-wave",
                                    lambda: self.update_p_wave())

        # Variables for logging the S-wave
        self.s_wave = None
        self.C_h = None
        self.field_logger.add_field("s-wave", "S-wave",
                                    lambda: self.update_s_wave())

        self.mechanical_energy = None
        self.field_logger.add_functional("mechanical_energy",
                                         lambda: assemble(self.mechanical_energy))

    @override
    def update_source_expression(self, t):
        self.time.assign(t)

    # Old abstract methods
    @override
    def _initialize_model_parameters(self):
        pass

    #  Obs: methods from IsotropicWave class that will be deprecated
    @override
    def _create_function_space(self):
        if self.wave_type != WaveType.ISOTROPIC_ELASTIC:
            self.property_space = \
                create_function_space(self.mesh, self.method, self.degree)
        return create_function_space(self.mesh, self.method, self.degree,
                                     dim=self.dimension)

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
    def get_forward_solution_receivers(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            data_with_halos = self.u_n.dat.data_ro_with_halos[:]
        return self.receivers.interpolate(data_with_halos)

    @override
    def get_function(self):
        return self.u_n

    @override
    def get_function_name(self):
        return "Displacement"

    @override
    def matrix_building(self):
        self.current_time = 0.0

        self.u_n = Function(self.function_space,
                            name=self.get_function_name())
        self.u_nm1 = Function(self.function_space,
                              name=self.get_function_name())
        self.u_np1 = Function(self.function_space,
                              name=self.get_function_name())

        abc_dict = self.input_dictionary.get("absorving_boundary_conditions", None)
        if abc_dict is not None:
            abc_active = abc_dict.get("status", False)
            if abc_active:
                dt_scheme = abc_dict.get("local", {}).get("dt_scheme", None)
                if dt_scheme == "backward_2nd":
                    self.u_nm2 = Function(self.function_space,
                                          name=self.get_function_name())

        if self.wave_type == WaveType.ISOTROPIC_ELASTIC:
            self.mechanical_energy = mechanical_energy_form(self)
        else:
            C_tensor = AnisotropyTensor.c_vti_tensor(self.PropISO, self.PropVTI)
            if self.wave_type == WaveType.ANISOTROPIC_TTI_ELASTIC:
                C_tensor = AnisotropyTensor.c_tti_tensor(C_tensor, self.PropTTI)
            self.C_tensor = C_tensor
            self.mechanical_energy = mechanical_energy_form_elastic(self)

        self.parse_initial_conditions()
        self.parse_boundary_conditions()
        self.parse_volumetric_forces()

        # TODO: Change variable names when refatoring
        if self.abc_boundary_layer_type is None or \
                self.abc_boundary_layer_type == "local":
            isotropic_elastic_without_pml(self) \
                if self.wave_type == WaveType.ISOTROPIC_ELASTIC else \
                anisotropic_elastic_without_pml(self)
        elif self.abc_boundary_layer_type == "PML":
            isotropic_elastic_with_pml(self)

    @override
    def rhs_no_pml(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.B

    def rhs_no_pml_source(self):
        if self.abc_boundary_layer_type == "PML":
            raise NotImplementedError
        else:
            return self.source_function

    def parse_initial_conditions(self):
        time_dict = self.input_dictionary["time_axis"]
        initial_condition = time_dict.get("initial_condition", None)
        if initial_condition is not None:
            x_vec = self.get_spatial_coordinates()
            self.u_n.interpolate(initial_condition(x_vec, 0 - self.dt))
            self.u_nm1.interpolate(initial_condition(x_vec, 0 - 2*self.dt))

    def parse_boundary_conditions(self):
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
                raise Exception(
                    f"Unsupported boundary condition with tag: {tag}")
            self.bcs.append(DirichletBC(subspace, value, idbc))

    def parse_volumetric_forces(self):
        acquisition_dict = self.input_dictionary["acquisition"]
        body_forces_data = acquisition_dict.get("body_forces", None)
        if body_forces_data is not None:
            x_vec = self.get_spatial_coordinates()
            self.body_forces = body_forces_data(x_vec, self.time)

    def update_p_wave(self):
        if self.p_wave is None:
            self.D_h = FunctionSpace(self.mesh, "DG", 0)
            self.p_wave = Function(self.D_h)

        self.p_wave.assign(project(div(self.get_function()), self.D_h))

        return self.p_wave

    def update_s_wave(self):
        if self.s_wave is None:
            if self.dimension == 2:
                self.C_h = FunctionSpace(self.mesh, "DG", 0)
            else:
                self.C_h = VectorFunctionSpace(self.mesh, "DG", 0)
            self.s_wave = Function(self.C_h)

        self.s_wave.assign(project(curl(self.get_function()), self.C_h))

        return self.s_wave

    def get_anisotropy_properties(self, iso_constants, vti_constants=None,
                                  tti_constants=None, anisotropy='weak'):

        vP, vS, rho = iso_constants
        self.PropISO = PropISO(self.property_space, vP=vP,
                               vS=vS, rho=rho)

        if self.wave_type != WaveType.ISOTROPIC_ELASTIC:
            epsilon, gamma, delta = vti_constants
            self.PropVTI = PropVTI(self.property_space, epsilon=epsilon,
                                   gamma=gamma, delta=delta, anisotropy=anisotropy)

        if self.wave_type == WaveType.ANISOTROPIC_TTI_ELASTIC:
            theta, phi = tti_constants
            self.PropTTI = PropTTI(self.property_space, theta=theta, phi=phi)


# ToDo: it should be in model_parameters or similar
class PropISO:
    """Isotropic properties.

    Attributes
    ----------
    vP: `Firedrake.Function`
        P-wave velocity [m/s]
    vS: `Firedrake.Function`
        S-wave velocity [m/s]
    rho: `Firedrake.Function`
        Density [kg/m³]
    """

    def __init__(self, W, vP=2500.0, vS=1200.0, rho=2200.0):
        self.vP = Function(W).assign(Constant(vP))
        self.vS = Function(W).assign(Constant(vS))
        self.rho = Function(W).assign(Constant(rho))


class PropVTI:
    """VTI properties.

    Attributes
    ----------
    epsilon: `Firedrake.Function`
        Thomsen parameter epsilon
    gamma: `Firedrake.Function`
        Thomsen parameter gamma
    delta: `Firedrake.Function`
        Thomsen parameter delta
    anisotropy: `str`
        Type of anisotropy: 'weak' or 'exact'
    """

    def __init__(self, W, epsilon=0.2, gamma=0.1, delta=0.15, anisotropy='weak'):
        self.epsilon = Function(W).assign(Constant(epsilon))
        self.gamma = Function(W).assign(Constant(gamma))
        self.delta = Function(W).assign(Constant(delta))
        self.anisotropy = anisotropy


class PropTTI:
    """TTI properties.

    Attributes
    ----------
    theta: `Firedrake.Function`
        Tilt angle in degrees
    phi: `Firedrake.Function`
        Azimuth angle in degrees (default is 0: 2D case)

    """

    def __init__(self, W, theta=30.0, phi=0.0):
        self.theta = Function(W).assign(Constant(theta))
        self.phi = Function(W).assign(Constant(phi))
