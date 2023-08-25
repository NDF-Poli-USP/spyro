import firedrake as fire

from .acousticNoPML import AcousticWaveNoPML
from .acousticPML import AcousticWavePML
from .mms_acoustic import AcousticWaveMMS
from .wave import Wave
from .time_integration import time_integrator
from ..io.basicio import ensemble_propagator
from ..domains.quadrature import quadrature_rules
from .acoustic_solver_construction_no_pml import construct_solver_or_matrix_no_pml
from .acoustic_solver_construction_with_pml import construct_solver_or_matrix_with_pml


def get_abc_type(dictionary):
    if "BCs" in dictionary:
        return "PML"
    elif "absorving_boundary_conditions" in dictionary:
        return dictionary["absorving_boundary_conditions"]["damping_type"]
    else:
        return None


class NewAcousticWave(Wave):
    def forward_solve(self):
        """Solves the forward problem.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self._get_initial_velocity_model()
        self.c = self.initial_velocity_model
        self.matrix_building()
        self.wave_propagator()

    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        self.current_time = 0.0
        quad_rule, k_rule, s_rule = quadrature_rules(self.function_space)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule

        abc_type = self.abc_boundary_layer_type

        # Just to document variables that will be overwritten
        self.trial_function = None
        self.u_nm1 = None
        self.u_n = None
        self.lhs = None
        self.solver = None
        self.rhs = None
        self.B = None
        if abc_type is None:
            construct_solver_or_matrix_no_pml(self)
        elif abc_type == "PML":
            V = self.function_space
            Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
            self.vector_function_space = Z
            self.X = None
            self.X_n = None
            self.X_nm1 = None
            construct_solver_or_matrix_with_pml(self)

    @ensemble_propagator
    def wave_propagator(self, dt=None, final_time=None, source_num=0):
        """Propagates the wave forward in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the wave object.
        final_time: Python 'float' (optional)
            Time which simulation ends. If not mentioned uses the default,
            that was estabilished in the wave object.

        Returns:
        --------
        usol: Firedrake 'Function'
            Pressure wavefield at the final time.
        u_rec: numpy array
            Pressure wavefield at the receivers across the timesteps.
        """
        if final_time is not None:
            self.final_time = final_time
        if dt is not None:
            self.dt = dt

        usol, usol_recv = time_integrator(self, source_id=source_num)

        return usol, usol_recv


def AcousticWave(dictionary=None):
    if dictionary["acquisition"]["source_type"] == "MMS":
        return AcousticWaveMMS(dictionary=dictionary)

    has_abc = False
    if "BCs" in dictionary:
        has_abc = dictionary["BCs"]["status"]
    elif "absorving_boundary_conditions" in dictionary:
        has_abc = dictionary["absorving_boundary_conditions"]["status"]

    if has_abc:
        abc_type = get_abc_type(dictionary)
    else:
        abc_type = None

    if has_abc is False:
        return AcousticWaveNoPML(dictionary=dictionary)
    elif has_abc and abc_type == "PML":
        return AcousticWavePML(dictionary=dictionary)
    elif has_abc and abc_type == "HABC":
        raise NotImplementedError("HABC not implemented yet")
