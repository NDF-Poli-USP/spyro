from abc import abstractmethod, ABCMeta
from firedrake import Constant

from ..wave import Wave
from ...utils.typing import override


class ElasticWave(Wave, metaclass=ABCMeta):
    '''Base class for elastic wave propagators'''

    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)
        self.time = Constant(0)  # Time variable

    @override
    def _initialize_model_parameters(self):
        d = self.input_dictionary.get("synthetic_data", False)
        if bool(d) and "type" in d:
            if d["type"] == "object":
                self.initialize_model_parameters_from_object(d)
            elif d["type"] == "file":
                self.initialize_model_parameters_from_file(d)
            elif d["type"] == "conditional":
                self.initialize_model_parameters_with_expression(d)
            else:
                raise Exception(f"Invalid synthetic data type: {d['type']}")
        else:
            raise Exception("Input dictionary must contain ['synthetic_data']['type']")

    @override
    def initialize_model_parameters_with_expression(self, synthetic_data_dict: dict):
        """
        Inicializa parâmetros do modelo aceitando expressões Firedrake (ex: conditional)
        ou funções preguiçosas (lambda mesh: conditional(...)).
        Diferencia parâmetros escalares (rho, lambda, mu) de velocidades (c, c_s).
        """
        import ufl
        import numpy as np
        import firedrake as fire

        # --- Funções auxiliares ---
        def constant_wrapper(value):
            """Converte escalares em Constant, mantém objetos do Firedrake."""
            if np.isscalar(value):
                return fire.Constant(value)
            return value

        def get_value(key, default=None):
            """Obtém valor do dicionário, chama lambda se necessário, aplica wrapper."""
            val = synthetic_data_dict.get(key, default)
            if callable(val):
                try:
                    val = val(self.mesh)
                except TypeError:
                    raise TypeError(
                        f"Callable for key '{key}' must accept one argument (mesh)."
                    )
            return constant_wrapper(val)

        def handle_expression(expr, name="parameter", vector=False):
            """
            Interpola expressões UFL (ex: conditional) em Function,
            mantendo Functions e Constants existentes.
            Se vector=True, cria Function vetorial; caso contrário, escalar.
            """
            if expr is None:
                return None

            # Se já for Function, apenas retorna
            if isinstance(expr, fire.Function):
                return expr

            # Se for expressão simbólica UFL
            if isinstance(expr, ufl.core.expr.Expr):
                if vector:
                    # Para campos vetoriais (ex: deslocamento, velocidade vetorial)
                    if getattr(self, "dg_velocity_model", False):
                        V = fire.FunctionSpace(self.mesh, "DG", 0)
                    else:
                        V = getattr(self, "function_space", None)
                        if V is None:
                            V = fire.VectorFunctionSpace(self.mesh, "CG", 1)
                else:
                    # Para parâmetros escalares
                    V = fire.FunctionSpace(self.mesh, "CG", 1)

                f = fire.Function(V, name=name)
                f.interpolate(expr)
                return f

            # Se for Constant ou escalar
            if isinstance(expr, fire.Constant) or np.isscalar(expr):
                if vector:
                    # transformar escalar em vetor, replicando valores
                    return fire.as_vector([expr, expr])
                return fire.Constant(expr)

            raise TypeError(
                f"Unsupported type for {name}: {type(expr)}. "
                "Expected Constant, Function, UFL expression, or callable(mesh)."
            )

        # --- Obtém parâmetros escalares ---
        self.rho = handle_expression(get_value("density"), "density", vector=False)
        self.lmbda = handle_expression(
            get_value("lambda", default=get_value("lame_first")), "lambda", vector=False
        )
        self.mu = handle_expression(
            get_value("mu", get_value("lame_second")), "mu", vector=False
        )

        # --- Obtém velocidades (c, c_s) ---
        self.c = handle_expression(get_value("p_wave_velocity"), "p_wave_velocity", vector=False)
        self.c_s = handle_expression(get_value("s_wave_velocity"), "s_wave_velocity", vector=False)

        # --- Opções de consistência ---
        option_1 = bool(self.rho) and bool(self.lmbda) and bool(self.mu) \
            and not bool(self.c) and not bool(self.c_s)
        option_2 = bool(self.rho) and bool(self.c) and bool(self.c_s) \
            and not bool(self.lmbda) and not bool(self.mu)

        if option_1:
            self.c = ((self.lmbda + 2*self.mu) / self.rho)**0.5
            self.c_s = (self.mu / self.rho)**0.5
        elif option_2:
            self.mu = self.rho * self.c_s**2
            self.lmbda = self.rho * self.c**2 - 2*self.mu
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

        # --- Armazena o modelo inicial (para visualização, debug, etc.) ---
        if isinstance(self.c, fire.Function):
            self.initial_velocity_model = self.c.copy(deepcopy=True)

            
    @abstractmethod
    def initialize_model_parameters_from_object(self, synthetic_data_dict):
        pass

    @abstractmethod
    def initialize_model_parameters_from_file(self, synthetic_data_dict):
        pass

    @override
    def update_source_expression(self, t):
        self.time.assign(t)
