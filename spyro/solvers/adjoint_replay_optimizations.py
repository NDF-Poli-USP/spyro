"""Replay optimizations for Spyro pyadjoint tapes."""

from __future__ import annotations

from types import MethodType

import firedrake as fire
from ufl.corealg.traversal import traverse_unique_terminals

from firedrake.adjoint_utils.blocks.solving import (
    NonlinearVariationalSolveBlock,
    Solver,
)
from firedrake.adjoint_utils.blocks.function import FunctionAssignBlock


SPYRO_ACOUSTIC_NO_PML_STEP_TAG = "spyro_acoustic_no_pml_step"
SPYRO_ACOUSTIC_NO_PML_NO_ABC_STEP_TAG = "spyro_acoustic_no_pml_no_abc_step"
SPYRO_ACOUSTIC_STEP_TAGS = {
    SPYRO_ACOUSTIC_NO_PML_STEP_TAG,
    SPYRO_ACOUSTIC_NO_PML_NO_ABC_STEP_TAG,
}


def _assign_map(assign_map):
    for coeff, value in assign_map.items():
        coeff.assign(value)


def _uses_nonzero_initial_guess(block):
    solver_parameters = block.forward_kwargs.get("solver_parameters", {})
    value = solver_parameters.get("ksp_initial_guess_nonzero", False)
    return value not in (False, 0, "0", "false", "False", None)


def _optimized_acoustic_no_pml_replace_forms(self, solver=Solver.FORWARD):
    """Replace replay coefficients with fewer redundant control copies.

    Firedrake's generic ``NonlinearVariationalSolveBlock`` copies every
    coefficient in both the residual and Jacobian before every recomputed
    timestep.  For Spyro's acoustic no-PML central-difference step the Jacobian
    coefficients are static over the replay, so they only need to be copied once
    per reduced-functional evaluation.  The time-varying residual coefficients
    are still copied for every timestep.
    """
    if solver != Solver.FORWARD:
        return NonlinearVariationalSolveBlock._ad_solver_replace_forms(
            self,
            solver,
        )

    problem = self._ad_solvers["forward_nlvs"]._problem
    state = self._ad_solvers.setdefault("spyro_replay_state", {})
    recompute_count = self._ad_solvers.get("recompute_count")

    if state.get("recompute_count") != recompute_count:
        state["recompute_count"] = recompute_count
        state["jacobian_coefficients_assigned"] = False
        state["jacobian_coefficients"] = frozenset()

    if not state["jacobian_coefficients_assigned"]:
        jacobian_map = self._ad_assign_map(problem.J, solver)
        _assign_map(jacobian_map)
        state["jacobian_coefficients"] = frozenset(jacobian_map)
        state["jacobian_coefficients_assigned"] = True

    residual_map = self._ad_assign_map(problem.F, solver)
    jacobian_coefficients = state["jacobian_coefficients"]
    skip_initial_guess = (
        problem.is_linear
        and problem._constant_jacobian
        and not _uses_nonzero_initial_guess(self)
        and not state.get("force_solution_coefficient", False)
    )
    solution_count = self.func.count()
    dynamic_residual_map = {
        coeff: value
        for coeff, value in residual_map.items()
        if coeff not in jacobian_coefficients
        and not (
            skip_initial_guess
            and problem._ad_count_map.get(coeff) == solution_count
        )
    }
    _assign_map(dynamic_residual_map)


def _optimized_acoustic_no_pml_prepare_recompute_component(
    self,
    inputs,
    relevant_outputs,
):
    """Avoid rebuilding unused UFL forms during solve-block recompute."""
    return None, None, self._create_initial_guess(), ()


def _coefficient_name(coefficient):
    name = getattr(coefficient, "name", None)
    if name is None:
        return None
    return name()


def _saved_dependency_by_name(block, inputs, name):
    for dependency, value in zip(block.get_dependencies(), inputs):
        if _coefficient_name(dependency.output) == name:
            return value
    return None


def _cell_measure(form):
    if hasattr(form, "components"):
        form = form.components()[0]
    for integral in form.integrals():
        if integral.integral_type() == "cell":
            return fire.Measure(
                "dx",
                domain=integral.ufl_domain(),
                subdomain_id=integral.subdomain_id(),
                metadata=integral.metadata(),
            )
    return fire.dx(domain=form.ufl_domain())


def _infer_dt_from_form(form):
    forms = form.components() if hasattr(form, "components") else [form]
    for component in forms:
        if not hasattr(component, "integrals"):
            continue
        for integral in component.integrals():
            for terminal in traverse_unique_terminals(integral.integrand()):
                if isinstance(terminal, fire.Constant):
                    value = float(terminal)
                    if value > 0.0:
                        return value**0.5
    return None


def _optimized_acoustic_no_pml_prepare_evaluate_adj(
    self,
    inputs,
    adj_inputs,
    relevant_dependencies,
):
    if self._should_compute_boundary_adjoint(relevant_dependencies):
        return NonlinearVariationalSolveBlock.prepare_evaluate_adj(
            self,
            inputs,
            adj_inputs,
            relevant_dependencies,
        )

    velocity = _saved_dependency_by_name(self, inputs, "velocity")
    operators = _cached_acoustic_no_pml_operators(self)
    adj_sol = _optimized_acoustic_no_pml_adjoint_solve(
        self,
        adj_inputs[0],
        velocity,
        operators,
    )
    adj_sol_bdy = None
    if adj_sol is None:
        adj_sol, adj_sol_bdy = self._adjoint_solve(adj_inputs[0], False)
        self.adj_sol = adj_sol
    if self.adj_cb is not None:
        self.adj_cb(adj_sol)

    return {
        "adj_sol": self.adj_state,
        "adj_sol_bdy": adj_sol_bdy,
        "velocity": velocity,
        "pressure": _saved_dependency_by_name(self, inputs, "pressure"),
        "pressure_t_dt": _saved_dependency_by_name(
            self,
            inputs,
            "pressure t-dt",
        ),
        "pressure_t_plus_dt": self.get_outputs()[0].saved_output,
        "operators": operators,
    }


def _cached_acoustic_no_pml_operators(block):
    cache = block._ad_solvers.setdefault("spyro_adjoint_operator_cache", {})
    if "acoustic_no_pml" in cache:
        return cache["acoustic_no_pml"]

    velocity = next(
        (
            dependency.output
            for dependency in block.get_dependencies()
            if _coefficient_name(dependency.output) == "velocity"
        ),
        None,
    )
    if velocity is None:
        return None

    dt = block.solver_kwargs.get("appctx", {}).get("dt")
    if dt is None:
        dt = _infer_dt_from_form(block.lhs)
    if dt is None:
        return None

    V = velocity.function_space()
    trial = fire.TrialFunction(V)
    test = fire.TestFunction(V)
    dx = _cell_measure(block.lhs)
    stiffness_matrix = fire.assemble(
        fire.dot(fire.grad(trial), fire.grad(test)) * dx,
        mat_type="aij",
    )
    mass_weights = fire.assemble(test * dx)

    cache["acoustic_no_pml"] = {
        "inv_dt2": 1.0 / fire.Constant(float(dt) ** 2),
        "inv_dt2_float": 1.0 / float(dt) ** 2,
        "dx": dx,
        "stiffness_matrix": stiffness_matrix,
        "stiffness_tensor": fire.Cofunction(V.dual()),
        "mass_weights": mass_weights,
    }
    return cache["acoustic_no_pml"]


def _optimized_acoustic_no_pml_adjoint_solve(
    block,
    dJdu,
    velocity,
    operators,
):
    """Solve the no-PML adjoint mass equation by its lumped diagonal.

    The no-ABC acoustic step has
    ``dF/du = M(c^{-2}) / dt^2``.  With Spyro's lumped element variant this is
    diagonal, so the adjoint solve is a pointwise scaling of the dual RHS.
    """
    if block.bcs or velocity is None or operators is None:
        return None

    mass_weights = operators["mass_weights"].dat.data_ro
    if mass_weights.min() <= 0.0:
        return None

    if block.adj_state is None:
        block.adj_state = fire.Function(block.function_space)
    block.adj_state.dat.data[:] = (
        dJdu.dat.data_ro
        * velocity.dat.data_ro ** 2
        / (mass_weights * operators["inv_dt2_float"])
    )
    return block.adj_state


def _optimized_acoustic_no_pml_evaluate_adj_component(
    self,
    inputs,
    adj_inputs,
    block_variable,
    idx,
    prepared=None,
):
    coefficient = block_variable.output
    coefficient_name = _coefficient_name(coefficient)

    if coefficient_name == "pressure t+dt":
        return None

    if coefficient_name == "pressure t-dt":
        adjoint = fire.Cofunction(coefficient.function_space().dual())
        adjoint.dat.data[:] = -adj_inputs[0].dat.data_ro
        return adjoint

    adj_sol = prepared["adj_sol"]
    c = prepared["velocity"]
    u_nm1 = prepared["pressure_t_dt"]
    u_n = prepared["pressure"]
    u_np1 = prepared["pressure_t_plus_dt"]
    operators = prepared["operators"]

    if (
        c is None
        or u_nm1 is None
        or u_n is None
        or u_np1 is None
        or operators is None
    ):
        return NonlinearVariationalSolveBlock.evaluate_adj_component(
            self,
            inputs,
            adj_inputs,
            block_variable,
            idx,
            {
                "form": self._create_F_form(),
                "adj_sol": adj_sol,
                "adj_sol_bdy": prepared["adj_sol_bdy"],
            },
        )

    if coefficient_name == "velocity":
        adjoint = fire.Cofunction(coefficient.function_space().dual())
        adjoint.dat.data[:] = (
            2.0
            * c.dat.data_ro ** (-3)
            * (
                u_np1.dat.data_ro
                - 2.0 * u_n.dat.data_ro
                + u_nm1.dat.data_ro
            )
            * operators["inv_dt2_float"]
            * adj_sol.dat.data_ro
            * operators["mass_weights"].dat.data_ro
        )
        return adjoint

    if coefficient_name == "pressure":
        stiffness_tensor = operators["stiffness_tensor"]
        with adj_sol.dat.vec_ro as adj_vec, stiffness_tensor.dat.vec_wo as out_vec:
            operators["stiffness_matrix"].petscmat.mult(adj_vec, out_vec)
        adjoint = fire.Cofunction(coefficient.function_space().dual())
        adjoint.dat.data[:] = (
            2.0 * adj_inputs[0].dat.data_ro
            - stiffness_tensor.dat.data_ro
        )
        return adjoint

    return None


def _is_same_space_function_assignment(block):
    if block.expr is not None:
        return False
    dependencies = block.get_dependencies()
    outputs = block.get_outputs()
    if len(dependencies) != 1 or len(outputs) != 1:
        return False

    dependency = dependencies[0].output
    output = outputs[0].output
    if not isinstance(dependency, fire.Function):
        return False
    if not isinstance(output, fire.Function):
        return False
    return dependency.function_space() == output.function_space()


def _optimized_assign_prepare_evaluate_adj(
    self,
    inputs,
    adj_inputs,
    relevant_dependencies,
):
    if not _is_same_space_function_assignment(self):
        return FunctionAssignBlock.prepare_evaluate_adj(
            self,
            inputs,
            adj_inputs,
            relevant_dependencies,
        )
    return adj_inputs[0]


def _optimized_assign_evaluate_adj_component(
    self,
    inputs,
    adj_inputs,
    block_variable,
    idx,
    prepared=None,
):
    if not _is_same_space_function_assignment(self):
        return FunctionAssignBlock.evaluate_adj_component(
            self,
            inputs,
            adj_inputs,
            block_variable,
            idx,
            prepared,
        )
    if isinstance(prepared, fire.Cofunction):
        return prepared
    return FunctionAssignBlock.evaluate_adj_component(
        self,
        inputs,
        adj_inputs,
        block_variable,
        idx,
        prepared,
    )


def _can_accumulate_dat_directly(lhs, rhs):
    if not isinstance(lhs, (fire.Function, fire.Cofunction)):
        return False
    if not isinstance(rhs, type(lhs)):
        return False
    if lhs.function_space() != rhs.function_space():
        return False
    return lhs.function_space().mesh().comm.size == 1


def _optimized_add_adj_output(self, val):
    if self.adj_value is None:
        self.adj_value = val
    elif _can_accumulate_dat_directly(self.adj_value, val):
        self.adj_value.dat.data[:] += val.dat.data_ro
    else:
        self.adj_value += val


def _install_fast_adjoint_accumulator(block_variable):
    if getattr(block_variable, "_spyro_fast_adj_accumulator", False):
        return False
    block_variable.add_adj_output = MethodType(
        _optimized_add_adj_output,
        block_variable,
    )
    block_variable._spyro_fast_adj_accumulator = True
    return True


def optimize_acoustic_no_pml_replay(tape):
    """Install replay and adjoint optimizations on tagged Spyro solve blocks."""
    optimized = 0
    for block in tape.get_blocks():
        for block_variable in block.get_dependencies() + block.get_outputs():
            if _install_fast_adjoint_accumulator(block_variable):
                optimized += 1

        if isinstance(block, FunctionAssignBlock):
            if not _is_same_space_function_assignment(block):
                continue
            if getattr(block, "_spyro_assign_optimized", False):
                continue

            block.prepare_evaluate_adj = MethodType(
                _optimized_assign_prepare_evaluate_adj,
                block,
            )
            block.evaluate_adj_component = MethodType(
                _optimized_assign_evaluate_adj_component,
                block,
            )
            block._spyro_assign_optimized = True
            optimized += 1
            continue

        if not isinstance(block, NonlinearVariationalSolveBlock):
            continue
        if block.tag not in SPYRO_ACOUSTIC_STEP_TAGS:
            continue
        if getattr(block, "_spyro_replay_optimized", False):
            continue

        block._ad_solver_replace_forms = MethodType(
            _optimized_acoustic_no_pml_replace_forms,
            block,
        )
        block.prepare_recompute_component = MethodType(
            _optimized_acoustic_no_pml_prepare_recompute_component,
            block,
        )
        if block.tag == SPYRO_ACOUSTIC_NO_PML_NO_ABC_STEP_TAG:
            _cached_acoustic_no_pml_operators(block)
            block.prepare_evaluate_adj = MethodType(
                _optimized_acoustic_no_pml_prepare_evaluate_adj,
                block,
            )
            block.evaluate_adj_component = MethodType(
                _optimized_acoustic_no_pml_evaluate_adj_component,
                block,
            )
        block._spyro_replay_optimized = True
        optimized += 1
    return optimized
