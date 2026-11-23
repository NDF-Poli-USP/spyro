import firedrake as fire
import numpy as np
from sys import float_info, exit
from ...domains.quadrature import quadrature_rules
from ...utils.error_management import (
    sanitize_num_array,
    validate_firedrake_parameter,
    validate_numeric,
    validate_parameter,
)
from functools import reduce
from ...tools.habc_tools import point_cloud_field
from ...utils.utils import global_max, global_min
from ...version_control import is_firedrake_new

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class Dir_Point_BC(fire.DirichletBC):
    """Class for Eikonal boundary conditions at a point.

    Attributes
    ----------
    nodes : `array`
        Points where the boundary condition is to be applied.
    """

    def __init__(self, V, value, nodes):
        """
        Initialize the Dir_Point_BC class

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space where the boundary condition is applied.
        value : `FiredrakeConstant`
            Value of the boundary condition.
        nodes : `array`
            Points where the boundary condition is to be applied.

        Returns
        -------
        None
        """

        # Calling superclass init and providing a dummy subdomain id
        super(Dir_Point_BC, self).__init__(V, value, 0)

        # Overriding the "nodes" property
        self.nodes = nodes


class Eikonal_Modeling:
    """Class for the Eikonal equation for Linear and Nonlinear analysis.

    Attributes
    ----------
    dimension : `int`
        The spatial dimension of the problem.
    ele_type_eik : `string`, optional
        Finite element type. 'consistent' or 'underintegrated'. Default is 'consistent'.
    f_est : `float`
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03.
    degree_eik : `int`
        Finite element order.
    source_locations: `list`of `tuples`
        Source locations as tuples of coordinates.
    tol : `float`, optional
        User solver tolerance. Default is 1e-16.
    source_bc : `str`
        Dirichlet condition at the sources: 'nearest_node' or 'local_time'.

    Methods
    -------
    nonlinear_eik()
        Assemble the Nonlinear Eikonal with stabilizing term.
    define_int_dom()
        Define the integration domain for the Eikonal equation.
    eikonal_bcs()
        Impose Dirichlet BCs for eikonal equation.
    eikonal_solver()
        Solve the Eikonal equation for model without absorbing layer.
    initial_guess()
        Provide an initial guess for the Eikonal solver.
    linear_eik()
        Assemble the linear Eikonal.
    local_travel_time()
        Local travel time from a source, prescribed by 'local_time'.
    linear_solution()
        Solve the linear Eikonal equation.
    nonlinear_solution()
        Solve the nonlinear Eikonal equation.
    solver_opts()
        Set the eikonal solver parameters.
    """

    def __init__(
        self,
        dimension,
        source_locations,
        ele_type_eik="consistent",
        degree_eik=None,
        f_est=0.03,
        tol=1e-16,
        source_bc="nearest_node",
    ):
        """Initialize the Eikonal_Modeling class.

        Parameters
        ----------
        dimension : `int`
            The spatial dimension of the problem.
        source_locations: `list`of `tuples`
            List of tuples containing all source locations.
        ele_type_eik : `string`, optional
            Finite element type. Options: 'consistent' or 'underintegrated'.
            Default is 'consistent'.
        degree_eik : `int`, optional
            Finite element order for the Eikonal analysis. Default is None.
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.03.
        tol : `float`, optional
            User solver tolerance. Default is 1e-16.
        source_bc : `str`, optional
            Dirichlet condition at the sources (see `eikonal_bcs`). Options:
            'nearest_node' (zero at the nearest node) or 'local_time' (local
            travel time at every node within one cell of the source). Default
            is 'nearest_node'.

        Returns
        -------
        None
        """

        # Dimension of the problem
        self.dimension = dimension

        # Source locations
        self.source_locations = source_locations

        # Finite element type.
        allowed_ele_types = ["consistent", "underintegrated"]
        self.ele_type_eik = validate_parameter(
            "ele_type_eik", ele_type_eik, allowed_ele_types
        )

        # Finite element order for the Eikonal analysis
        self.degree_eik = (
            degree_eik
            if degree_eik is not None
            else (2 if self.dimension == 2 else 1)
            if self.ele_type_eik == "consistent"
            else (4 if self.dimension == 2 else 3)
        )

        # Factor for the stabilizing term in Eikonal equation
        self.f_est = f_est

        # User solver tolerance
        self.tol = tol

        # Dirichlet condition at the sources
        self.source_bc = validate_parameter(
            "source_bc", source_bc, ["nearest_node", "local_time"]
        )

    def eikonal_bcs(
        self,
        V: fire.FunctionSpace,
        lmin: float,
        c: fire.Function | None = None,
        c_min: float | None = None,
    ) -> tuple[list[Dir_Point_BC], fire.Function]:
        """Impose Dirichlet BCs for eikonal equation.

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space where the boundary condition is applied.
        lmin : `float`
            Minimum mesh size. It scales the tie tolerance between nodes and
            is the radius of the source region for 'local_time'.
        c : `Firedrake.Function`, optional
            Velocity model on the mesh of ``V``. Required for 'local_time',
            ignored for 'nearest_node'. Default is `None`.
        c_min : `float`, optional
            Lower bound for the velocity in the local travel time, as in
            `eikonal_solver`. Default is `None` (no bound).

        Returns
        -------
        bcs_eik : `list`
            Dirichlet BCs for eikonal.
        sou_marker : `Firedrake.Function`
            Function marking the source locations in the mesh.

        Raises
        ------
        TypeError
            If ``V`` is not a function space, or ``c`` is not a
            `Firedrake.Function` (``c`` may be `None` only for 'nearest_node').
        ValueError
            If ``lmin`` or ``c_min`` is not positive, or a source has no node
            of ``V`` within the largest cell diameter of the mesh, which means
            it lies outside the mesh.

        Notes
        -----
        With 'nearest_node', each source is snapped to its nearest node of
        ``V`` (all nodes at that distance if there is a tie) and the Eikonal
        is zero there. Nodes tie when their distances differ by less than
        ``1e-8 * lmin``, since round-off makes the distances of equidistant
        nodes differ in the last digits.

        With 'local_time', every node within ``lmin`` of a source (and at
        least its nearest nodes) takes the local travel time from the source,
        see `local_travel_time`, and the smallest one when several sources
        reach it. The zero condition on a patch of nodes around the source
        makes the travel time too short by up to the patch size over the
        velocity (first-order error in the node spacing), and tied nodes off
        the ray from the source make it too long. Prescribing the local
        travel time on a ball reduces this to second order, as in the source
        initialisation of fast-marching methods. The ball should not reach
        an absorbing boundary, otherwise the minimum Eikonal there is set by
        the initialisation and not by the solve.

        Parallel safety: the minimum distance is a collective PETSc
        reduction, the source velocity is evaluated on a vertex-only mesh and
        reduced over all ranks (as in `Minimum_Eikonal.ident_eik_on_bnd`), so
        every rank builds the same local travel time. The node selection
        reads the halos, so every rank holds the source nodes it can see, as
        ``DirichletBC.nodes`` does, and the prescribed values are an
        interpolated `Firedrake.Function` whose halo copies agree.

        The nodes of ``V`` include the cell vertices, and every point of a cell
        is within the cell diameter of its vertices. The largest cell diameter
        is therefore a bound on the distance from any source inside the mesh
        to its nearest node, whatever the element degree or the node layout
        (equispaced, Gauss-Lobatto-Legendre or KMV). A bound derived from the
        nodal spacing, such as ``lmin / (degree_eik + 1)``, rejects sources
        that lie between nodes: the spacing of Gauss-Lobatto-Legendre nodes is
        not uniform, and the distance to the nearest node grows with the
        square root of the dimension.
        """
        local_time = self.source_bc == "local_time"
        validate_firedrake_parameter("V", V, "FunctionSpace")
        validate_numeric("lmin", lmin, lower_bound=0.0)
        validate_firedrake_parameter(
            "c", c, "Function", accept_parameter_as_none=not local_time
        )
        validate_numeric("c_min", c_min, accept_parameter_as_none=True, lower_bound=0.0)

        mesh = V.mesh()
        coords = fire.SpatialCoordinate(mesh)
        dist2 = fire.Function(V)
        cell_diam = fire.Function(fire.FunctionSpace(mesh, "DG", 0))
        cell_diam.interpolate(fire.CellDiameter(mesh))
        radius = global_max(cell_diam)
        tie_tol = 1e-8 * lmin
        nodes = []
        times = []
        for source in self.source_locations:
            dist2.interpolate(sum((x - s) ** 2 for x, s in zip(coords, source)))
            d2min = global_min(dist2)
            if d2min > radius**2:
                raise ValueError(
                    f"Source {source} has no mesh node within {radius} km, "
                    "the largest cell diameter, so it lies outside the mesh."
                )
            d_set = max(np.sqrt(d2min), lmin) if local_time else np.sqrt(d2min)
            d2set = (d_set + tie_tol) ** 2
            nodes.append(np.flatnonzero(dist2.dat.data_ro_with_halos <= d2set))
            if local_time:
                times.append(self.local_travel_time(source, coords, c, c_min=c_min))
        nodes = np.unique(np.concatenate(nodes))

        if local_time:
            value = fire.Function(V).interpolate(reduce(fire.min_value, times))
        else:
            value = fire.Constant(0.0)

        sou_marker = fire.Function(V, name="source_marker")
        Dir_Point_BC(V, fire.Constant(1.0), nodes).apply(sou_marker)
        bcs_eik = [Dir_Point_BC(V, value, nodes)]

        return bcs_eik, sou_marker

    @staticmethod
    def local_travel_time(
        source: tuple[float, ...],
        coords: fire.SpatialCoordinate,
        c: fire.Function,
        c_min: float | None = None,
    ):
        """Return the local travel time from a source as a UFL expression.

        Parameters
        ----------
        source : `tuple`
            Source coordinates.
        coords : `ufl.SpatialCoordinate`
            Spatial coordinates of the mesh of ``c``.
        c : `Firedrake.Function`
            Velocity model.
        c_min : `float`, optional
            Lower bound for the velocity. Default is `None` (no bound).

        Returns
        -------
        `ufl.core.expr.Expr`
            Travel time ``||x - s|| (1 / c(s) + 1 / c(x)) / 2``.

        Raises
        ------
        TypeError
            If ``c`` is not a `Firedrake.Function`.
        ValueError
            If ``c_min`` is not positive.

        Notes
        -----
        The slowness along the straight segment from the source to ``x`` is
        averaged with the trapezoidal rule. The time is exact in a homogeneous
        medium and its error is second order in ``||x - s||`` where the
        velocity is smooth; across a discontinuity it is bounded by
        ``||x - s||`` times the jump of the slowness. The velocity at the source
        is evaluated on a vertex-only mesh by every rank and reduced with
        `global_min`, so it is the same on every rank.
        """
        validate_firedrake_parameter("c", c, "Function")
        validate_numeric("c_min", c_min, accept_parameter_as_none=True, lower_bound=0.0)
        c_src = global_min(
            point_cloud_field(c.function_space().mesh(), [source], c, None)
        )
        c_node = c
        if c_min is not None:
            c_src = max(c_src, c_min)
            # A Constant keeps the value out of the compiled kernel
            c_node = fire.max_value(c, fire.Constant(c_min))
        dist = fire.sqrt(sum((x - s) ** 2 for x, s in zip(coords, source)))
        return 0.5 * dist * (1.0 / fire.Constant(c_src) + 1.0 / c_node)

    def define_int_dom(self, V):
        """Define the integration domain for the Eikonal equation.

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.

        Returns
        -------
        dx : `FiredrakeMeasure`
            Integration domain for the Eikonal equation.
        """

        if self.ele_type_eik == "consistent":
            dx = fire.dx

        if self.ele_type_eik == "underintegrated":
            quad_rule = quadrature_rules(V)[0]
            dx = fire.dx(**quad_rule)

        return dx

    def linear_eik(self, u, vy, c, V):
        """Assemble the linear Eikonal.

        Parameters
        ----------
        u : `Firedrake.TrialFunction`
            Trial function.
        vy : `Firedrake.TestFunction`
            Test function.
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.

        Returns
        -------
        FL : `Firedrake.Form`
            Linear Eikonal equation.
        """

        # Parameters
        f = fire.Constant(1.0)
        dx = self.define_int_dom(V)

        # Weak form
        lhs = fire.inner(fire.grad(u), fire.grad(vy)) * dx
        rhs = f / c * vy * dx
        FL = lhs - rhs

        return FL

    def nonlinear_eik(self, u, vy, c, V, diam_mesh, f_est=1.0):
        """Assemble the Nonlinear Eikonal with stabilizing term.

        Parameters
        ----------
        u : `Firedrake.TrialFunction`
            Trial function.
        vy : `Firedrake.TestFunction`
            Test function.
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters.
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal equation. Default is 1.0.

        Returns
        -------
        FNL: `Firedrake.Form`
            Nonlinear Eikonal equation.
        """

        # Parameters
        f = fire.Constant(1.0)
        dx = self.define_int_dom(V)

        # Stabilizer
        eps = fire.Constant(f_est) * diam_mesh

        # Weak form
        delta = fire.Constant(float_info.epsilon)  # float_info.min
        gr_norm = fire.sqrt(fire.inner(fire.grad(u), fire.grad(u))) + delta
        FNL = (
            gr_norm * vy * dx
            - f / c * vy * dx
            + eps * fire.inner(fire.grad(u), fire.grad(vy)) * dx
        )

        return FNL

    @staticmethod
    def solver_opts(
        nl_solver="newtonls",
        l_solver="preonly",
        user_atol=1e-16,
        user_iter=50,
        monitor=False,
    ):
        """Set the solver parameters.

        Parameters
        ----------
        nl_solver : `str`, optional
            Nonlinear solver type (See PETSC documentation).
        l_solver : `str`, optional
            Linear solver type.
        user_atol : `float`, optional
            Absolute user tolerance.
        user_iter : `float`, optional
            Maximum user iterations.
        monitor : 'bool', optional
            Prints the solver progress.

        Returns
        -------
        param_solver: `dict`
            Solver parameters

        PETSC Documentation
        -------------------
        https://petsc.org/release/manualpages/SNES/SNESType/
        https://petsc.org/release/manualpages/KSP/KSPType/
        https://petsc.org/release/manualpages/PC/PCType/

        Tolerance Types
        ---------------
        atol: F(x) ≤ atol
        rtol: F(x) ≤ rtol∗F(x0)
        stol: || delta x || < stol*|| x ||
        haptol: lhs - rhs < haptol
        haptol < atol < rtol < stol
        """

        # Tolerances and iterations
        user_rtol = user_atol * 1e2
        user_stol = user_atol * 1e3
        ksp_max_it = user_iter

        param_solver = {"snes_type": nl_solver, "ksp_type": l_solver}

        if nl_solver == "newtontr":  # newton, cauchy, dogleg
            param_solver.update({"snes_tr_fallback_type": "newton"})

        if nl_solver == "ngmres":  # difference, none, linesearch
            param_solver.update({"snes_ngmres_select_type": "linesearch"})

        if nl_solver == "qn":
            param_solver.update(
                {
                    "snes_qn_m_type": 5,
                    "snes_qn_powell_descent": True,
                    # lbfgs, broyden, badbroyden
                    "snes_qn_type": "badbroyden",
                    # diagonal, none, scalar, jacobian
                    "snes_qn_scale_type": "jacobian",
                }
            )

        if nl_solver == "ngs":
            param_solver.update(
                {
                    "snes_ngs_sweeps": 2,
                    "snes_ngs_atol": user_atol,
                    "snes_ngs_rtol": user_rtol,
                    "snes_ngs_stol": user_stol,
                    "snes_ngs_max_it": user_iter,
                }
            )

        if nl_solver == "ncg":
            # fr, prp, dy, hs, cd
            param_solver.update({"snes_ncg_type": "cd"})

        if l_solver == "preonly":
            ig_nz = False
            pc_type = "lu"  # lu, cholesky

            # umfpack only supports serial matrices
            param_solver.update({"pc_factor_mat_solver_type": "mumps"})
        else:
            ig_nz = True
            pc_type = "ilu"  # ilu, icc, lu, cholesky

        if l_solver == "gmres":
            param_solver.update(
                {"ksp_gmres_restart": 3, "ksp_gmres_haptol": user_atol * 1e-4}
            )

        param_solver.update(
            {
                "snes_linesearch_type": "l2",  # l2, cp, basic
                "snes_linesearch_damping": 1.0,
                "snes_linesearch_maxlambda"
                if is_firedrake_new()
                else "snes_linesearch_maxstep": 1.0,
                "snes_max_funcs": 1000,
                "snes_linesearch_order": 2,
                "snes_linesearch_alpha": 1e-4,
                "snes_max_it": user_iter,
                "snes_linesearch_rtol": user_rtol,
                "snes_linesearch_atol": user_atol,
                "snes_rtol": user_atol,
                "snes_atol": user_rtol,
                "snes_stol": user_stol,
                "ksp_max_it": ksp_max_it,
                "ksp_rtol": user_rtol,
                "ksp_atol": user_atol,
                "ksp_initial_guess_nonzero": ig_nz,
                "pc_type": pc_type,
                "pc_factor_reuse_ordering": True,
                "snes_monitor": None,
            }
        )

        if monitor:  # For debugging
            param_solver.update(
                {
                    "snes_view": None,
                    "snes_converged_reason": None,
                    "snes_linesearch_monitor": None,
                    "ksp_monitor_true_residual": None,
                    "ksp_converged_reason": None,
                    "report": True,
                    "error_on_nonconvergence": True,
                }
            )

        return param_solver

    @staticmethod
    def initial_guess(c, c_min, V, diam_mesh, typ_igs="constant"):
        """Provide an initial guess for the Eikonal solver.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        c_min : `float`
            Minimum velocity value in the model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters.
        typ_igs : `str`, optional
            Type of initial guess. 'constant' or 'variable'. Default is 'constant'.

        Returns
        -------
        init_guess : `float` or `ufl.algebra.Division`
            Initial guess for the Eikonal solver.
        """

        # Mesh cell diameters
        cell_diameter_function = fire.Function(V)
        cell_diameter_function.interpolate(diam_mesh)

        # Initial guess
        if typ_igs == "constant":
            cell_diam_max = global_max(cell_diameter_function)
            init_guess = cell_diam_max / c_min

        elif typ_igs == "variable":
            init_guess = cell_diameter_function / fire.Constant(c_min)

        else:
            validate_parameter("typ_igs", typ_igs, ["constant", "variable"])

        return init_guess

    def linear_solution(
        self, wf_parameters, nl_solver="vinewtonssls", l_solver="preonly", user_iter=50
    ):
        """Solve the linear Eikonal equation.

        Parameters
        ----------
        wf_parameters : `list`
            List containing the weak form parameters.
            Structure: [u, vy, c, c_min, V, diam_mesh]
            - u : `Firedrake.TrialFunction`
                Trial function.
            - vy : `Firedrake.TestFunction`
                Test function.
            - c : `Firedrake.Function`
                Velocity model without absorbing layer.
            - c_min : `float`
                Minimum velocity value in the model without absorbing layer.
            - V : `Firedrake.FunctionSpace`
                Function space for the Eikonal modeling.
            - diam_mesh : `ufl.geometry.CellDiameter`
                Mesh cell diameters.
        nl_solver : `str`, optional
            Nonlinear solver type. Default is 'vinewtonssls'.
            Options: 'vinewtonssls', 'vinewtonrsls', 'newtonls', 'newtontr',
            'qn', 'ncg', 'ngs', 'ngmres' (See PETSC documentation).
        l_solver : `str`, optional.
            Linear solver type. Default is 'preonly'.
            Options: 'preonly', 'bcgs', 'gmres' (See PETSC documentation).
        user_iter : `int`, optional
            Maximum user iterations. Default is 50.

        Returns
        -------
        yp : `Firedrake.Function`
            Linear Eikonal field.

        PETSC Documentation
        -------------------
        https://petsc.org/release/manualpages/SNES/SNESType/
        https://petsc.org/release/manualpages/KSP/KSPType/
        https://petsc.org/release/manualpages/PC/PCType/
        """

        # Weak form parameters
        u, vy, c, c_min, V, diam_mesh = wf_parameters

        # Initial guess
        yp = fire.Function(V, name="Eikonal (Time [s])")
        yp.assign(self.initial_guess(c, c_min, V, diam_mesh))

        # Linear Eikonal
        FeikL = self.linear_eik(u, vy, c, V)

        # Jacobian of the linear Eikonal
        J = fire.derivative(FeikL, yp)

        user_atol = self.tol**0.75
        while True:
            try:
                # Solver parameters
                p = self.solver_opts(
                    nl_solver=nl_solver,
                    l_solver=l_solver,
                    user_atol=user_atol,
                    user_iter=user_iter,
                )

                # Solving LIN Eikonal
                fire.solve(
                    fire.lhs(FeikL) == fire.rhs(FeikL),
                    yp,
                    bcs=self.bcs_eik,
                    solver_parameters=p,
                    J=J,
                )

                # Final parameters
                solv_ok = "Solver Executed Successfully. "
                print((solv_ok + "AbsTol: {:.1e}").format(user_atol), flush=True)

                return yp

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}", flush=True)

                # Adjusting tolerance
                user_atol = (
                    user_atol * 10 if user_atol < 1e-5 else round(user_atol + 1e-5, 5)
                )
                if user_atol > 1e-4:
                    print("Tolerance too high. Exiting.", flush=True)
                    break

    def nonlinear_solution(
        self,
        wf_parameters,
        nl_solver="vinewtonssls",
        l_solver="preonly",
        user_iter=50,
        lin_sol=None,
    ):
        """Solve the nonlinear Eikonal equation.

        Parameters
        ----------
        wf_parameters : `list`
            List containing the weak form parameters.
            Structure: [vy, c, c_min, V, diam_mesh]
            - vy : `Firedrake.TestFunction`
                Test function.
            - c : `Firedrake.Function`
                Velocity model without absorbing layer.
            - c_min : `float`
                Minimum velocity value in the model without absorbing layer.
            - V : `Firedrake.FunctionSpace`
                Function space for the Eikonal modeling.
            - diam_mesh : `ufl.geometry.CellDiameter`
                Mesh cell diameters.
        nl_solver : `str`, optional
            Nonlinear solver type. Default is 'vinewtonssls'.
            Options: 'vinewtonssls', 'vinewtonrsls', 'newtonls', 'newtontr',
            'qn', 'ncg', 'ngs', 'ngmres' (See PETSC documentation).
        l_solver : `str`, optional.
            Linear solver type. Default is 'preonly'.
            Options: 'preonly', 'bcgs', 'gmres' (See PETSC documentation).
        user_iter : `int`, optional
            Maximum user iterations. Default is 50.
        lin_sol : `Firedrake.Function`, optional
            Linear Eikonal solution. Default is `None`.
            If `None`, an initial guess will be computed.

        Returns
        -------
        yp : `Firedrake.Function`
            Nonlinear Eikonal field.

        PETSC Documentation
        -------------------
        https://petsc.org/release/manualpages/SNES/SNESType/
        https://petsc.org/release/manualpages/KSP/KSPType/
        https://petsc.org/release/manualpages/PC/PCType/
        """

        # Weak form parameters
        vy, c, c_min, V, diam_mesh = wf_parameters

        if lin_sol is None:
            # Initial guess for nonlinear Eikonal
            data_eikL = self.initial_guess(c, c_min, V, diam_mesh).dat.data_with_halos[
                :
            ]
        else:
            # Clean numerical instabilities
            data_eikL = sanitize_num_array(lin_sol.dat.data_with_halos[:])

        user_atol = self.tol
        user_est = self.f_est
        while True:
            try:
                print(f"Iteration for Festab: {user_est:.2f}", flush=True)

                # Preserving intial guess
                yp = fire.Function(V, name="Eikonal (Time [s])")
                yp.dat.data_with_halos[:] = data_eikL

                # Solver parameters
                p = self.solver_opts(
                    nl_solver=nl_solver,
                    l_solver=l_solver,
                    user_atol=user_atol,
                    user_iter=user_iter,
                )
                # Linear Eikonal
                FeikNL = self.nonlinear_eik(yp, vy, c, V, diam_mesh, f_est=user_est)

                # Jacobian of the nonlinear Eikonal
                J = fire.derivative(FeikNL, yp)

                # Solving NL Eikonal
                fire.solve(FeikNL == 0, yp, bcs=self.bcs_eik, solver_parameters=p, J=J)

                # Final parameters
                solv_ok = "Solver Executed Successfully. "
                print((solv_ok + "AbsTol: {:.1e}").format(user_atol), flush=True)
                print((solv_ok + "Festab: {:.2f}").format(user_est), flush=True)

                return yp

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}", flush=True)

                # Adjusting stabilizing factor
                user_est += 0.01
                if user_est > 1.0:
                    user_est = self.f_est
                    print(
                        "\nHigh Stabilizing Factor. Increasing Tolerance!", flush=True
                    )
                    user_atol = (
                        user_atol * 10
                        if user_atol < 1e-5
                        else round(user_atol + 1e-5, 5)
                    )
                    if user_atol > 1e-4:
                        print("High Tolerance. Exiting!", flush=True)
                        exit("No Results for Eikonal Equation")

    def eikonal_solver(self, c, c_min, V, diam_mesh):
        """Solve the Eikonal equation for model without absorbing layer.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model without absorbing layer.
        c_min : `float`
            Minimum velocity value in the model without absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the Eikonal modeling.
        diam_mesh : `ufl.geometry.CellDiameter`
            Mesh cell diameters.

        Returns
        -------
        yp : `Firedrake.Function`
            Eikonal field.

        Notes
        -----
        The velocity is bounded below by ``c_min`` in the weak forms. In the
        cells cut by a velocity discontinuity, a high-order interpolant
        undershoots its nodal values and can even turn negative, which makes
        the Eikonal equation ill-posed and the nonlinear solver fail.

        ``c_min`` enters the forms as a `Firedrake.Constant`, so its value is
        passed to the compiled kernels instead of being written into their
        code. The ranks must compile the same kernels, because PyOP2 generates
        the code of a kernel on one rank and broadcasts it to the others, and
        one kernel then serves every ``c_min``.
        """

        # Functions
        u = fire.TrialFunction(V)
        vy = fire.TestFunction(V)

        # Velocity bounded below by its minimum nodal value
        c = fire.max_value(c, fire.Constant(c_min))

        # Weak form parameters
        wf_parameters = [u, vy, c, c_min, V, diam_mesh]

        # Linear Eikonal
        print("\nSolving Pre-Eikonal", flush=True)
        yp = self.linear_solution(wf_parameters)

        # Nonlinear Eikonal
        print("\nSolving Post-Eikonal", flush=True)
        yp = self.nonlinear_solution(wf_parameters[1:], lin_sol=yp)

        return yp
