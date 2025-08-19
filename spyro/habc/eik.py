import firedrake as fire
import numpy as np
from sys import float_info
from os import getcwd

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class Dir_point_bc(fire.DirichletBC):
    '''
    Class for Eikonal boundary conditions at a point.

    Attributes
    ----------
    nodes : `array`
        Points where the boundary condition is to be applied
    '''

    def __init__(self, V, value, nodes):
        '''
        Initialize the Dir_point_bc class.

        Parameters
        ----------
        V : `firedrake function space`
            Function space where the boundary condition is applied
        value : `firedrake constant`
            Value of the boundary condition
        nodes : `array`
            Points where the boundary condition is to be applied

        Returns
        -------
        None
        '''

        # Calling superclass init and providing a dummy subdomain id
        super(Dir_point_bc, self).__init__(V, value, 0)

        # Overriding the "nodes" property
        self.nodes = nodes


class Eikonal():
    '''
    Class for the Nonlinear Eikonal.

    Attributes
    ----------
    bnds : `list` of `arrays`
        Mesh point indices on boundaries of the domain. Structure:
        - [left_boundary, right_boundary, bottom_boundary] for 2D
        - [left_boundary, right_boundary, bottom_boundary,
            left_bnd_y, right_bnd_y] for 3D
    bcs_eik : `list`
        Dirichlet BCs for eikonal
    c : `firedrake function`
        Velocity model without absorbing layer
    mesh: `firedrake mesh`
        Original mesh without absorbing layer
    path_save : `str`
        Path to save Eikonal results
    yp : `firedrake function`
        Eikonal field
    x_data : `array`
        x-coordinates of the domain
    z_data : `array`
        z-coordinates of the domain
    y_data : `array`
        y-coordinates of the domain (3D)

    Methods
    -------
    assemble_eik()
        Assemble the Nonlinear Eikonal with stabilizing term
    define_bcs()
        Impose Dirichlet BCs for Eikonal equation
    clean_inst_num()
        Set NaNs and negative values to zero in an array
    ident_crit_eik()
        Identify the critical points at boundaries subject to reflections
    ident_eik_on_bnd()
        Identify Eikonal minimum values on boundary
    linear_eik()
        Assemble the linear Eikonal
    solve_eik()
        Solve the nonlinear Eikonal
    solve_prop()
        Set the solver parameters
    '''

    def __init__(self, Wave):
        '''
        Initialize the Eikonal class.

        Parameters
        ----------
        Wave : `wave`
            Wave object

        Returns
        -------
        None
        '''

        # Communicator MPI4py
        self.comm = Wave.comm

        # Setting the mesh
        self.mesh = Wave.mesh_original

        # Velocity profile model
        self.c = Wave.c

        # Extract node positions and boundary data
        self.bnds, node_positions = Wave.boundary_data(typ_bnd='eikonal')
        self.z_data, self.x_data = node_positions[:2]
        if Wave.dimension == 3:  # 3D
            self.y_data = node_positions[-1]

        # Path to save data
        self.path_save = getcwd() + "/output/preamble/"

    def define_bcs(self, Wave):
        '''
        Impose Dirichlet BCs for eikonal equation

        Parameters
        ----------
        Wave : `wave`
            Wave object

        Returns
        -------
        None
        '''

        print("\nDefining Eikonal BCs")

        # Identify source locations
        possou = Wave.sources.point_locations

        if Wave.dimension == 2:  # 2D
            sou_ids = [np.where(np.isclose(self.z_data, z_s) & np.isclose(
                self.x_data, x_s))[0] for z_s, x_s in possou]

        if Wave.dimension == 3:  # 3D
            sou_ids = [np.where(np.isclose(self.z_data, z_s) & np.isclose(
                self.x_data, x_s) & np.isclose(self.y_data, y_s))[0]
                for z_s, x_s, y_s in possou]

        # Define BCs for eikonal
        self.bcs_eik = [Dir_point_bc(
            Wave.funct_space_eik, fire.Constant(0.0), ids) for ids in sou_ids]

        # Mark source locations
        sou_marker = fire.Function(Wave.funct_space_eik, name="source_marker")
        sou_marker.assign(0)
        sou_marker.dat.data_with_halos[sou_ids] = 1

        # Save source marker
        outfile = fire.VTKFile(self.path_save + "souEik.pvd")
        outfile.write(sou_marker)

    @staticmethod
    def clean_inst_num(data_arr):
        ''''
        Set NaNs and negative values to zero in an array

        Parameters
        ----------
        data_arr : `array`
            An array with possible with possible NaN or negative components

        Returns
        -------
        data_arr : `array`
            An array with null or positive components
        '''
        data_arr[np.where(np.isnan(data_arr) | np.isinf(
            data_arr) | (data_arr < 0.0))] = 0.0
        return data_arr

    def linear_eik(self, u, vy):
        '''
        Assemble the linear Eikonal

        Parameters
        ----------
        u : `firedrake trial function`
            Trial function
        vy : `firedrake test function`
            Test function

        Returns
        -------
        FL : `firedrake form`
            Linear Eikonal equation
        '''
        f = fire.Constant(1.0)
        lhs = fire.inner(fire.grad(u), fire.grad(vy)) * fire.dx
        rhs = f / self.c * vy * fire.dx
        FL = lhs - rhs
        return FL

    def assemble_eik(self, Wave, u, vy, f_est=1.0):
        '''
        Assemble the Nonlinear Eikonal with stabilizing term

        Parameters
        ----------
        Wave : `wave`
            Wave object
        u : `firedrake trial function`
            Trial function
        vy : `firedrake test function`
            Test function
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal equation

        Returns
        -------
        F: `firedrake form`
            Nonlinear Eikonal equation
        '''

        # Stabilizer
        eps = fire.Constant(f_est) * Wave.diam_mesh
        delta = fire.Constant(float_info.epsilon)  # float_info.min
        gr_norm = fire.sqrt(fire.inner(fire.grad(u), fire.grad(u))) + delta
        f = fire.Constant(1.0)
        F = gr_norm * vy * fire.dx - f / self.c * vy * fire.dx + \
            eps * fire.inner(fire.grad(u), fire.grad(vy)) * fire.dx
        return F

    @staticmethod
    def solve_prop(nl_solver='newtonls', l_solver='preonly',
                   user_atol=1e-16, user_iter=50, monitor=False):
        '''
        Set the solver parameters

        Parameters
        ----------
        nl_solver : `str`, optional
            Nonlinear solver type (See PETSC documentation)
        l_solver : `str`, optional
            Linear solver type
        user_atol : `float`, optional
            Absolute user tolerance
        user_iter : `float`, optional
            Maximum user iterations
        monitor : 'bool', optional
            Prints the solver progress

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
        '''

        # Tolerances and iterations
        user_rtol = user_atol * 1e2
        user_stol = user_atol * 1e3
        ksp_max_it = user_iter

        param_solver = {'snes_type': nl_solver, 'ksp_type': l_solver}

        if nl_solver == 'newtontr':  # newton, cauchy, dogleg
            param_solver.update({'snes_tr_fallback_type': 'newton'})

        if nl_solver == 'ngmres':  # difference, none, linesearch
            param_solver.update({'snes_ngmres_select_type': 'linesearch'})

        if nl_solver == 'qn':
            param_solver.update({'snes_qn_m_type': 5,
                                 'snes_qn_powell_descent': True,
                                 # lbfgs, broyden, badbroyden
                                 'snes_qn_type': 'badbroyden',
                                 # diagonal, none, scalar, jacobian
                                 'snes_qn_scale_type': 'jacobian'})

        if nl_solver == 'ngs':
            param_solver.update({'snes_ngs_sweeps': 2,
                                 'snes_ngs_atol': user_atol,
                                 'snes_ngs_rtol': user_rtol,
                                 'snes_ngs_stol': user_stol,
                                 'snes_ngs_max_it': user_iter})

        if nl_solver == 'ncg':
            # fr, prp, dy, hs, cd
            param_solver.update({'snes_ncg_type': 'cd'})

        if l_solver == 'preonly':
            ig_nz = False
            pc_type = 'lu'  # lu, cholesky

            # mumps
            param_solver.update({'pc_factor_mat_solver_type': 'umfpack'})
        else:
            ig_nz = True
            pc_type = 'ilu'  # ilu, icc, lu, cholesky

        if l_solver == 'gmres':
            param_solver.update({'ksp_gmres_restart': 3,
                                 'ksp_gmres_haptol': user_atol * 1e-4})

        param_solver.update({
            'snes_linesearch_type': 'l2',  # l2, cp, basic
            'snes_linesearch_damping': 1.0,
            'snes_linesearch_maxstep': 1.0,
            'snes_max_funcs': 1000,
            'snes_linesearch_order': 2,
            'snes_linesearch_alpha': 1e-4,
            'snes_max_it': user_iter,
            'snes_linesearch_rtol': user_rtol,
            'snes_linesearch_atol': user_atol,
            'snes_rtol': user_atol,
            'snes_atol': user_rtol,
            'snes_stol': user_stol,
            'ksp_max_it': ksp_max_it,
            'ksp_rtol': user_rtol,
            'ksp_atol': user_atol,
            'ksp_initial_guess_nonzero': ig_nz,
            'pc_type': pc_type,
            'pc_factor_reuse_ordering': True,
            'snes_monitor': None,
        })

        if monitor:  # For debugging
            param_solver.update({
                'snes_view': None,
                'snes_converged_reason': None,
                'snes_linesearch_monitor': None,
                'ksp_monitor_true_residual': None,
                'ksp_converged_reason': None,
                'report': True,
                'error_on_nonconvergence': True})
        return param_solver

    def solve_eik(self, Wave, tol=1e-16, f_est=0.06):
        '''
        Solve the nonlinear Eikonal

        Parameters
        ----------
        Wave : `wave`
            Wave object
        tol : `float`, optional
            User solver tolerance
        f_est: `float`, optional
            Factor for the stabilizing term in Eikonal equation

        Returns
        -------
        None
        '''

        # Functions
        yp = fire.Function(Wave.funct_space_eik, name='Eikonal (Time [s])')
        u = fire.TrialFunction(Wave.funct_space_eik)
        vy = fire.TestFunction(Wave.funct_space_eik)

        # Linear Eikonal
        print("\nSolving Pre-Eikonal")
        FeikL = self.linear_eik(u, vy)
        J = fire.derivative(FeikL, yp)

        # Initial guess
        cell_diameter_function = fire.Function(Wave.funct_space_eik)
        cell_diameter_function.interpolate(Wave.diam_mesh)
        yp.assign(cell_diameter_function.dat.data_with_halos.max()
                  / self.c.dat.data_with_halos.min())

        # Linear Eikonal
        user_atol = tol**0.75
        # vinewtonssls, vinewtonrsls, newtonls, newtontr, qn, ncg, ngs, ngmres
        nl_solver = 'vinewtonssls'
        l_solver = 'preonly'  # preonly, bcgs, gmres
        while True:
            try:
                # Solver parameters
                pL = self.solve_prop(nl_solver=nl_solver, l_solver=l_solver,
                                     user_atol=user_atol, user_iter=50)

                # Solving LIN Eikonal
                fire.solve(fire.lhs(FeikL) == fire.rhs(FeikL), yp,
                           bcs=self.bcs_eik, solver_parameters=pL, J=J)

                solv_ok = "Solver Executed Successfully. "
                print((solv_ok + 'AbsTol: {:.1e}').format(user_atol))
                break

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}")

                # Adjusting tolerance
                user_atol = user_atol * 10 if user_atol < 1e-5 \
                    else round(user_atol + 1e-5, 5)
                if user_atol > 1e-4:
                    print("Tolerance too high. Exiting.")
                    break

        # Clean numerical instabilities
        data_eikL = self.clean_inst_num(yp.dat.data_with_halos[:])

        # Nonlinear Eikonal
        print("\nSolving Post-Eikonal")
        user_atol = tol
        # vinewtonrsls, vinewtonssls, newtonls, qn, ncg, newtontr, ngs, ngmres
        nl_solver = 'vinewtonssls'
        l_solver = 'preonly'  # preonly, bcgs, gmres
        user_est = f_est
        while True:
            try:
                print(f"Iteration for Festab: {user_est:.2f}")

                # Preserving intial guess
                yp = fire.Function(
                    Wave.funct_space_eik, name='Eikonal (Time [s])')
                yp.dat.data_with_halos[:] = data_eikL

                # Solver parameters
                pNL = self.solve_prop(nl_solver=nl_solver, l_solver=l_solver,
                                      user_atol=user_atol, user_iter=50)
                # Solving NL Eikonal
                Feik = self.assemble_eik(Wave, yp, vy, f_est=user_est)
                J = fire.derivative(Feik, yp)
                fire.solve(Feik == 0, yp, bcs=self.bcs_eik,
                           solver_parameters=pNL, J=J)

                # Final parameters
                solv_ok = "Solver Executed Successfully. "
                print((solv_ok + 'AbsTol: {:.1e}').format(user_atol))
                print((solv_ok + 'Festab: {:.2f}').format(user_est))

                self.yp = yp
                break

            except fire.ConvergenceError as e:
                print(f"Error Solving: {e}")

                # Adjusting stabilizing factor
                user_est += 0.01
                if user_est > 1.0:
                    user_est = f_est
                    print("\nHigh Stabilizing Factor. Increasing Tolerance!")
                    user_atol = user_atol * 10 if user_atol < 1e-5 \
                        else round(user_atol + 1e-5, 5)
                    if user_atol > 1e-4:
                        print("High Tolerance. Exiting!")
                        break

        # Save Eikonal results
        eikonal_file = fire.VTKFile(self.path_save + "Eik.pvd")
        eikonal_file.write(self.yp)

    def ident_eik_on_bnd(self, boundary):
        '''
        Identify Eikonal minimum values on a boundary

        Parameters
        ----------
        boundary : `array`
            Domain boundary subject to reflections

        Returns
        -------
        eikmin : `float`
            Minimum eikonal value
        idxmin : 'int'
            Array index corresponding to the minimum eikonal value
        '''

        boundary_eik = self.yp.dat.data_with_halos[boundary]
        eikmin = boundary_eik.min()
        idxbnd = np.where(boundary_eik == eikmin)[0][0]
        idxmin = boundary[0][idxbnd]  # Original index

        return eikmin, idxmin

    def ident_crit_eik(self, Wave):
        '''
        Identify the critical points at boundaries subject to reflections

        Parameters
        ----------
        Wave : `wave`
            Wave object

        Returns
        -------
        eik_bnd: `list`
            Properties on boundaries according to minimum values of Eikonal
            Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
            - pt_cr : Critical point coordinates
            - c_bnd : Propagation speed at critical point
            - eikmin : Eikonal value in seconds
            - z_par : Inverse of minimum Eikonal (Equivalent to c_bound/lref)
            - lref : Distance to the closest source from critical point
            - sou_cr : Critical source coordinates
        '''

        if Wave.dimension == 2:  # 2D
            bnds_str = ['Left Boundary', 'Right Boundary', 'Bottom Boundary']
        if Wave.dimension == 3:  # 3D
            bnds_str = ['Xmin Boundary', 'Xmax Boundary', 'Bottom Boundary',
                        'Ymin Boundary', 'Ymax Boundary', ]

        # Source locations
        possou = Wave.sources.point_locations

        # Loop over boundaries
        eik_bnd = []
        print("\nIdentifying Critical Points on Boundaries")
        eik_str = "Min Eikonal on {0:>16} (ms): {1:>7.3f} "
        for bnd, bnd_str in zip(self.bnds, bnds_str):

            # Identify Eikonal minimum
            eikmin, idxmin = self.ident_eik_on_bnd(bnd)

            if Wave.dimension == 2:  # 2D
                pt_cr = (self.z_data[idxmin], self.x_data[idxmin])
            if Wave.dimension == 3:  # 3D
                pt_cr = (self.z_data[idxmin], self.x_data[idxmin],
                         self.y_data[idxmin])

            # Identifying propagation speed at critical point
            c_bnd = np.float64(self.c.at(pt_cr).item())

            # Print critical point coordinates
            if Wave.dimension == 2:  # 2D
                pnt_str = "at (in km): ({2:3.3f}, {3:3.3f})"
            if Wave.dimension == 3:  # 3D
                pnt_str = "at (in km): ({2:3.3f}, {3:3.3f}, {4:3.3f})"

            print((eik_str + pnt_str).format(bnd_str, 1e3 * eikmin, *pt_cr))

            # Identify closest source
            lref_allsou = [np.linalg.norm(
                np.asarray(pt_cr) - np.asarray(p_sou)) for p_sou in possou]
            idxsou = np.argmin(lref_allsou)
            lref = lref_allsou[idxsou]
            sou_cr = possou[idxsou]
            z_par = 1 / eikmin

            # Grouping properties
            eik_bnd.append([pt_cr, c_bnd, eikmin, z_par, lref, sou_cr])

        # Sort the list by the minimum Eikonal and then by the maximum velocity
        return sorted(eik_bnd, key=lambda x: (x[2], -x[1]))
