from firedrake import dx, Constant, Function
from firedrake import File, CellDiameter, sqrt, inner, grad, TestFunction
from firedrake import TrialFunction, solve, lhs, rhs
import firedrake as fire
import numpy as np


def mapBound(yp, mesh, Lx, Ly):
    '''
    Mapping of positions inside of absorbing layer
    '''

    Lx = -Lx
    xcoord = mesh.coordinates.dat.data[:, 0]
    ycoord = mesh.coordinates.dat.data[:, 1]

    # np.finfo(float).eps
    eps = 1e-10
    ref_bound = ((xcoord <= 0-eps)
                 & ((ycoord <= 0+eps) |
                 (ycoord >= Ly-eps))) | \
        ((ycoord >= 0 - eps) & (xcoord <= Lx + eps))

    x_boundary = xcoord[ref_bound]
    y_boundary = ycoord[ref_bound]

    boundary_points = []
    min_vertical = min_horizontal = np.inf
    max_vertical = max_horizontal = -np.inf

    for i, _ in enumerate(x_boundary):
        point_x = x_boundary[i]
        point_y = y_boundary[i]
        if abs(point_x-Lx) <= eps:
            point_x += eps
        if abs(point_x-0) <= eps:
            point_x -= eps
        if abs(point_y-0) <= eps:
            point_y += eps
        if abs(point_y-Ly) <= eps:
            point_y -= eps

        point = (point_x, point_y)
        boundary_points.append(point)

        value = yp.at(point)

        if y_boundary[i] >= Ly-eps or y_boundary[i] <= 0 + eps:
            if value < min_horizontal:
                min_horizontal = value
                point_h = point
            if value > max_horizontal:
                max_horizontal = value
                point_h_max = point
        else:
            if value < min_vertical:
                min_vertical = value
                point_v = point
            if value > max_vertical:
                max_vertical = value
                point_v_max = point

    points = [
        min_horizontal,
        min_vertical,
        max_horizontal,
        max_vertical,
        point_h,
        point_v,
        point_h_max,
        point_v_max,
    ]

    values = sort_primary_secondary_points(points)
    return values


def sort_primary_secondary_points(points):

    min_horizontal = points[0]
    min_vertical = points[1]
    max_horizontal = points[2]
    max_vertical = points[3]
    point_h = points[4]
    point_v = points[5]
    point_h_max = points[6]
    point_v_max = points[7]

    if min_horizontal < min_vertical:
        min_value = min_horizontal
        min_point = point_h
        sec_value = min_vertical
        sec_point = point_v
    else:
        min_value = min_vertical
        min_point = point_v
        sec_value = min_horizontal
        sec_point = point_h

    if max_horizontal > max_vertical:
        max_value = max_horizontal
        max_point = point_h_max
    else:
        max_value = max_vertical
        max_point = point_v_max

    sorted_points = [
        min_value,
        min_point,
        sec_value,
        sec_point,
        max_value,
        max_point,
    ]

    return sorted_points


class Eikonal_Solve():
    """
    Solve the Eikonal Equation

    Parameters
    ----------
    HABC : HABC
        HABC class
    show : bool, optional
        Show the Eikonal solution, by default False
    """
    def __init__(self, HABC, show=False):
        """
        Solve the Eikonal Equation

        Parameters
        ----------
        HABC : HABC
            HABC class
        show : bool, optional
            Show the Eikonal solution, by default False

        """
        self.mesh = HABC.mesh_without_habc
        self.c_eik = HABC.c_without_habc
        self.V = HABC.function_space_without_habc
        self.sources = HABC.Wave.sources
        print('Defining Eikonal Boundaries')

        self.mask, self.weak_bc_constant, self.mask_zero = self.define_mask(
            show=show
        )

        yp = self.solve_eikonal(show=show)
        if show is True:
            eikonal_file = File('out/Eik.pvd')
            eikonal_file.write(yp)

        temp_values = mapBound(
            yp,
            self.mesh,
            HABC.Wave.length_z,
            HABC.Wave.length_x
        )

        min_value = temp_values[0]
        min_point = temp_values[1]
        sec_value = temp_values[2]
        sec_point = temp_values[3]
        max_value = temp_values[4]
        max_point = temp_values[5]

        self.min_value = min_value
        self.min_point = min_point
        self.sec_value = sec_value
        self.sec_point = sec_point
        self.max_value = max_value
        self.max_point = max_point

        coordinates_critical_points = np.empty([2, 4])

        min_px, min_py = min_point
        sec_px, sec_py = sec_point
        c_eik = self.c_eik
        coordinates_critical_points[0, :] = [
            min_px,
            min_py,
            c_eik.at(min_point),
            min_value
        ]
        coordinates_critical_points[1, :] = [
            sec_px,
            sec_py,
            c_eik.at(sec_point),
            sec_value
        ]

        if show is True:
            np.savetxt(
                'out/Eik.txt',
                coordinates_critical_points,
                delimiter='\t'
            )

        Z = 1 / min_value
        cref = c_eik.at(min_point)

        self.Z = Z
        self.min_point = min_point
        self.cref = cref
        self.min_value = min_value
        self.max_value = max_value

    def define_mask(self, show=False):
        c = self.c_eik
        Eik_FS = self.V
        sources = self.sources
        sources.current_source = 0

        mask = Function(Eik_FS)
        mask = sources.make_mask(mask)
        if show is True:
            File('mask_test.pvd').write(mask)
            File('c_test.pvd').write(c)

        k = Constant(1e4)
        u0 = Constant(1e-3)

        return mask, k, u0

    def solve_eikonal(self, show=False):
        print('-------------------------------------------')
        print('Solve Pre-Eikonal')
        print('-------------------------------------------')
        yp = self.linear_eikonal(show=show)
        print('-------------------------------------------')
        print('Solved pre-eikonal')
        print('-------------------------------------------')
        print('-------------------------------------------')
        print('Solve Post-Eikonal')
        print('-------------------------------------------')
        yp = self.nonlinear_eikonal(yp, show=show)

        return yp

    def linear_eikonal(self, show=False):
        c = self.c_eik
        Eik_FS = self.V
        sources = self.sources
        sources.current_source = 0

        yp = Function(Eik_FS)
        vy = TestFunction(Eik_FS)
        u = TrialFunction(Eik_FS)

        mask = self.mask
        k = self.weak_bc_constant
        u0 = self.mask_zero

        print('-------------------------------------------')
        print('Solve Pre-Eikonal')
        print('-------------------------------------------')
        f = Constant(1.0)
        F1 = inner(grad(u), grad(vy))*dx \
            - f/c*vy*dx + mask * k * inner(u - u0, vy) * dx

        A = fire.assemble(lhs(F1))

        B = fire.Function(Eik_FS)
        B = fire.assemble(rhs(F1), tensor=B)

        solver_parameters = {
            'pc_type': 'hypre',
            'ksp_type': 'gmres',
            'linear_ksp_monitor': None,
            "ksp_monitor": None,
            'ksp_converged_reason': None,
            'ksp_monitor_true_residual': None,
            "ksp_max_it": 20,
        }

        solve(A, yp, B)
        print('-------------------------------------------')
        print('Solved pre-eikonal')
        print('-------------------------------------------')

        if show is True:
            output = File('linear.pvd')
            output.write(yp)
        return yp

    def nonlinear_eikonal(self, yp, show=False):
        mesh = self.mesh
        c = self.c_eik
        Eik_FS = self.V
        sources = self.sources
        sources.current_source = 0

        f = Constant(1.0)
        eps = CellDiameter(mesh)  # Stabilizer

        # mask = sources.make_mask_element(mask)
        mask = self.mask
        k = self.weak_bc_constant
        u0 = self.mask_zero

        vy = TestFunction(Eik_FS)

        weak_bc = mask * k * inner(yp - u0, vy) * dx
        F = inner(sqrt(inner(grad(yp), grad(yp))), vy)*dx \
            + eps*inner(grad(yp), grad(vy))*dx - f / c*vy*dx + weak_bc
        L = 0

        print('Solve Post-Eikonal')

        solver_parameters = {
            'snes_type': 'vinewtonssls',
            "snes_max_it": 1000,
            "snes_atol": 5e-6,
            "snes_rtol": 1e-20,
            'snes_linesearch_type': 'l2',  # basic bt nleqerr cp l2
            "snes_linesearch_damping": 1.00,  # for basic,l2,cp
            "snes_linesearch_maxstep": 0.50,  # bt,l2,cp
            'snes_linesearch_order': 2,  # for newtonls com bt
            'ksp_type': 'gmres',
            'pc_type': 'lu',
            'nonlinear_solver': 'snes',
        }

        solve(F == L, yp, solver_parameters=solver_parameters)
        if show is True:
            output = File('nonlinear.pvd')
            output.write(yp)

        return yp
