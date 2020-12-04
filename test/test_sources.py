import math
from copy import deepcopy

import firedrake as fire
import numpy as np
from firedrake import Constant, dot, dx, grad

import spyro

"""Read in an external mesh and interpolate velocity to it"""
from .inputfiles.Model1_2d_CG import model as model


def test_ricker_varies_in_time():
    """This test ricker time variation when applied to a time-
    dependent PDE (acoustic wave second order in pressure) in
    firedrake. It tests if the right hand side varies in time
    and if the applied ricker function behaves correctly
    """
    ### initial ricker tests
    modelRicker = deepcopy(model)
    frequency = 2
    amplitude = 3

    # tests if ricker starts at zero
    delay = 1.5 * math.sqrt(6.0) / (math.pi * frequency)
    t = 0.0
    test1 = math.isclose(
        spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude),
        0,
        abs_tol=1e-3,
    )

    # tests if the minimum value is correct and occurs at correct location
    minimum = -amplitude * 2 / math.exp(3.0 / 2.0)
    t = 0.0 + delay + math.sqrt(6.0) / (2.0 * math.pi * frequency)
    test2 = math.isclose(
        spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude), minimum
    )
    t = 0.0 + delay - math.sqrt(6.0) / (2.0 * math.pi * frequency)
    test3 = math.isclose(
        spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude), minimum
    )

    # tests if maximum value in correct and occurs at correct location
    t = 0.0 + delay
    test4 = math.isclose(
        spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude),
        amplitude,
    )

    ### model parameters necessary for source creation
    modelRicker["acquisition"]["source_type"] = "Ricker"
    modelRicker["acquisition"]["num_sources"] = 1
    modelRicker["acquisition"]["source_pos"] = [(0.5, 0.5)]
    modelRicker["opts"]["method"] = "CG"
    modelRicker["opts"]["degree"] = 2
    modelRicker["opts"]["dimension"] = 2
    comm = spyro.utils.mpi_init(modelRicker)
    mesh = fire.UnitSquareMesh(10, 10)
    element = fire.FiniteElement("CG", mesh.ufl_cell(), 2, variant="equispaced")
    V = fire.FunctionSpace(mesh, element)
    excitations = spyro.Sources(modelRicker, mesh, V, comm).create()

    ### Defining variables for our wave problem
    t = 0.0
    dt = 0.01
    # dt = fire.Constant(delta_t)
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)
    u_prevs0 = fire.Function(V)
    u_prevs1 = fire.Function(V)

    ### Calling ricker source term
    excitation = excitations[0]
    ricker = Constant(0)
    expr = excitation * ricker
    ricker.assign(
        spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude)
    )
    f = fire.Function(V).assign(expr)

    ### Creating form of a simple second order time-dependent wave equation with
    ### uniform density and wave velocity of 1
    F = (
        (u - 2.0 * u_prevs0 + u_prevs1) / (dt ** 2) * v * dx
        + dot(grad(u_prevs0), grad(v)) * dx
        - f * v * dx
    )
    a, b = fire.lhs(F), fire.rhs(F)

    ### Creating solver object
    bcs = fire.DirichletBC(V, 0.0, "on_boundary")
    A = fire.assemble(a, bcs=None)
    B = fire.assemble(b)

    params = {"ksp_type": "preonly", "pc_type": "jacobi"}
    solver = fire.LinearSolver(A, P=None, solver_parameters=params)
    u_h = fire.Function(V)

    steps = 50
    p_y = np.zeros((steps))
    r_y = np.zeros((steps))
    t_x = np.zeros((steps))

    for step in range(1, steps):
        t = step * float(dt)
        ricker.assign(
            spyro.sources.timedependentSource(modelRicker, t, frequency, amplitude)
        )
        f.assign(expr)
        B = fire.assemble(b)

        solver.solve(u_h, B)

        u_prevs0.assign(u_h)
        u_prevs1.assign(u_prevs0)

        udat = u_h.dat.data[:]

        r_y[step - 1] = spyro.sources.timedependentSource(
            modelRicker, t, frequency, amplitude, delay=1.5
        )
        p_y[step - 1] = udat[204]  # hardcoded mesh and degree dependent location

    #### Add way to test inside PDE

    assert all([test1, test2, test3, test4])


if __name__ == "__main__":
    test_ricker_varies_in_time()
