"""Constructs Firedrake solver for the acosutic wave with typical BCs, NRBCs or HABCs."""

import numpy as np
import firedrake as fire
from firedrake import ds, dx, dot, grad
from ..utils.typing import AbsorbingBCsType


def construct_solver_or_matrix_no_pml(wave):
    """Builds solver operators for wave propagator with typical BCs, NRBCs or HABCs.

    Doesn't create mass matrices if matrix_free option is on, which it is by default.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
    """
    V = wave.function_space
    quad_rule = wave.quadrature_rule

    # typical CG FEM in 2d/3d
    u = fire.TrialFunction(V)
    v = fire.TestFunction(V)

    u_nm1 = fire.Function(V, name="pressure t-dt")
    u_n = fire.Function(V, name="pressure")
    u_np1 = fire.Function(V, name="pressure t+dt")
    wave.u_nm1 = u_nm1
    wave.u_n = u_n
    wave.u_np1 = u_np1

    wave.current_time = 0.0
    dt = wave.dt

    # -------------------------------------------------------
    m1 = (
        (1 / (wave.c * wave.c))
        * ((u - 2.0 * u_n + u_nm1) / dt**2)
        * v
        * dx(**quad_rule)
    )
    a = dot(grad(u_n), grad(v)) * dx(**quad_rule)  # explicit

    le = 0.0
    q = wave.source_expression
    if q is not None:
        le += - q * v * dx(**quad_rule)

    if wave.abc_active and not wave.abc_get_ref_model:
        weak_expr_abc = dot((u_n - u_nm1) / dt, v)

        f_abc = (1 / wave.c) * weak_expr_abc
        qr_s = wave.surface_quadrature_rule

        if wave.abc_type == AbsorbingBCsType.HYBRID:

            # NRBC
            le += wave.cosHig * f_abc * ds(**qr_s)

            # Damping
            le += wave.eta_mask * weak_expr_abc * \
                (1 / (wave.c * wave.c)) * \
                wave.eta_habc * dx(**quad_rule)

        else:
            if wave.absorb_top:
                le += f_abc*ds(1, **qr_s)
            if wave.absorb_bottom:
                le += f_abc*ds(2, **qr_s)
            if wave.absorb_right:
                le += f_abc*ds(3, **qr_s)
            if wave.absorb_left:
                le += f_abc*ds(4, **qr_s)
            if wave.dimension == 3:
                if wave.absorb_front:
                    le += f_abc*ds(5, **qr_s)
                if wave.absorb_back:
                    le += f_abc*ds(6, **qr_s)

    if wave.damping_layer is not None:
        # Same sign convention as the absorbing term above.
        gamma = damping_profile(wave)
        le += (gamma / (wave.c * wave.c)) * dot((u_n - u_nm1) / dt, v) \
            * dx(**quad_rule)

    # form = m1 + a - le
    # Signal for le is + in derivation, see Salas et al (2022)
    # doi: https://doi.org/10.1016/j.apm.2022.09.014
    # TODO: Add citation
    form = m1 + a + le
    wave.rhs = fire.rhs(form)
    wave.lhs = fire.lhs(form)
    wave.set_forward_residual_form(
        fire.action(wave.lhs, u_np1) - wave.rhs,
        (u_np1, u_n, u_nm1),
        state_space=V,
        state_name="residual pressure",
    )

    wave.source_function = fire.Cofunction(V.dual())

    # Pressure-free top surface (marker 1). Without it the top is natural:
    # zero normal derivative, i.e. a rigid surface.
    bcs = fire.DirichletBC(V, 0.0, 1) if wave.free_surface else None
    # Kept on the wave so that the UFL-derived adjoint applies the same
    # boundaries, homogeneous.
    wave.bcs = [] if bcs is None else [bcs]

    lin_var = fire.LinearVariationalProblem(
        wave.lhs,
        wave.rhs + wave.source_function,
        u_np1, bcs=bcs, constant_jacobian=True)
    solver_parameters = dict(wave.solver_parameters)
    solver_parameters["mat_type"] = "matfree"
    wave.solver = fire.LinearVariationalSolver(
        lin_var, solver_parameters=solver_parameters,
    )


def damping_profile(wave) -> fire.Function:
    """Quadratic damping coefficient of the layer along the bottom and sides.

    The layer is the outer ``pad_length`` of the mesh on the bottom and on
    both sides, found from the extent of the mesh; the top is not damped.
    With :math:`d` the distance into the layer and :math:`L` its thickness,

    .. math::

        \\gamma = \\gamma_{\\max} \\left(\\frac{d}{L}\\right)^2, \\qquad
        \\gamma_{\\max} = \\frac{3 v_{\\max} \\ln(1/R)}{2L},

    with :math:`R` the theoretical reflection of the layer and
    :math:`v_{\\max}` the largest velocity of the model.

    Parameters
    ----------
    wave : `acoustic_wave.AcousticWave`
        Wave with its mesh, space, velocity and ``damping_layer`` set.

    Returns
    -------
    firedrake.Function
        :math:`\\gamma` in the wave's function space (1/s), zero inside.
    """
    V = wave.function_space
    mesh = V.mesh()
    comm = mesh.comm
    pad = wave.damping_layer["pad_length"]
    reflection = wave.damping_layer["reflection"]
    coordinates = mesh.coordinates.dat.data_ro
    z_min = comm.allreduce(float(coordinates[:, 0].min()), op=_min_op())
    x_min = comm.allreduce(float(coordinates[:, 1].min()), op=_min_op())
    x_max = comm.allreduce(float(coordinates[:, 1].max()), op=_max_op())
    velocity = wave.c.dat.data_ro
    v_max = comm.allreduce(float(velocity.max()) if velocity.size else 0.0,
                           op=_max_op())
    gamma_max = 3.0 * v_max * np.log(1.0 / reflection) / (2.0 * pad)

    z, x = fire.SpatialCoordinate(mesh)[0], fire.SpatialCoordinate(mesh)[1]
    d_x = fire.max_value(fire.max_value(x_min + pad - x, x - (x_max - pad)), 0.0)
    d_z = fire.max_value(z_min + pad - z, 0.0)
    d = fire.max_value(d_x, d_z) / pad
    return fire.Function(V, name="damping").interpolate(gamma_max * d * d)


def _min_op():
    """MPI minimum reduction.

    Returns
    -------
    mpi4py.MPI.Op
        ``MPI.MIN``.
    """
    from mpi4py import MPI
    return MPI.MIN


def _max_op():
    """MPI maximum reduction.

    Returns
    -------
    mpi4py.MPI.Op
        ``MPI.MAX``.
    """
    from mpi4py import MPI
    return MPI.MAX
