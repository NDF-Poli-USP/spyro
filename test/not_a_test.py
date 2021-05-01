import pytest
from firedrake import *
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL

import spyro

from .inputfiles.Model1_gradient_2d import model


def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    vp_exact = Function(V).interpolate(
        4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((z - 1.5) ** 2 + (x + 1.5) ** 2)))
    )
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    vp_guess = Function(V).interpolate(4.0 + 0.0 * x)
    File("guess_vel.pvd").write(vp_guess)
    return vp_guess


wavelet = spyro.sources.full_ricker_wavelet(
    model["timeaxis"]["dt"],
    model["timeaxis"]["tf"],
    model["acquisition"]["frequency"],
)


@pytest.mark.skip(reason="no way of currently testing this")
def test_gradient_talyor_remainder_v2():
    comm = spyro.utils.mpi_init(model)

    mesh, V = spyro.io.read_mesh(model, comm)

    vp_guess = _make_vp_guess(V, mesh)

    sources = spyro.Sources(model, mesh, V, comm).create()

    receivers = spyro.Receivers(model, mesh, V, comm).create()

    vp_exact = _make_vp_exact(V, mesh)

    _, p_exact_recv = spyro.solvers.forward(
        model, mesh, comm, vp_exact, sources, wavelet, receivers
    )

    qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

    class L2Inner(object):
        def __init__(self):
            self.A = assemble(
                TrialFunction(V) * TestFunction(V) * dx(rule=qr_x), mat_type="matfree"
            )
            self.Ap = as_backend_type(self.A).mat()

        def eval(self, _u, _v):
            upet = as_backend_type(_u).vec()
            vpet = as_backend_type(_v).vec()
            A_u = self.Ap.createVecLeft()
            self.Ap.mult(upet, A_u)
            return vpet.dot(A_u)

    class Objective(ROL.Objective):
        def __init__(self, inner_product):
            ROL.Objective.__init__(self)
            self.inner_product = inner_product
            self.p_guess = None
            self.misfit = None

        def value(self, x, tol):
            """Compute the functional"""
            self.p_guess, p_guess_recv = spyro.solvers.forward(
                model,
                mesh,
                comm,
                vp_guess,
                sources,
                wavelet,
                receivers,
                output=False,
            )
            self.misfit = spyro.utils.evaluate_misfit(model, p_guess_recv, p_exact_recv)
            J = spyro.utils.compute_functional(model, self.misfit)
            return J

        def gradient(self, g, x, tol):
            dJ = spyro.solvers.gradient(
                model,
                mesh,
                comm,
                vp_guess,
                receivers,
                self.p_guess,
                self.misfit,
            )
            g.scale(0)
            g.vec += dJ

        def update(self, x, flag, iteration):
            vp_guess.assign(Function(V, x.vec, name="velocity"))

    paramsDict = {
        "Step": {
            "Line Search": {"Descent Method": {"Type": "Quasi-Newton Method"}},
            "Type": "Line Search",
        },
        "Status Test": {"Gradient Tolerance": 1e-12, "Iteration Limit": 20},
    }
    params = ROL.ParameterList(paramsDict, "Parameters")

    inner_product = L2Inner()
    obj = Objective(inner_product)
    u = Function(V).assign(vp_guess)
    opt = FeVector(u.vector(), inner_product)
    d = Function(V)

    x, y = SpatialCoordinate(mesh)
    d.interpolate(sin(x * pi) * sin(y * pi))
    d = FeVector(d.vector(), inner_product)
    # check the gradient using d model pertubation 4 iterations and 2nd order test
    obj.checkGradient(opt, d, 4, 2)


if __name__ == "__main__":
    test_gradient_talyor_remainder_v2()
