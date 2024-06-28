import firedrake as fire
import spyro


# build temp space

mesh = spyro.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)