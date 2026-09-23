import firedrake as fire
from firedrake import (
    exp,
    pi,
    sin,
    And,
    conditional,
)


# ============================================================
# Mesh
# ============================================================

Lx = 10.0       # km
Lz = 5.0        # km

nx = 400
nz = 200

mesh = fire.RectangleMesh(nx, nz, Lx, Lz)

V = fire.FunctionSpace(mesh, "CG", 1)

x, z = fire.SpatialCoordinate(mesh)

# POST-SALT SEDIMENTS

vp_sed = 2.5 + 0.25 * z
vs_sed = 1.2 + 0.12 * z
rho_sed = 2.10 + 0.05 * z



# SALT GEOMETRY

# Top of salt
z_top_salt = (
    1.0
    + 0.15 * sin(2.0 * pi * x / Lx)
    + 0.10 * exp(-((x - 3.0) / 1.0)**2)
)


# Bottom of salt
#
# A large central salt body
#
z_bottom_salt = (
    2.0
    + 0.25 * exp(-((x - 5.0) / 1.8)**2)
    + 0.15 * sin(2.0 * pi * x / Lx)
)


inside_salt = And(
    z >= z_top_salt,
    z <= z_bottom_salt,
)


# SALT PROPERTIES

vp_salt = 4.5
vs_salt = 2.3
rho_salt = 2.20

# PRE-SALT CARBONATE

vp_presalt = 5.3 + 0.15 * z
vs_presalt = 2.8 + 0.08 * z
rho_presalt = 2.55 + 0.02 * z


# BASEMENT

z_basement = 3.8

vp_basement = 6.2
vs_basement = 3.5
rho_basement = 2.85

vp_expr = conditional(
    z < z_top_salt,
    vp_sed,

    conditional(
        inside_salt,
        vp_salt,

        conditional(
            z < z_basement,
            vp_presalt,
            vp_basement,
        ),
    ),
)

vs_expr = conditional(
    z < z_top_salt,
    vs_sed,

    conditional(
        inside_salt,
        vs_salt,

        conditional(
            z < z_basement,
            vs_presalt,
            vs_basement,
        ),
    ),
)

rho_expr = conditional(
    z < z_top_salt,
    rho_sed,

    conditional(
        inside_salt,
        rho_salt,

        conditional(
            z < z_basement,
            rho_presalt,
            rho_basement,
        ),
    ),
)

vp = fire.Function(V, name="Vp")
vs = fire.Function(V, name="Vs")
rho = fire.Function(V, name="Rho")

vp.interpolate(vp_expr)
vs.interpolate(vs_expr)
rho.interpolate(rho_expr)

# ============================================================
# Output
# ============================================================

fire.VTKFile("vp.pvd").write(vp)
fire.VTKFile("vs.pvd").write(vs)
fire.VTKFile("rho.pvd").write(rho)
