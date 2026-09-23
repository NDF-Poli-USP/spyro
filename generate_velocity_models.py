import firedrake as fire
from firedrake import (
    exp,
    pi,
    sin,
    And,
    conditional,
)


def get_velocities(wave):
    
    z = wave.mesh_z
    x = wave.mesh_x
    Lx = wave.mesh_parameters.length_x
    Lz = wave.mesh_parameters.length_z

    # ============================================================
    # Function spaces
    # ============================================================

    V = wave.scalar_function_space

    vp = fire.Function(V, name="Vp")
    vs = fire.Function(V, name="Vs")
    rho = fire.Function(V, name="Rho")


    # ============================================================
    # POST-SALT SEDIMENTS
    # ============================================================

    # Properties increase/decrease with depth.
    #
    # Since z is negative below the surface, use -z as depth.

    depth = -z

    vp_sed = 2.5 + 0.25 * depth
    vs_sed = 1.2 + 0.12 * depth
    rho_sed = 2.10 + 0.05 * depth


    # ============================================================
    # SALT GEOMETRY
    # ============================================================

    # Depth to the top of salt.
    #
    # Positive depth values are used here for easier interpretation.
    #
    # Top of salt is approximately 1 km deep.

    depth_top_salt = (
        1.0
        + 0.15 * sin(2.0 * pi * x / Lx)
        + 0.10 * exp(-((x - 3.0) / 1.0)**2)
    )


    # Depth to the bottom of salt.
    #
    # Large central salt body.

    depth_bottom_salt = (
        2.0
        + 0.25 * exp(-((x - 5.0) / 1.8)**2)
        + 0.15 * sin(2.0 * pi * x / Lx)
    )


    # Convert depths to the z-coordinate convention:
    #
    #   z = -depth
    #
    z_top_salt = -depth_top_salt
    z_bottom_salt = -depth_bottom_salt


    # ============================================================
    # SALT REGION
    # ============================================================

    inside_salt = And(
        z <= z_top_salt,
        z >= z_bottom_salt,
    )


    # ============================================================
    # SALT PROPERTIES
    # ============================================================

    vp_salt = 4.5
    vs_salt = 2.3
    rho_salt = 2.20


    # ============================================================
    # PRE-SALT CARBONATE
    # ============================================================

    vp_presalt = 5.3 + 0.15 * depth
    vs_presalt = 2.8 + 0.08 * depth
    rho_presalt = 2.55 + 0.02 * depth


    # ============================================================
    # BASEMENT
    # ============================================================

    depth_basement = 3.8
    z_basement = -depth_basement

    vp_basement = 6.2
    vs_basement = 3.5
    rho_basement = 2.85


    # ============================================================
    # VELOCITY MODEL
    # ============================================================

    vp_expr = conditional(
        z > z_top_salt,
        vp_sed,

        conditional(
            inside_salt,
            vp_salt,

            conditional(
                z > z_basement,
                vp_presalt,
                vp_basement,
            ),
        ),
    )


    vs_expr = conditional(
        z > z_top_salt,
        vs_sed,

        conditional(
            inside_salt,
            vs_salt,

            conditional(
                z > z_basement,
                vs_presalt,
                vs_basement,
            ),
        ),
    )


    rho_expr = conditional(
        z > z_top_salt,
        rho_sed,

        conditional(
            inside_salt,
            rho_salt,

            conditional(
                z > z_basement,
                rho_presalt,
                rho_basement,
            ),
        ),
    )


    # ============================================================
    # Interpolate
    # ============================================================

    vp.interpolate(vp_expr)
    vs.interpolate(vs_expr)
    rho.interpolate(rho_expr)

    return vp, vs, rho
