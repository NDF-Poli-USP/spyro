import numpy as np
from spyro.mpi.spyro_mpi import SpyroEnsemble
from .meshing_winslow2d import (
    winslow_smooth_numba,
    winslow_smooth_vectorized,
    winslow_smooth_default,
)
from .meshing_utils import (
    generate_water_profile_from_segy,
    get_surface_entities_by_physical_name,
    get_nodes_on_surface_entities,
    get_water_interface_node_indices,
    align_water_columns_to_interface_x,
    intersect,
    make_arc,
)


def build_gmsh_geometry_and_groups(
    gmsh,
    fname,
    length_x,
    depth_z,
    padding_type,
    padding_x,
    padding_z,
    hyper_n,
    water_interface,
    water_search_value,
    structured_mesh,
    min_element_size,
):
    """
    Generates the geometry, handles padding/water interfaces, and creates
    physical groups for the Gmsh model. Returns a dictionary of boundary
    parameters needed for subsequent mesh smoothing operations.
    """
    xc = length_x / 2.0
    zc = depth_z / 2.0
    a_val, b_val = None, None

    pad_x_min, pad_x_max, pad_z_min = None, None, None
    z_water_left, z_water_right = None, None

    if padding_type == "hyperelliptical":
        a_val = (length_x / 2.0) + padding_x
        b_val = abs(depth_z / 2.0) + padding_z
    elif padding_type == "rectangular":
        pad_x_min, pad_x_max = -padding_x, length_x + padding_x
        pad_z_min = depth_z - padding_z

    if water_interface:
        x_coords, z_bottom = generate_water_profile_from_segy(
            fname,
            z_min=0.0,
            z_max=depth_z,
            x_min=0.0,
            x_max=length_x,
            value=water_search_value,
            tolerance=0.001,
        )
        z_water_left, z_water_right = float(z_bottom[0]), float(z_bottom[-1])

        p_b = [gmsh.model.occ.addPoint(x, float(z), 0.0) for x, z in zip(x_coords, z_bottom)]
        bottom_curve = gmsh.model.occ.addSpline(p_b)
        gmsh.model.occ.synchronize()

        pt_top_left = gmsh.model.occ.addPoint(float(x_coords[0]), 0.0, 0.0)
        pt_top_right = gmsh.model.occ.addPoint(float(x_coords[-1]), 0.0, 0.0)
        line_left = gmsh.model.occ.addLine(pt_top_left, p_b[0])
        line_right = gmsh.model.occ.addLine(p_b[-1], pt_top_right)
        line_top = gmsh.model.occ.addLine(pt_top_right, pt_top_left)

        curve_loop = gmsh.model.occ.addCurveLoop(
            [line_left, bottom_curve, line_right, line_top]
        )
        water_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()

        if padding_type is None:
            rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            fragment_result, fragment_map = gmsh.model.occ.fragment(
                [(2, rectangle_tag), (2, water_surface)], []
            )
            gmsh.model.occ.synchronize()
            water_tags = [tag for dim, tag in fragment_map[1]]
            clipped_rect_tags = [
                tag for dim, tag in fragment_map[0] if tag not in water_tags
            ]
            gmsh.model.addPhysicalGroup(2, water_tags, name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, clipped_rect_tags, name="SubSurface")

        if padding_type == "rectangular":
            pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_pad_tl_top = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            pt_pad_tl_bot = gmsh.model.occ.addPoint(pad_x_min, z_water_left, 0.0)
            pt_pad_l_bot = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            pt_pad_tr_top = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)
            pt_pad_tr_bot = gmsh.model.occ.addPoint(pad_x_max, z_water_right, 0.0)
            pt_pad_r_bot = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            pt_pad_b_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            pt_pad_bc_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            pt_pad_b_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)

            rock_right = gmsh.model.occ.addLine(p_b[-1], pt_rock_br)
            rock_bottom = gmsh.model.occ.addLine(pt_rock_br, pt_rock_bl)
            rock_left = gmsh.model.occ.addLine(pt_rock_bl, p_b[0])
            pad_tr_top = gmsh.model.occ.addLine(pt_top_right, pt_pad_tr_top)
            pad_tr_right = gmsh.model.occ.addLine(pt_pad_tr_top, pt_pad_tr_bot)
            pad_tr_bot = gmsh.model.occ.addLine(pt_pad_tr_bot, p_b[-1])
            pad_mr_right = gmsh.model.occ.addLine(pt_pad_tr_bot, pt_pad_r_bot)
            pad_mr_bot = gmsh.model.occ.addLine(pt_pad_r_bot, pt_rock_br)
            pad_tl_top = gmsh.model.occ.addLine(pt_top_left, pt_pad_tl_top)
            pad_tl_left = gmsh.model.occ.addLine(pt_pad_tl_top, pt_pad_tl_bot)
            pad_tl_bot = gmsh.model.occ.addLine(pt_pad_tl_bot, p_b[0])
            pad_ml_left = gmsh.model.occ.addLine(pt_pad_tl_bot, pt_pad_l_bot)
            pad_ml_bot = gmsh.model.occ.addLine(pt_pad_l_bot, pt_rock_bl)
            pad_bl_left = gmsh.model.occ.addLine(pt_pad_l_bot, pt_pad_b_left)
            pad_bl_bot = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bc_left)
            pad_bl_right = gmsh.model.occ.addLine(pt_pad_bc_left, pt_rock_bl)
            pad_bc_bot = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_bc_right)
            pad_bc_right = gmsh.model.occ.addLine(pt_pad_bc_right, pt_rock_br)
            pad_br_bot = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_b_right)
            pad_br_right = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_r_bot)

            surf_rock = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [bottom_curve, rock_right, rock_bottom, rock_left]
                    )
                ]
            )
            surf_pad_tr = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [line_right, pad_tr_top, pad_tr_right, pad_tr_bot]
                    )
                ]
            )
            surf_pad_mr = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-pad_tr_bot, pad_mr_right, pad_mr_bot, -rock_right]
                    )
                ]
            )
            surf_pad_tl = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [pad_tl_top, pad_tl_left, pad_tl_bot, -line_left]
                    )
                ]
            )
            surf_pad_ml = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-pad_tl_bot, pad_ml_left, pad_ml_bot, rock_left]
                    )
                ]
            )
            surf_pad_bl = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-pad_ml_bot, pad_bl_left, pad_bl_bot, pad_bl_right]
                    )
                ]
            )
            surf_pad_bc = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-pad_bl_right, pad_bc_bot, pad_bc_right, rock_bottom]
                    )
                ]
            )
            surf_pad_br = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-pad_bc_right, pad_br_bot, pad_br_right, pad_mr_bot]
                    )
                ]
            )

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [
                    surf_pad_tr,
                    surf_pad_mr,
                    surf_pad_tl,
                    surf_pad_ml,
                    surf_pad_bl,
                    surf_pad_bc,
                    surf_pad_br,
                ],
                name="Padding",
            )

        if padding_type == "hyperelliptical":

            pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            z_mid_l, z_mid_r = (z_water_left + depth_z) / 2.0, (z_water_right + depth_z) / 2.0
            x_mid_l, x_mid_r = length_x * 0.25, length_x * 0.75

            pt_mid_l = gmsh.model.occ.addPoint(0.0, z_mid_l, 0.0)
            pt_mid_r = gmsh.model.occ.addPoint(length_x, z_mid_r, 0.0)
            pt_bot_mid_l = gmsh.model.occ.addPoint(x_mid_l, depth_z, 0.0)
            pt_bot_mid_r = gmsh.model.occ.addPoint(x_mid_r, depth_z, 0.0)

            x_o_tl, z_o_tl = intersect(0.0, 0.0, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_o_wl, z_o_wl = intersect(
                0.0, z_water_left, -1, 0, xc, zc, a_val, b_val, hyper_n
            )
            x_o_ml, z_o_ml = intersect(
                0.0, z_mid_l, -1, 0, xc, zc, a_val, b_val, hyper_n
            )
            x_o_bl_45, z_o_bl_45 = intersect(
                0.0, depth_z, -1, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_bml, z_o_bml = intersect(
                x_mid_l, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_bmr, z_o_bmr = intersect(
                x_mid_r, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_br_45, z_o_br_45 = intersect(
                length_x, depth_z, 1, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_mr, z_o_mr = intersect(
                length_x, z_mid_r, 1, 0, xc, zc, a_val, b_val, hyper_n
            )
            x_o_wr, z_o_wr = intersect(
                length_x, z_water_right, 1, 0, xc, zc, a_val, b_val, hyper_n
            )
            x_o_tr, z_o_tr = intersect(
                length_x, 0.0, 1, 0, xc, zc, a_val, b_val, hyper_n
            )

            pt_o_tl = gmsh.model.occ.addPoint(x_o_tl, z_o_tl, 0.0)
            pt_o_wl = gmsh.model.occ.addPoint(x_o_wl, z_o_wl, 0.0)
            pt_o_ml = gmsh.model.occ.addPoint(x_o_ml, z_o_ml, 0.0)
            pt_o_bl_45 = gmsh.model.occ.addPoint(x_o_bl_45, z_o_bl_45, 0.0)
            pt_o_bml = gmsh.model.occ.addPoint(x_o_bml, z_o_bml, 0.0)
            pt_o_bmr = gmsh.model.occ.addPoint(x_o_bmr, z_o_bmr, 0.0)
            pt_o_br_45 = gmsh.model.occ.addPoint(x_o_br_45, z_o_br_45, 0.0)
            pt_o_mr = gmsh.model.occ.addPoint(x_o_mr, z_o_mr, 0.0)
            pt_o_wr = gmsh.model.occ.addPoint(x_o_wr, z_o_wr, 0.0)
            pt_o_tr = gmsh.model.occ.addPoint(x_o_tr, z_o_tr, 0.0)

            rock_r_upper = gmsh.model.occ.addLine(p_b[-1], pt_mid_r)
            rock_r_lower = gmsh.model.occ.addLine(pt_mid_r, pt_rock_br)
            rock_b_right = gmsh.model.occ.addLine(pt_rock_br, pt_bot_mid_r)
            rock_b_mid = gmsh.model.occ.addLine(pt_bot_mid_r, pt_bot_mid_l)
            rock_b_left = gmsh.model.occ.addLine(pt_bot_mid_l, pt_rock_bl)
            rock_l_lower = gmsh.model.occ.addLine(pt_rock_bl, pt_mid_l)
            rock_l_upper = gmsh.model.occ.addLine(pt_mid_l, p_b[0])

            ray_tl = gmsh.model.occ.addLine(pt_top_left, pt_o_tl)
            ray_wl = gmsh.model.occ.addLine(p_b[0], pt_o_wl)
            ray_ml = gmsh.model.occ.addLine(pt_mid_l, pt_o_ml)
            ray_bl_45 = gmsh.model.occ.addLine(pt_rock_bl, pt_o_bl_45)
            ray_bml = gmsh.model.occ.addLine(pt_bot_mid_l, pt_o_bml)
            ray_bmr = gmsh.model.occ.addLine(pt_bot_mid_r, pt_o_bmr)
            ray_br_45 = gmsh.model.occ.addLine(pt_rock_br, pt_o_br_45)
            ray_mr = gmsh.model.occ.addLine(pt_mid_r, pt_o_mr)
            ray_wr = gmsh.model.occ.addLine(p_b[-1], pt_o_wr)
            ray_tr = gmsh.model.occ.addLine(pt_top_right, pt_o_tr)

            arc_tl_wl = make_arc(
                pt_o_tl,
                pt_o_wl,
                x_o_tl,
                z_o_tl,
                x_o_wl,
                z_o_wl,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_wl_ml = make_arc(
                pt_o_wl,
                pt_o_ml,
                x_o_wl,
                z_o_wl,
                x_o_ml,
                z_o_ml,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_ml_bl45 = make_arc(
                pt_o_ml,
                pt_o_bl_45,
                x_o_ml,
                z_o_ml,
                x_o_bl_45,
                z_o_bl_45,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_bl45_bml = make_arc(
                pt_o_bl_45,
                pt_o_bml,
                x_o_bl_45,
                z_o_bl_45,
                x_o_bml,
                z_o_bml,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_bml_bmr = make_arc(
                pt_o_bml,
                pt_o_bmr,
                x_o_bml,
                z_o_bml,
                x_o_bmr,
                z_o_bmr,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_bmr_br45 = make_arc(
                pt_o_bmr,
                pt_o_br_45,
                x_o_bmr,
                z_o_bmr,
                x_o_br_45,
                z_o_br_45,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_br45_mr = make_arc(
                pt_o_br_45,
                pt_o_mr,
                x_o_br_45,
                z_o_br_45,
                x_o_mr,
                z_o_mr,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_mr_wr = make_arc(
                pt_o_mr,
                pt_o_wr,
                x_o_mr,
                z_o_mr,
                x_o_wr,
                z_o_wr,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_wr_tr = make_arc(
                pt_o_wr,
                pt_o_tr,
                x_o_wr,
                z_o_wr,
                x_o_tr,
                z_o_tr,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )

            surf_rock = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [
                            bottom_curve,
                            rock_r_upper,
                            rock_r_lower,
                            rock_b_right,
                            rock_b_mid,
                            rock_b_left,
                            rock_l_lower,
                            rock_l_upper,
                        ]
                    )
                ]
            )
            surf_pad_tl = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_left, ray_wl, -arc_tl_wl, -ray_tl])]
            )
            surf_pad_ml1 = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_l_upper, ray_ml, -arc_wl_ml, -ray_wl]
                    )
                ]
            )
            surf_pad_ml2 = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_l_lower, ray_bl_45, -arc_ml_bl45, -ray_ml]
                    )
                ]
            )
            surf_pad_b_l = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_b_left, ray_bml, -arc_bl45_bml, -ray_bl_45]
                    )
                ]
            )
            surf_pad_b_m = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_b_mid, ray_bmr, -arc_bml_bmr, -ray_bml]
                    )
                ]
            )
            surf_pad_b_r = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_b_right, ray_br_45, -arc_bmr_br45, -ray_bmr]
                    )
                ]
            )
            surf_pad_mr2 = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_r_lower, ray_mr, -arc_br45_mr, -ray_br_45]
                    )
                ]
            )
            surf_pad_mr1 = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [-rock_r_upper, ray_wr, -arc_mr_wr, -ray_mr]
                    )
                ]
            )
            surf_pad_tr = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_right, ray_tr, -arc_wr_tr, -ray_wr])]
            )

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [
                    surf_pad_tl,
                    surf_pad_ml1,
                    surf_pad_ml2,
                    surf_pad_b_l,
                    surf_pad_b_m,
                    surf_pad_b_r,
                    surf_pad_mr2,
                    surf_pad_mr1,
                    surf_pad_tr,
                ],
                name="Padding",
            )

            if structured_mesh:
                len_radial = max(padding_x, padding_z)
                len_x_left, len_x_mid, len_x_right = (
                    length_x * 0.25,
                    length_x * 0.50,
                    length_x * 0.25,
                )
                len_z_water = abs(z_water_left)
                len_z_rock_upper, len_z_rock_lower = abs(z_mid_l - z_water_left), abs(
                    depth_z - z_mid_l
                )

                n_radial = max(2, int(np.ceil(len_radial / min_element_size)) + 1)
                n_x_left = max(2, int(np.ceil(len_x_left / min_element_size)) + 1)
                n_x_mid = max(2, int(np.ceil(len_x_mid / min_element_size)) + 1)
                n_x_right = max(2, int(np.ceil(len_x_right / min_element_size)) + 1)
                n_x_total = n_x_left + n_x_mid + n_x_right - 2
                n_z_water = max(2, int(np.ceil(len_z_water / min_element_size)) + 1)
                n_z_rock_upper = max(
                    2, int(np.ceil(len_z_rock_upper / min_element_size)) + 1
                )
                n_z_rock_lower = max(
                    2, int(np.ceil(len_z_rock_lower / min_element_size)) + 1
                )

                for curve in [
                    ray_tl,
                    ray_wl,
                    ray_ml,
                    ray_bl_45,
                    ray_bml,
                    ray_bmr,
                    ray_br_45,
                    ray_mr,
                    ray_wr,
                    ray_tr,
                ]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_radial)
                for curve in [line_left, line_right, arc_tl_wl, arc_wr_tr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z_water)
                for curve in [rock_l_upper, rock_r_upper, arc_wl_ml, arc_mr_wr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z_rock_upper)
                for curve in [rock_l_lower, rock_r_lower, arc_ml_bl45, arc_br45_mr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z_rock_lower)
                for curve in [rock_b_left, arc_bl45_bml]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_left)
                for curve in [rock_b_mid, arc_bml_bmr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_mid)
                for curve in [rock_b_right, arc_bmr_br45]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_right)
                for curve in [line_top, bottom_curve]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_total)

                padding_surfs = [
                    surf_pad_tl,
                    surf_pad_ml1,
                    surf_pad_ml2,
                    surf_pad_b_l,
                    surf_pad_b_m,
                    surf_pad_b_r,
                    surf_pad_mr2,
                    surf_pad_mr1,
                    surf_pad_tr,
                ]
                for surf in padding_surfs:
                    gmsh.model.mesh.setTransfiniteSurface(surf)
                    gmsh.model.mesh.setRecombine(2, surf)

                gmsh.model.mesh.setTransfiniteSurface(
                    water_surface, cornerTags=[pt_top_left, pt_top_right, p_b[-1], p_b[0]]
                )
                gmsh.model.mesh.setRecombine(2, water_surface)
                gmsh.model.mesh.setTransfiniteSurface(
                    surf_rock, cornerTags=[p_b[0], p_b[-1], pt_rock_br, pt_rock_bl]
                )
                gmsh.model.mesh.setRecombine(2, surf_rock)

    if not water_interface:
        if padding_type is None:
            rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [rectangle_tag], name="SubSurface")

        if padding_type == "rectangular":
            pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            pt_pad_tl = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            pt_pad_bl = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            pt_pad_b_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            pt_pad_bc_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            pt_pad_b_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)
            pt_pad_br = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            pt_pad_tr = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)

            line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
            line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
            line_bottom = gmsh.model.occ.addLine(pt_br, pt_bl)
            line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

            pad_top_left = gmsh.model.occ.addLine(pt_pad_tl, pt_tl)
            pad_left = gmsh.model.occ.addLine(pt_pad_bl, pt_pad_tl)
            pad_bot_left_horiz = gmsh.model.occ.addLine(pt_bl, pt_pad_bl)

            pad_corner_bl_left = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bl)
            pad_corner_bl_bot = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_b_left)
            pad_bot_left_vert = gmsh.model.occ.addLine(pt_bl, pt_pad_bc_left)

            pad_bot_mid = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_bc_left)
            pad_bot_right_vert = gmsh.model.occ.addLine(pt_br, pt_pad_bc_right)

            pad_corner_br_bot = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_bc_right)
            pad_corner_br_right = gmsh.model.occ.addLine(pt_pad_br, pt_pad_b_right)

            pad_bot_right_horiz = gmsh.model.occ.addLine(pt_pad_br, pt_br)
            pad_right = gmsh.model.occ.addLine(pt_pad_tr, pt_pad_br)
            pad_top_right = gmsh.model.occ.addLine(pt_tr, pt_pad_tr)

            loop_internal = gmsh.model.occ.addCurveLoop(
                [line_top, line_right, line_bottom, line_left]
            )
            surf_internal = gmsh.model.occ.addPlaneSurface([loop_internal])
            loop_pad_left = gmsh.model.occ.addCurveLoop(
                [pad_top_left, -line_left, -pad_bot_left_horiz, pad_left]
            )
            surf_pad_left = gmsh.model.occ.addPlaneSurface([loop_pad_left])
            loop_pad_bl = gmsh.model.occ.addCurveLoop(
                [
                    pad_bot_left_horiz,
                    -pad_corner_bl_left,
                    -pad_corner_bl_bot,
                    -pad_bot_left_vert,
                ]
            )
            surf_pad_bl = gmsh.model.occ.addPlaneSurface([loop_pad_bl])
            loop_pad_bot = gmsh.model.occ.addCurveLoop(
                [pad_bot_left_vert, -pad_bot_mid, -pad_bot_right_vert, line_bottom]
            )
            surf_pad_bot = gmsh.model.occ.addPlaneSurface([loop_pad_bot])
            loop_pad_br = gmsh.model.occ.addCurveLoop(
                [
                    pad_bot_right_vert,
                    -pad_corner_br_bot,
                    -pad_corner_br_right,
                    -pad_bot_right_horiz,
                ]
            )
            surf_pad_br = gmsh.model.occ.addPlaneSurface([loop_pad_br])
            loop_pad_right = gmsh.model.occ.addCurveLoop(
                [line_right, -pad_bot_right_horiz, -pad_right, -pad_top_right]
            )
            surf_pad_right = gmsh.model.occ.addPlaneSurface([loop_pad_right])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surf_internal], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [surf_pad_left, surf_pad_bl, surf_pad_bot, surf_pad_br, surf_pad_right],
                name="Padding",
            )

        if padding_type == "hyperelliptical":

            pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            x_mid_l, x_mid_r = length_x * 0.25, length_x * 0.75
            pt_bot_mid_l = gmsh.model.occ.addPoint(x_mid_l, depth_z, 0.0)
            pt_bot_mid_r = gmsh.model.occ.addPoint(x_mid_r, depth_z, 0.0)

            x_o_tl, z_o_tl = intersect(0.0, 0.0, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_o_bl, z_o_bl = intersect(
                0.0, depth_z, -1, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_bml, z_o_bml = intersect(
                x_mid_l, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_bmr, z_o_bmr = intersect(
                x_mid_r, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_br, z_o_br = intersect(
                length_x, depth_z, 1, -1, xc, zc, a_val, b_val, hyper_n
            )
            x_o_tr, z_o_tr = intersect(
                length_x, 0.0, 1, 0, xc, zc, a_val, b_val, hyper_n
            )

            pt_o_tl = gmsh.model.occ.addPoint(x_o_tl, z_o_tl, 0.0)
            pt_o_bl = gmsh.model.occ.addPoint(x_o_bl, z_o_bl, 0.0)
            pt_o_bml = gmsh.model.occ.addPoint(x_o_bml, z_o_bml, 0.0)
            pt_o_bmr = gmsh.model.occ.addPoint(x_o_bmr, z_o_bmr, 0.0)
            pt_o_br = gmsh.model.occ.addPoint(x_o_br, z_o_br, 0.0)
            pt_o_tr = gmsh.model.occ.addPoint(x_o_tr, z_o_tr, 0.0)

            line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
            line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
            line_bot_right = gmsh.model.occ.addLine(pt_br, pt_bot_mid_r)
            line_bot_mid = gmsh.model.occ.addLine(pt_bot_mid_r, pt_bot_mid_l)
            line_bot_left = gmsh.model.occ.addLine(pt_bot_mid_l, pt_bl)
            line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

            ray_tl = gmsh.model.occ.addLine(pt_tl, pt_o_tl)
            ray_bl = gmsh.model.occ.addLine(pt_bl, pt_o_bl)
            ray_bml = gmsh.model.occ.addLine(pt_bot_mid_l, pt_o_bml)
            ray_bmr = gmsh.model.occ.addLine(pt_bot_mid_r, pt_o_bmr)
            ray_br = gmsh.model.occ.addLine(pt_br, pt_o_br)
            ray_tr = gmsh.model.occ.addLine(pt_tr, pt_o_tr)

            arc_tl_bl = make_arc(
                pt_o_tl,
                pt_o_bl,
                x_o_tl,
                z_o_tl,
                x_o_bl,
                z_o_bl,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_bl_bml = make_arc(
                pt_o_bl,
                pt_o_bml,
                x_o_bl,
                z_o_bl,
                x_o_bml,
                z_o_bml,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_bml_bmr = make_arc(
                pt_o_bml,
                pt_o_bmr,
                x_o_bml,
                z_o_bml,
                x_o_bmr,
                z_o_bmr,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_bmr_br = make_arc(
                pt_o_bmr,
                pt_o_br,
                x_o_bmr,
                z_o_bmr,
                x_o_br,
                z_o_br,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )
            arc_br_tr = make_arc(
                pt_o_br,
                pt_o_tr,
                x_o_br,
                z_o_br,
                x_o_tr,
                z_o_tr,
                xc,
                zc,
                a_val,
                b_val,
                hyper_n,
            )

            surf_rock = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [
                            line_top,
                            line_right,
                            line_bot_right,
                            line_bot_mid,
                            line_bot_left,
                            line_left,
                        ]
                    )
                ]
            )
            surf_pad_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([ray_bl, -arc_tl_bl, -ray_tl, -line_left])]
            )
            surf_pad_bot_left = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [line_bot_left, ray_bl, arc_bl_bml, -ray_bml]
                    )
                ]
            )
            surf_pad_bot_mid = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [line_bot_mid, ray_bml, arc_bml_bmr, -ray_bmr]
                    )
                ]
            )
            surf_pad_bot_right = gmsh.model.occ.addPlaneSurface(
                [
                    gmsh.model.occ.addCurveLoop(
                        [line_bot_right, ray_bmr, arc_bmr_br, -ray_br]
                    )
                ]
            )
            surf_pad_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_right, ray_br, arc_br_tr, -ray_tr])]
            )

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [
                    surf_pad_left,
                    surf_pad_bot_left,
                    surf_pad_bot_mid,
                    surf_pad_bot_right,
                    surf_pad_right,
                ],
                name="Padding",
            )

            if structured_mesh:
                len_radial = max(padding_x, padding_z)
                len_x_left, len_x_mid, len_x_right = (
                    length_x * 0.25,
                    length_x * 0.50,
                    length_x * 0.25,
                )
                len_z_total = abs(depth_z)

                n_radial = max(2, int(np.ceil(len_radial / min_element_size)) + 1)
                n_x_left = max(2, int(np.ceil(len_x_left / min_element_size)) + 1)
                n_x_mid = max(2, int(np.ceil(len_x_mid / min_element_size)) + 1)
                n_x_right = max(2, int(np.ceil(len_x_right / min_element_size)) + 1)
                n_x_total = n_x_left + n_x_mid + n_x_right - 2
                n_z = max(2, int(np.ceil(len_z_total / min_element_size)) + 1)

                for curve in [ray_tl, ray_bl, ray_bml, ray_bmr, ray_br, ray_tr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_radial)

                for curve in [line_left, line_right, arc_tl_bl, arc_br_tr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z)

                for curve in [line_bot_left, arc_bl_bml]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_left)
                for curve in [line_bot_mid, arc_bml_bmr]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_mid)
                for curve in [line_bot_right, arc_bmr_br]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_right)

                gmsh.model.mesh.setTransfiniteCurve(line_top, n_x_total)

                gmsh.model.mesh.setTransfiniteSurface(
                    surf_pad_left, cornerTags=[pt_bl, pt_o_bl, pt_o_tl, pt_tl]
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surf_pad_bot_left,
                    cornerTags=[pt_bot_mid_l, pt_bl, pt_o_bl, pt_o_bml],
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surf_pad_bot_mid,
                    cornerTags=[pt_bot_mid_r, pt_bot_mid_l, pt_o_bml, pt_o_bmr],
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surf_pad_bot_right,
                    cornerTags=[pt_br, pt_bot_mid_r, pt_o_bmr, pt_o_br],
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surf_pad_right, cornerTags=[pt_tr, pt_br, pt_o_br, pt_o_tr]
                )

                padding_surfs = [
                    surf_pad_left,
                    surf_pad_bot_left,
                    surf_pad_bot_mid,
                    surf_pad_bot_right,
                    surf_pad_right,
                ]
                for surf in padding_surfs:
                    gmsh.model.mesh.setRecombine(2, surf)

                gmsh.model.mesh.setTransfiniteSurface(
                    surf_rock, cornerTags=[pt_tl, pt_tr, pt_br, pt_bl]
                )
                gmsh.model.mesh.setRecombine(2, surf_rock)

    return {
        "z_water_left": z_water_left,
        "z_water_right": z_water_right,
        "a_val": a_val,
        "b_val": b_val,
        "xc": xc,
        "zc": zc,
        "pad_x_min": pad_x_min,
        "pad_x_max": pad_x_max,
        "pad_z_min": pad_z_min,
        # Surfaces
        "water_surface": locals().get("water_surface"),
        "surf_pad_tl": locals().get("surf_pad_tl"),
        "surf_pad_tr": locals().get("surf_pad_tr"),
        "surf_pad_left": locals().get("surf_pad_left"),
        # Arcs
        "arc_wl_ml": locals().get("arc_wl_ml"),
        "arc_ml_bl45": locals().get("arc_ml_bl45"),
        "arc_bl45_bml": locals().get("arc_bl45_bml"),
        "arc_bml_bmr": locals().get("arc_bml_bmr"),
        "arc_bmr_br45": locals().get("arc_bmr_br45"),
        "arc_br45_mr": locals().get("arc_br45_mr"),
        "arc_mr_wr": locals().get("arc_mr_wr"),
        "arc_tl_bl": locals().get("arc_tl_bl"),
        "arc_bl_bml": locals().get("arc_bl_bml"),
        "arc_bmr_br": locals().get("arc_bmr_br"),
        "arc_br_tr": locals().get("arc_br_tr"),
        # Lines / Rock Boundaries
        "rock_l_lower": locals().get("rock_l_lower"),
        "rock_l_upper": locals().get("rock_l_upper"),
        "rock_r_lower": locals().get("rock_r_lower"),
        "rock_r_upper": locals().get("rock_r_upper"),
        "rock_b_left": locals().get("rock_b_left"),
        "rock_b_mid": locals().get("rock_b_mid"),
        "rock_b_right": locals().get("rock_b_right"),
        "line_left": locals().get("line_left"),
        "line_right": locals().get("line_right"),
        "line_bot_left": locals().get("line_bot_left"),
        "line_bot_mid": locals().get("line_bot_mid"),
        "line_bot_right": locals().get("line_bot_right"),
        # Rays
        "ray_bl_45": locals().get("ray_bl_45"),
        "ray_br_45": locals().get("ray_br_45"),
        "ray_tl": locals().get("ray_tl"),
        "ray_bl": locals().get("ray_bl"),
        "ray_br": locals().get("ray_br"),
        "ray_tr": locals().get("ray_tr"),
        "x_o_tl": locals().get("x_o_tl"),
        "x_o_tr": locals().get("x_o_tr"),
    }


def apply_structured_winslow_smoothing2d(
    gmsh,
    geom_params,
    length_x,
    depth_z,
    padding_type,
    water_interface,
    hyper_n,
    winslow_implementation,
    winslow_iterations,
    winslow_omega,
    n_samples,
    n_traces,
    domain_xmin,
    domain_xmax,
    domain_zmin,
    domain_zmax,
    ef_segy2,
    z_water_left,
    z_water_right,
    pad_x_min,
    pad_x_max,
    pad_z_min,
    a_val,
    b_val,
    xc,
    zc,
    apply_winslow,
):
    """
    Extracts nodes and elements from a structured Gmsh model, calculates physical
    boundaries, applies Winslow smoothing, and updates the Gmsh node coordinates.
    """
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points_3d = np.asarray(node_coords, dtype=float).reshape(-1, 3)
    points_2d = points_3d[:, :2]

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    all_quad_nodes = []
    for i, t_type in enumerate(elem_types):
        if t_type == 3:
            all_quad_nodes.extend(elem_node_tags[i])

    if not all_quad_nodes:
        raise ValueError("No quadrilaterals found! Winslow requires quads.")

    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
    quads = np.asarray(
        [tag_to_index[tag] for tag in all_quad_nodes], dtype=np.int64
    ).reshape(-1, 4)

    if water_interface:
        water_surface_entities = get_surface_entities_by_physical_name("WaterSurface")
        water_surface_nodes = get_nodes_on_surface_entities(
            tag_to_index, water_surface_entities
        )
        interface_nodes = get_water_interface_node_indices(
            tag_to_index=tag_to_index,
            water_surface_entities=water_surface_entities,
            length_x=length_x,
            tol=1e-8,
        )
        SpyroEnsemble.print("Aligning water columns with water spline X positions...")
        points_2d, n_snapped, n_cols = align_water_columns_to_interface_x(
            points_2d=points_2d,
            water_surface_nodes=water_surface_nodes,
            interface_nodes=interface_nodes,
            quads=quads,
        )
        SpyroEnsemble.print(
            f"Snapped {n_snapped} water-surface nodes onto {n_cols} spline-X columns."
        )
    else:
        interface_nodes = set()
        water_surface_nodes = set()

    move_all = set()
    move_x_only = set()
    move_z_only = set()
    move_hyperellipse = set()
    locked = set()

    def get_nodes(dim, tags):
        nodes = set()
        for t in tags:
            if t is None:
                continue
            n, _, _ = gmsh.model.mesh.getNodes(dim, t, includeBoundary=True)
            if len(n) > 0:
                nodes.update([tag_to_index[node] for node in n])
        return nodes

    if padding_type == "hyperelliptical":
        if water_interface:
            locked_surfs = [
                geom_params.get("water_surface"),
                geom_params.get("surf_pad_tl"),
                geom_params.get("surf_pad_tr"),
            ]
            locked_surface_nodes = get_nodes(2, locked_surfs)
            outer_middle_arcs = [
                geom_params.get(k)
                for k in [
                    "arc_wl_ml",
                    "arc_ml_bl45",
                    "arc_bl45_bml",
                    "arc_bml_bmr",
                    "arc_bmr_br45",
                    "arc_br45_mr",
                    "arc_mr_wr",
                ]
            ]
            z_slide_curves = [
                geom_params.get(k)
                for k in [
                    "rock_l_lower",
                    "rock_l_upper",
                    "rock_r_lower",
                    "rock_r_upper",
                ]
            ]
            x_slide_curves = [
                geom_params.get(k)
                for k in ["rock_b_left", "rock_b_mid", "rock_b_right"]
            ]
            diag_rays = [geom_params.get("ray_bl_45"), geom_params.get("ray_br_45")]
        else:
            locked_surface_nodes = set()

            outer_middle_arcs = [
                geom_params.get(k)
                for k in [
                    "arc_tl_bl",
                    "arc_bl_bml",
                    "arc_bml_bmr",
                    "arc_bmr_br",
                    "arc_br_tr",
                ]
            ]

            z_slide_curves = [
                geom_params.get("line_left"),
                geom_params.get("line_right"),
            ]
            x_slide_curves = [
                geom_params.get("line_bot_left"),
                geom_params.get("line_bot_mid"),
                geom_params.get("line_bot_right"),
            ]
            diag_rays = [geom_params.get("ray_bl"), geom_params.get("ray_br")]

        outer_arc_nodes = get_nodes(1, outer_middle_arcs)
        z_slide_nodes = get_nodes(1, z_slide_curves)
        x_slide_nodes = get_nodes(1, x_slide_curves)
        locked_diag_nodes = get_nodes(1, diag_rays)

        corner_indices = set()
        tol = 1e-3
        x_o_tl = geom_params.get("x_o_tl")
        x_o_tr = geom_params.get("x_o_tr")

        for i, pt in enumerate(points_2d):
            x, z = pt
            if water_interface:
                if (
                    (abs(x - 0.0) < tol and abs(z - depth_z) < tol)
                    or (abs(x - length_x) < tol and abs(z - depth_z) < tol)
                    or (abs(x - 0.0) < tol and abs(z - z_water_left) < tol)
                    or (abs(x - length_x) < tol and abs(z - z_water_right) < tol)
                ):
                    corner_indices.add(i)
            else:
                if (
                    (abs(x - 0.0) < tol and abs(z - depth_z) < tol)
                    or (abs(x - length_x) < tol and abs(z - depth_z) < tol)
                    or (abs(x - 0.0) < tol and abs(z - 0.0) < tol)
                    or (abs(x - length_x) < tol and abs(z - 0.0) < tol)
                    or (
                        x_o_tl is not None
                        and abs(x - x_o_tl) < tol
                        and abs(z - 0.0) < tol
                    )
                    or (
                        x_o_tr is not None
                        and abs(x - x_o_tr) < tol
                        and abs(z - 0.0) < tol
                    )
                ):
                    corner_indices.add(i)

        for i in range(len(points_2d)):
            if i in corner_indices:
                locked.add(i)
            elif i in locked_diag_nodes:
                locked.add(i)
            elif i in locked_surface_nodes:
                if i in x_slide_nodes:
                    move_x_only.add(i)
                elif i in z_slide_nodes:
                    move_z_only.add(i)
                else:
                    locked.add(i)
            elif not water_interface and abs(points_2d[i][1] - 0.0) < tol:
                move_x_only.add(i)  # Lock top boundary Z axis, slide in X
            elif i in outer_arc_nodes:
                move_hyperellipse.add(i)
            elif i in x_slide_nodes:
                move_x_only.add(i)
            elif i in z_slide_nodes:
                move_z_only.add(i)
            else:
                move_all.add(i)

    elif padding_type == "rectangular":
        tol = 1e-3
        if water_interface:
            corners_to_lock = [
                (0.0, depth_z),
                (length_x, depth_z),
                (pad_x_min, depth_z),
                (pad_x_max, depth_z),
                (0.0, pad_z_min),
                (length_x, pad_z_min),
                (pad_x_min, pad_z_min),
                (pad_x_max, pad_z_min),
            ]
        else:
            corners_to_lock = [
                (0.0, 0.0),
                (length_x, 0.0),
                (pad_x_min, 0.0),
                (pad_x_max, 0.0),
                (0.0, depth_z),
                (length_x, depth_z),
                (pad_x_min, depth_z),
                (pad_x_max, depth_z),
                (0.0, pad_z_min),
                (length_x, pad_z_min),
                (pad_x_min, pad_z_min),
                (pad_x_max, pad_z_min),
            ]

        for i, pt in enumerate(points_2d):
            x, z = pt
            is_locked = False
            for cx, cz in corners_to_lock:
                if abs(x - cx) < tol and abs(z - cz) < tol:
                    locked.add(i)
                    is_locked = True
                    break
            if is_locked:
                continue

            if water_interface:
                if (x <= 0.0 + tol and z >= z_water_left - tol) or (
                    x >= length_x - tol and z >= z_water_right - tol
                ):
                    locked.add(i)
                    continue
                if i in water_surface_nodes:
                    locked.add(i)
                    continue

            if (
                abs(x - pad_x_min) < tol
                or abs(x - pad_x_max) < tol
                or abs(x - 0.0) < tol
                or abs(x - length_x) < tol
            ):
                move_z_only.add(i)
                continue

            if (
                abs(z - pad_z_min) < tol
                or abs(z - depth_z) < tol
                or (not water_interface and abs(z - 0.0) < tol)
            ):
                move_x_only.add(i)
                continue

            move_all.add(i)

    elif padding_type is None:
        tol = 1e-3
        corners_to_lock = [
            (0.0, 0.0),
            (length_x, 0.0),
            (0.0, depth_z),
            (length_x, depth_z),
        ]

        for i, pt in enumerate(points_2d):
            x, z = pt
            is_locked = False
            for cx, cz in corners_to_lock:
                if abs(x - cx) < tol and abs(z - cz) < tol:
                    locked.add(i)
                    is_locked = True
                    break
            if is_locked:
                continue

            if water_interface and i in interface_nodes:
                locked.add(i)
                continue

            if abs(x - 0.0) < tol or abs(x - length_x) < tol:
                move_z_only.add(i)
                continue
            if abs(z - 0.0) < tol or abs(z - depth_z) < tol:
                move_x_only.add(i)
                continue

            move_all.add(i)

    SpyroEnsemble.print(f"Nodes Breakdown | Total: {len(points_2d)}")
    SpyroEnsemble.print(
        f"Move All: {len(move_all)} | X-Slide: {len(move_x_only)} | Z-Slide: {len(move_z_only)} | hyperellipse: {len(move_hyperellipse)} | Locked: {len(locked)}"
    )

    if apply_winslow:
        SpyroEnsemble.print("Applying Winslow smoothing...")
        if winslow_implementation in ("fast", "numba"):
            nx_grid, nz_grid = n_samples, n_traces
            segy_grid_x = np.linspace(domain_xmin, domain_xmax, nx_grid)
            segy_grid_z = np.linspace(domain_zmin, domain_zmax, nz_grid)
            x_grid, z_grid = np.meshgrid(segy_grid_x, segy_grid_z, indexing="ij")
            sizes_flat = ef_segy2(x_grid.flatten(), z_grid.flatten())
            segy_grid_vals = sizes_flat.reshape((nx_grid, nz_grid))

            if winslow_implementation == "fast":
                smoothed_points_2d = winslow_smooth_vectorized(
                    points=points_2d,
                    quads=quads,
                    segy_grid_x=segy_grid_x,
                    segy_grid_z=segy_grid_z,
                    segy_grid_vals=segy_grid_vals,
                    move_all=move_all,
                    move_x_only=move_x_only,
                    move_z_only=move_z_only,
                    move_hyperellipse=move_hyperellipse,
                    hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                    iterations=winslow_iterations,
                    omega=winslow_omega,
                )
            elif winslow_implementation == "numba":
                smoothed_points_2d = winslow_smooth_numba(
                    points=points_2d,
                    quads=quads,
                    segy_grid_x=segy_grid_x,
                    segy_grid_z=segy_grid_z,
                    segy_grid_vals=segy_grid_vals,
                    move_all=move_all,
                    move_x_only=move_x_only,
                    move_z_only=move_z_only,
                    move_hyperellipse=move_hyperellipse,
                    hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                    iterations=winslow_iterations,
                    omega=winslow_omega,
                )
        else:
            smoothed_points_2d = winslow_smooth_default(
                points=points_2d,
                quads=quads,
                sizing_fn=ef_segy2,
                move_all=move_all,
                move_x_only=move_x_only,
                move_z_only=move_z_only,
                move_hyperellipse=move_hyperellipse,
                hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                iterations=winslow_iterations,
                omega=winslow_omega,
            )

        smoothed_points_3d = np.zeros_like(points_3d)
        smoothed_points_3d[:, :2] = smoothed_points_2d

        SpyroEnsemble.print("Updating nodes in Gmsh...")
        for i, tag in enumerate(node_tags):
            gmsh.model.mesh.setNode(int(tag), smoothed_points_3d[i].tolist(), [])
    else:
        SpyroEnsemble.print("Skipping Winslow smoothing...")
