import numpy as np
from .meshing_winslow2d import winslow_smooth_numba, winslow_smooth_vectorized, winslow_smooth_default
from .meshing_utils import (
    generate_water_profile_from_segy, get_surface_entities_by_physical_name, get_nodes_on_surface_entities,
    get_water_interface_node_indices, align_water_columns_to_interface_x
)


def build_gmsh_geometry_and_groups(
    gmsh, fname, length_x, depth_z, padding_type, padding_x, padding_z,
    hyper_n, water_interface, water_search_value, structured_mesh, minElementSize
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
    z_water_L, z_water_R = None, None

    if padding_type == "hyperelliptical":
        a_val = (length_x / 2.0) + padding_x
        b_val = abs(depth_z / 2.0) + padding_z
    elif padding_type == "rectangular":
        pad_x_min, pad_x_max = -padding_x, length_x + padding_x
        pad_z_min = depth_z - padding_z

    if water_interface:
        Xs, Z_bottom = generate_water_profile_from_segy(
            fname, z_min=0.0, z_max=depth_z, x_min=0.0, x_max=length_x,
            value=water_search_value, tolerance=1.0
        )
        z_water_L, z_water_R = float(Z_bottom[0]), float(Z_bottom[-1])

        pB = [gmsh.model.occ.addPoint(x, float(z), 0.0) for x, z in zip(Xs, Z_bottom)]
        bottom_curve = gmsh.model.occ.addSpline(pB)
        gmsh.model.occ.synchronize()

        pt_top_left = gmsh.model.occ.addPoint(float(Xs[0]), 0.0, 0.0)
        pt_top_right = gmsh.model.occ.addPoint(float(Xs[-1]), 0.0, 0.0)
        line_left = gmsh.model.occ.addLine(pt_top_left, pB[0])
        line_right = gmsh.model.occ.addLine(pB[-1], pt_top_right)
        line_top = gmsh.model.occ.addLine(pt_top_right, pt_top_left)

        curve_loop = gmsh.model.occ.addCurveLoop([line_left, bottom_curve, line_right, line_top])
        water_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()

        if padding_type is None:
            rectangle_tag = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            fragment_result, fragment_map = gmsh.model.occ.fragment([(2, rectangle_tag), (2, water_surface)], [])
            gmsh.model.occ.synchronize()
            water_tags = [tag for dim, tag in fragment_map[1]]
            clipped_rect_tags = [tag for dim, tag in fragment_map[0] if tag not in water_tags]
            gmsh.model.addPhysicalGroup(2, water_tags, name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, clipped_rect_tags, name="SubSurface")

        if padding_type == "rectangular":
            pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_pad_tl_top = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            pt_pad_tl_bot = gmsh.model.occ.addPoint(pad_x_min, z_water_L, 0.0)
            pt_pad_l_bot = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            pt_pad_tr_top = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)
            pt_pad_tr_bot = gmsh.model.occ.addPoint(pad_x_max, z_water_R, 0.0)
            pt_pad_r_bot = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            pt_pad_b_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            pt_pad_bc_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            pt_pad_bc_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            pt_pad_b_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)

            rock_right = gmsh.model.occ.addLine(pB[-1], pt_rock_br)
            rock_bottom = gmsh.model.occ.addLine(pt_rock_br, pt_rock_bl)
            rock_left = gmsh.model.occ.addLine(pt_rock_bl, pB[0])
            pad_tr_top = gmsh.model.occ.addLine(pt_top_right, pt_pad_tr_top)
            pad_tr_right = gmsh.model.occ.addLine(pt_pad_tr_top, pt_pad_tr_bot)
            pad_tr_bot = gmsh.model.occ.addLine(pt_pad_tr_bot, pB[-1])
            pad_mr_right = gmsh.model.occ.addLine(pt_pad_tr_bot, pt_pad_r_bot)
            pad_mr_bot = gmsh.model.occ.addLine(pt_pad_r_bot, pt_rock_br)
            pad_tl_top = gmsh.model.occ.addLine(pt_top_left, pt_pad_tl_top)
            pad_tl_left = gmsh.model.occ.addLine(pt_pad_tl_top, pt_pad_tl_bot)
            pad_tl_bot = gmsh.model.occ.addLine(pt_pad_tl_bot, pB[0])
            pad_ml_left = gmsh.model.occ.addLine(pt_pad_tl_bot, pt_pad_l_bot)
            pad_ml_bot = gmsh.model.occ.addLine(pt_pad_l_bot, pt_rock_bl)
            pad_bl_left = gmsh.model.occ.addLine(pt_pad_l_bot, pt_pad_b_left)
            pad_bl_bot = gmsh.model.occ.addLine(pt_pad_b_left, pt_pad_bc_left)
            pad_bl_right = gmsh.model.occ.addLine(pt_pad_bc_left, pt_rock_bl)
            pad_bc_bot = gmsh.model.occ.addLine(pt_pad_bc_left, pt_pad_bc_right)
            pad_bc_right = gmsh.model.occ.addLine(pt_pad_bc_right, pt_rock_br)
            pad_br_bot = gmsh.model.occ.addLine(pt_pad_bc_right, pt_pad_b_right)
            pad_br_right = gmsh.model.occ.addLine(pt_pad_b_right, pt_pad_r_bot)

            surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([bottom_curve, rock_right, rock_bottom, rock_left])])
            surf_pad_tr = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, pad_tr_top, pad_tr_right, pad_tr_bot])])
            surf_pad_mr = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_tr_bot, pad_mr_right, pad_mr_bot, -rock_right])])
            surf_pad_tl = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([pad_tl_top, pad_tl_left, pad_tl_bot, -line_left])])
            surf_pad_ml = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_tl_bot, pad_ml_left, pad_ml_bot, rock_left])])
            surf_pad_bl = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_ml_bot, pad_bl_left, pad_bl_bot, pad_bl_right])])
            surf_pad_bc = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_bl_right, pad_bc_bot, pad_bc_right, rock_bottom])])
            surf_pad_br = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-pad_bc_right, pad_br_bot, pad_br_right, pad_mr_bot])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_tr, surf_pad_mr, surf_pad_tl, surf_pad_ml, surf_pad_bl, surf_pad_bc, surf_pad_br], name="Padding")

        if padding_type == "hyperelliptical":
            def intersect(x0, z0, dx, dz):
                def f(s):
                    x, z = x0 + s * dx, z0 + s * dz
                    return (abs(x - xc) / a_val)**hyper_n + (abs(z - zc) / b_val)**hyper_n - 1.0
                s_low, s_high = 0.0, 1.0
                while f(s_high) < 0:
                    s_high *= 2.0
                for _ in range(100):
                    s_mid = (s_low + s_high) / 2.0
                    if f(s_mid) > 0:
                        s_high = s_mid
                    else:
                        s_low = s_mid
                return x0 + s_mid * dx, z0 + s_mid * dz

            pt_rock_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            pt_rock_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            z_mid_L, z_mid_R = (z_water_L + depth_z) / 2.0, (z_water_R + depth_z) / 2.0
            x_mid_L, x_mid_R = length_x * 0.25, length_x * 0.75

            pt_mid_L = gmsh.model.occ.addPoint(0.0, z_mid_L, 0.0)
            pt_mid_R = gmsh.model.occ.addPoint(length_x, z_mid_R, 0.0)
            pt_bot_midL = gmsh.model.occ.addPoint(x_mid_L, depth_z, 0.0)
            pt_bot_midR = gmsh.model.occ.addPoint(x_mid_R, depth_z, 0.0)

            x_O_TL, z_O_TL = intersect(0.0, 0.0, -1, 0)
            x_O_WL, z_O_WL = intersect(0.0, z_water_L, -1, 0)
            x_O_ML, z_O_ML = intersect(0.0, z_mid_L, -1, 0)
            x_O_BL_45, z_O_BL_45 = intersect(0.0, depth_z, -1, -1)
            x_O_BML, z_O_BML = intersect(x_mid_L, depth_z, 0, -1)
            x_O_BMR, z_O_BMR = intersect(x_mid_R, depth_z, 0, -1)
            x_O_BR_45, z_O_BR_45 = intersect(length_x, depth_z, 1, -1)
            x_O_MR, z_O_MR = intersect(length_x, z_mid_R, 1, 0)
            x_O_WR, z_O_WR = intersect(length_x, z_water_R, 1, 0)
            x_O_TR, z_O_TR = intersect(length_x, 0.0, 1, 0)

            pt_O_TL = gmsh.model.occ.addPoint(x_O_TL, z_O_TL, 0.0)
            pt_O_WL = gmsh.model.occ.addPoint(x_O_WL, z_O_WL, 0.0)
            pt_O_ML = gmsh.model.occ.addPoint(x_O_ML, z_O_ML, 0.0)
            pt_O_BL_45 = gmsh.model.occ.addPoint(x_O_BL_45, z_O_BL_45, 0.0)
            pt_O_BML = gmsh.model.occ.addPoint(x_O_BML, z_O_BML, 0.0)
            pt_O_BMR = gmsh.model.occ.addPoint(x_O_BMR, z_O_BMR, 0.0)
            pt_O_BR_45 = gmsh.model.occ.addPoint(x_O_BR_45, z_O_BR_45, 0.0)
            pt_O_MR = gmsh.model.occ.addPoint(x_O_MR, z_O_MR, 0.0)
            pt_O_WR = gmsh.model.occ.addPoint(x_O_WR, z_O_WR, 0.0)
            pt_O_TR = gmsh.model.occ.addPoint(x_O_TR, z_O_TR, 0.0)

            rock_R_upper = gmsh.model.occ.addLine(pB[-1], pt_mid_R)
            rock_R_lower = gmsh.model.occ.addLine(pt_mid_R, pt_rock_br)
            rock_B_right = gmsh.model.occ.addLine(pt_rock_br, pt_bot_midR)
            rock_B_mid = gmsh.model.occ.addLine(pt_bot_midR, pt_bot_midL)
            rock_B_left = gmsh.model.occ.addLine(pt_bot_midL, pt_rock_bl)
            rock_L_lower = gmsh.model.occ.addLine(pt_rock_bl, pt_mid_L)
            rock_L_upper = gmsh.model.occ.addLine(pt_mid_L, pB[0])

            ray_TL = gmsh.model.occ.addLine(pt_top_left, pt_O_TL)
            ray_WL = gmsh.model.occ.addLine(pB[0], pt_O_WL)
            ray_ML = gmsh.model.occ.addLine(pt_mid_L, pt_O_ML)
            ray_BL_45 = gmsh.model.occ.addLine(pt_rock_bl, pt_O_BL_45)
            ray_BML = gmsh.model.occ.addLine(pt_bot_midL, pt_O_BML)
            ray_BMR = gmsh.model.occ.addLine(pt_bot_midR, pt_O_BMR)
            ray_BR_45 = gmsh.model.occ.addLine(pt_rock_br, pt_O_BR_45)
            ray_MR = gmsh.model.occ.addLine(pt_mid_R, pt_O_MR)
            ray_WR = gmsh.model.occ.addLine(pB[-1], pt_O_WR)
            ray_TR = gmsh.model.occ.addLine(pt_top_right, pt_O_TR)

            def make_arc(p1_tag, p2_tag, x1, z1, x2, z2, num_pts=25):
                def get_theta(x, z):
                    vx, vz = (x - xc) / a_val, (z - zc) / b_val
                    vx = vx if abs(vx) > 1e-12 else 0.0
                    vz = vz if abs(vz) > 1e-12 else 0.0
                    return np.arctan2(np.sign(vz) * np.abs(vz)**(hyper_n / 2.0),
                                      np.sign(vx) * np.abs(vx)**(hyper_n / 2.0))
                t1, t2 = get_theta(x1, z1), get_theta(x2, z2)
                if t2 - t1 > np.pi:
                    t1 += 2 * np.pi
                elif t1 - t2 > np.pi:
                    t2 += 2 * np.pi
                pts = [p1_tag]
                for t in np.linspace(t1, t2, num_pts)[1:-1]:
                    cos_t, sin_t = np.cos(t), np.sin(t)
                    x = xc + a_val * np.sign(cos_t) * np.abs(cos_t)**(2.0 / hyper_n)
                    z = zc + b_val * np.sign(sin_t) * np.abs(sin_t)**(2.0 / hyper_n)
                    pts.append(gmsh.model.occ.addPoint(x, z, 0.0))
                pts.append(p2_tag)
                return gmsh.model.occ.addSpline(pts)

            arc_TL_WL = make_arc(pt_O_TL, pt_O_WL, x_O_TL, z_O_TL, x_O_WL, z_O_WL)
            arc_WL_ML = make_arc(pt_O_WL, pt_O_ML, x_O_WL, z_O_WL, x_O_ML, z_O_ML)
            arc_ML_BL45 = make_arc(pt_O_ML, pt_O_BL_45, x_O_ML, z_O_ML, x_O_BL_45, z_O_BL_45)
            arc_BL45_BML = make_arc(pt_O_BL_45, pt_O_BML, x_O_BL_45, z_O_BL_45, x_O_BML, z_O_BML)
            arc_BML_BMR = make_arc(pt_O_BML, pt_O_BMR, x_O_BML, z_O_BML, x_O_BMR, z_O_BMR)
            arc_BMR_BR45 = make_arc(pt_O_BMR, pt_O_BR_45, x_O_BMR, z_O_BMR, x_O_BR_45, z_O_BR_45)
            arc_BR45_MR = make_arc(pt_O_BR_45, pt_O_MR, x_O_BR_45, z_O_BR_45, x_O_MR, z_O_MR)
            arc_MR_WR = make_arc(pt_O_MR, pt_O_WR, x_O_MR, z_O_MR, x_O_WR, z_O_WR)
            arc_WR_TR = make_arc(pt_O_WR, pt_O_TR, x_O_WR, z_O_WR, x_O_TR, z_O_TR)

            surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([bottom_curve, rock_R_upper, rock_R_lower, rock_B_right, rock_B_mid, rock_B_left, rock_L_lower, rock_L_upper])])
            surf_pad_TL = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_left, ray_WL, -arc_TL_WL, -ray_TL])])
            surf_pad_ML1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_L_upper, ray_ML, -arc_WL_ML, -ray_WL])])
            surf_pad_ML2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_L_lower, ray_BL_45, -arc_ML_BL45, -ray_ML])])
            surf_pad_B_L = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_left, ray_BML, -arc_BL45_BML, -ray_BL_45])])
            surf_pad_B_M = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_mid, ray_BMR, -arc_BML_BMR, -ray_BML])])
            surf_pad_B_R = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_B_right, ray_BR_45, -arc_BMR_BR45, -ray_BMR])])
            surf_pad_MR2 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_R_lower, ray_MR, -arc_BR45_MR, -ray_BR_45])])
            surf_pad_MR1 = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([-rock_R_upper, ray_WR, -arc_MR_WR, -ray_MR])])
            surf_pad_TR = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, ray_TR, -arc_WR_TR, -ray_WR])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [water_surface], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_TL, surf_pad_ML1, surf_pad_ML2, surf_pad_B_L, surf_pad_B_M, surf_pad_B_R, surf_pad_MR2, surf_pad_MR1, surf_pad_TR], name="Padding")

            if structured_mesh:
                len_radial = max(padding_x, padding_z)
                len_x_left, len_x_mid, len_x_right = length_x * 0.25, length_x * 0.50, length_x * 0.25
                len_z_water = abs(z_water_L)
                len_z_rock_upper, len_z_rock_lower = abs(z_mid_L - z_water_L), abs(depth_z - z_mid_L)

                N_radial = max(2, int(np.ceil(len_radial / minElementSize)) + 1)
                N_X_left = max(2, int(np.ceil(len_x_left / minElementSize)) + 1)
                N_X_mid = max(2, int(np.ceil(len_x_mid / minElementSize)) + 1)
                N_X_right = max(2, int(np.ceil(len_x_right / minElementSize)) + 1)
                N_X_total = N_X_left + N_X_mid + N_X_right - 2
                N_Z_water = max(2, int(np.ceil(len_z_water / minElementSize)) + 1)
                N_Z_rock_upper = max(2, int(np.ceil(len_z_rock_upper / minElementSize)) + 1)
                N_Z_rock_lower = max(2, int(np.ceil(len_z_rock_lower / minElementSize)) + 1)

                for curve in [ray_TL, ray_WL, ray_ML, ray_BL_45, ray_BML, ray_BMR, ray_BR_45, ray_MR, ray_WR, ray_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_radial)
                for curve in [line_left, line_right, arc_TL_WL, arc_WR_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_water)
                for curve in [rock_L_upper, rock_R_upper, arc_WL_ML, arc_MR_WR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_rock_upper)
                for curve in [rock_L_lower, rock_R_lower, arc_ML_BL45, arc_BR45_MR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z_rock_lower)
                for curve in [rock_B_left, arc_BL45_BML]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_left)
                for curve in [rock_B_mid, arc_BML_BMR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_mid)
                for curve in [rock_B_right, arc_BMR_BR45]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_right)
                for curve in [line_top, bottom_curve]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_total)

                padding_surfs = [surf_pad_TL, surf_pad_ML1, surf_pad_ML2, surf_pad_B_L, surf_pad_B_M, surf_pad_B_R, surf_pad_MR2, surf_pad_MR1, surf_pad_TR]
                for surf in padding_surfs:
                    gmsh.model.mesh.setTransfiniteSurface(surf)
                    gmsh.model.mesh.setRecombine(2, surf)

                gmsh.model.mesh.setTransfiniteSurface(water_surface, cornerTags=[pt_top_left, pt_top_right, pB[-1], pB[0]])
                gmsh.model.mesh.setRecombine(2, water_surface)
                gmsh.model.mesh.setTransfiniteSurface(surf_rock, cornerTags=[pB[0], pB[-1], pt_rock_br, pt_rock_bl])
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

            loop_internal = gmsh.model.occ.addCurveLoop([line_top, line_right, line_bottom, line_left])
            surf_internal = gmsh.model.occ.addPlaneSurface([loop_internal])
            loop_pad_left = gmsh.model.occ.addCurveLoop([pad_top_left, -line_left, -pad_bot_left_horiz, pad_left])
            surf_pad_left = gmsh.model.occ.addPlaneSurface([loop_pad_left])
            loop_pad_bl = gmsh.model.occ.addCurveLoop([pad_bot_left_horiz, -pad_corner_bl_left, -pad_corner_bl_bot, -pad_bot_left_vert])
            surf_pad_bl = gmsh.model.occ.addPlaneSurface([loop_pad_bl])
            loop_pad_bot = gmsh.model.occ.addCurveLoop([pad_bot_left_vert, -pad_bot_mid, -pad_bot_right_vert, line_bottom])
            surf_pad_bot = gmsh.model.occ.addPlaneSurface([loop_pad_bot])
            loop_pad_br = gmsh.model.occ.addCurveLoop([pad_bot_right_vert, -pad_corner_br_bot, -pad_corner_br_right, -pad_bot_right_horiz])
            surf_pad_br = gmsh.model.occ.addPlaneSurface([loop_pad_br])
            loop_pad_right = gmsh.model.occ.addCurveLoop([line_right, -pad_bot_right_horiz, -pad_right, -pad_top_right])
            surf_pad_right = gmsh.model.occ.addPlaneSurface([loop_pad_right])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surf_internal], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_left, surf_pad_bl, surf_pad_bot, surf_pad_br, surf_pad_right], name="Padding")

        if padding_type == "hyperelliptical":
            def intersect(x0, z0, dx, dz):
                def f(s):
                    x, z = x0 + s * dx, z0 + s * dz
                    return (abs(x - xc) / a_val)**hyper_n + (abs(z - zc) / b_val)**hyper_n - 1.0
                s_low, s_high = 0.0, 1.0
                while f(s_high) < 0:
                    s_high *= 2.0
                for _ in range(100):
                    s_mid = (s_low + s_high) / 2.0
                    if f(s_mid) > 0:
                        s_high = s_mid
                    else:
                        s_low = s_mid
                return x0 + s_mid * dx, z0 + s_mid * dz

            pt_tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            pt_tr = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            pt_br = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            pt_bl = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            x_mid_L, x_mid_R = length_x * 0.25, length_x * 0.75
            pt_bot_midL = gmsh.model.occ.addPoint(x_mid_L, depth_z, 0.0)
            pt_bot_midR = gmsh.model.occ.addPoint(x_mid_R, depth_z, 0.0)

            x_O_TL, z_O_TL = intersect(0.0, 0.0, -1, 0)
            x_O_BL, z_O_BL = intersect(0.0, depth_z, -1, -1)
            x_O_BML, z_O_BML = intersect(x_mid_L, depth_z, 0, -1)
            x_O_BMR, z_O_BMR = intersect(x_mid_R, depth_z, 0, -1)
            x_O_BR, z_O_BR = intersect(length_x, depth_z, 1, -1)
            x_O_TR, z_O_TR = intersect(length_x, 0.0, 1, 0)

            pt_O_TL = gmsh.model.occ.addPoint(x_O_TL, z_O_TL, 0.0)
            pt_O_BL = gmsh.model.occ.addPoint(x_O_BL, z_O_BL, 0.0)
            pt_O_BML = gmsh.model.occ.addPoint(x_O_BML, z_O_BML, 0.0)
            pt_O_BMR = gmsh.model.occ.addPoint(x_O_BMR, z_O_BMR, 0.0)
            pt_O_BR = gmsh.model.occ.addPoint(x_O_BR, z_O_BR, 0.0)
            pt_O_TR = gmsh.model.occ.addPoint(x_O_TR, z_O_TR, 0.0)

            line_top = gmsh.model.occ.addLine(pt_tl, pt_tr)
            line_right = gmsh.model.occ.addLine(pt_tr, pt_br)
            line_bot_right = gmsh.model.occ.addLine(pt_br, pt_bot_midR)
            line_bot_mid = gmsh.model.occ.addLine(pt_bot_midR, pt_bot_midL)
            line_bot_left = gmsh.model.occ.addLine(pt_bot_midL, pt_bl)
            line_left = gmsh.model.occ.addLine(pt_bl, pt_tl)

            ray_TL = gmsh.model.occ.addLine(pt_tl, pt_O_TL)
            ray_BL = gmsh.model.occ.addLine(pt_bl, pt_O_BL)
            ray_BML = gmsh.model.occ.addLine(pt_bot_midL, pt_O_BML)
            ray_BMR = gmsh.model.occ.addLine(pt_bot_midR, pt_O_BMR)
            ray_BR = gmsh.model.occ.addLine(pt_br, pt_O_BR)
            ray_TR = gmsh.model.occ.addLine(pt_tr, pt_O_TR)

            def make_arc(p1_tag, p2_tag, x1, z1, x2, z2, num_pts=25):
                def get_theta(x, z):
                    vx, vz = (x - xc) / a_val, (z - zc) / b_val
                    vx = vx if abs(vx) > 1e-12 else 0.0
                    vz = vz if abs(vz) > 1e-12 else 0.0
                    return np.arctan2(np.sign(vz) * np.abs(vz)**(hyper_n / 2.0),
                                      np.sign(vx) * np.abs(vx)**(hyper_n / 2.0))
                t1, t2 = get_theta(x1, z1), get_theta(x2, z2)
                if t2 - t1 > np.pi:
                    t1 += 2 * np.pi
                elif t1 - t2 > np.pi:
                    t2 += 2 * np.pi
                pts = [p1_tag]
                for t in np.linspace(t1, t2, num_pts)[1:-1]:
                    cos_t, sin_t = np.cos(t), np.sin(t)
                    x = xc + a_val * np.sign(cos_t) * np.abs(cos_t)**(2.0 / hyper_n)
                    z = zc + b_val * np.sign(sin_t) * np.abs(sin_t)**(2.0 / hyper_n)
                    pts.append(gmsh.model.occ.addPoint(x, z, 0.0))
                pts.append(p2_tag)
                return gmsh.model.occ.addSpline(pts)

            arc_TL_BL = make_arc(pt_O_TL, pt_O_BL, x_O_TL, z_O_TL, x_O_BL, z_O_BL)
            arc_BL_BML = make_arc(pt_O_BL, pt_O_BML, x_O_BL, z_O_BL, x_O_BML, z_O_BML)
            arc_BML_BMR = make_arc(pt_O_BML, pt_O_BMR, x_O_BML, z_O_BML, x_O_BMR, z_O_BMR)
            arc_BMR_BR = make_arc(pt_O_BMR, pt_O_BR, x_O_BMR, z_O_BMR, x_O_BR, z_O_BMR)
            arc_BR_TR = make_arc(pt_O_BR, pt_O_TR, x_O_BR, z_O_BMR, x_O_TR, z_O_TR)

            surf_rock = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_top, line_right, line_bot_right, line_bot_mid, line_bot_left, line_left])])
            surf_pad_left = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([ray_BL, -arc_TL_BL, -ray_TL, -line_left])])
            surf_pad_bot_left = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_left, ray_BL, arc_BL_BML, -ray_BML])])
            surf_pad_bot_mid = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_mid, ray_BML, arc_BML_BMR, -ray_BMR])])
            surf_pad_bot_right = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_bot_right, ray_BMR, arc_BMR_BR, -ray_BR])])
            surf_pad_right = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop([line_right, ray_BR, arc_BR_TR, -ray_TR])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surf_rock], name="SubSurface")
            gmsh.model.addPhysicalGroup(2, [surf_pad_left, surf_pad_bot_left, surf_pad_bot_mid, surf_pad_bot_right, surf_pad_right], name="Padding")

            if structured_mesh:
                len_radial = max(padding_x, padding_z)
                len_x_left, len_x_mid, len_x_right = length_x * 0.25, length_x * 0.50, length_x * 0.25
                len_z_total = abs(depth_z)

                N_radial = max(2, int(np.ceil(len_radial / minElementSize)) + 1)
                N_X_left = max(2, int(np.ceil(len_x_left / minElementSize)) + 1)
                N_X_mid = max(2, int(np.ceil(len_x_mid / minElementSize)) + 1)
                N_X_right = max(2, int(np.ceil(len_x_right / minElementSize)) + 1)
                N_X_total = N_X_left + N_X_mid + N_X_right - 2
                N_Z = max(2, int(np.ceil(len_z_total / minElementSize)) + 1)

                for curve in [ray_TL, ray_BL, ray_BML, ray_BMR, ray_BR, ray_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_radial)

                for curve in [line_left, line_right, arc_TL_BL, arc_BR_TR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_Z)

                for curve in [line_bot_left, arc_BL_BML]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_left)
                for curve in [line_bot_mid, arc_BML_BMR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_mid)
                for curve in [line_bot_right, arc_BMR_BR]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, N_X_right)

                gmsh.model.mesh.setTransfiniteCurve(line_top, N_X_total)

                gmsh.model.mesh.setTransfiniteSurface(surf_pad_left, cornerTags=[pt_bl, pt_O_BL, pt_O_TL, pt_tl])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_bot_left, cornerTags=[pt_bot_midL, pt_bl, pt_O_BL, pt_O_BML])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_bot_mid, cornerTags=[pt_bot_midR, pt_bot_midL, pt_O_BML, pt_O_BMR])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_bot_right, cornerTags=[pt_br, pt_bot_midR, pt_O_BMR, pt_O_BR])
                gmsh.model.mesh.setTransfiniteSurface(surf_pad_right, cornerTags=[pt_tr, pt_br, pt_O_BR, pt_O_TR])

                padding_surfs = [surf_pad_left, surf_pad_bot_left, surf_pad_bot_mid, surf_pad_bot_right, surf_pad_right]
                for surf in padding_surfs:
                    gmsh.model.mesh.setRecombine(2, surf)

                gmsh.model.mesh.setTransfiniteSurface(surf_rock, cornerTags=[pt_tl, pt_tr, pt_br, pt_bl])
                gmsh.model.mesh.setRecombine(2, surf_rock)

    return {
        "z_water_L": z_water_L, "z_water_R": z_water_R,
        "a_val": a_val, "b_val": b_val, "xc": xc, "zc": zc,
        "pad_x_min": pad_x_min, "pad_x_max": pad_x_max, "pad_z_min": pad_z_min,

        # Surfaces
        "water_surface": locals().get("water_surface"),
        "surf_pad_TL": locals().get("surf_pad_TL"),
        "surf_pad_TR": locals().get("surf_pad_TR"),
        "surf_pad_left": locals().get("surf_pad_left"),

        # Arcs
        "arc_WL_ML": locals().get("arc_WL_ML"),
        "arc_ML_BL45": locals().get("arc_ML_BL45"),
        "arc_BL45_BML": locals().get("arc_BL45_BML"),
        "arc_BML_BMR": locals().get("arc_BML_BMR"),
        "arc_BMR_BR45": locals().get("arc_BMR_BR45"),
        "arc_BR45_MR": locals().get("arc_BR45_MR"),
        "arc_MR_WR": locals().get("arc_MR_WR"),
        "arc_TL_BL": locals().get("arc_TL_BL"),
        "arc_BL_BML": locals().get("arc_BL_BML"),
        "arc_BMR_BR": locals().get("arc_BMR_BR"),
        "arc_BR_TR": locals().get("arc_BR_TR"),

        # Lines / Rock Boundaries
        "rock_L_lower": locals().get("rock_L_lower"),
        "rock_L_upper": locals().get("rock_L_upper"),
        "rock_R_lower": locals().get("rock_R_lower"),
        "rock_R_upper": locals().get("rock_R_upper"),
        "rock_B_left": locals().get("rock_B_left"),
        "rock_B_mid": locals().get("rock_B_mid"),
        "rock_B_right": locals().get("rock_B_right"),
        "line_left": locals().get("line_left"),
        "line_right": locals().get("line_right"),
        "line_bot_left": locals().get("line_bot_left"),
        "line_bot_mid": locals().get("line_bot_mid"),
        "line_bot_right": locals().get("line_bot_right"),

        # Rays
        "ray_BL_45": locals().get("ray_BL_45"),
        "ray_BR_45": locals().get("ray_BR_45"),
        "ray_TL": locals().get("ray_TL"),
        "ray_BL": locals().get("ray_BL"),
        "ray_BR": locals().get("ray_BR"),
        "ray_TR": locals().get("ray_TR"),
        "x_O_TL": locals().get("x_O_TL"),
        "x_O_TR": locals().get("x_O_TR")
    }


def apply_structured_winslow_smoothing2d(
    gmsh, comm, geom_params, length_x, depth_z, padding_type,
    water_interface, hyper_n, winslow_implementation, winslow_iterations,
    winslow_omega, n_samples, n_traces, domain_xmin, domain_xmax,
    domain_zmin, domain_zmax, ef_segy2, parallel_print,
    z_water_L, z_water_R, pad_x_min, pad_x_max, pad_z_min, a_val, b_val, xc, zc, apply_winslow
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
    quads = np.asarray([tag_to_index[tag] for tag in all_quad_nodes], dtype=np.int64).reshape(-1, 4)

    if water_interface:
        water_surface_entities = get_surface_entities_by_physical_name("WaterSurface")
        water_surface_nodes = get_nodes_on_surface_entities(tag_to_index, water_surface_entities)
        interface_nodes = get_water_interface_node_indices(
            tag_to_index=tag_to_index, water_surface_entities=water_surface_entities, length_x=length_x, tol=1e-8
        )
        parallel_print("Aligning water columns with water spline X positions...", comm=comm)
        points_2d, n_snapped, n_cols = align_water_columns_to_interface_x(
            points_2d=points_2d, water_surface_nodes=water_surface_nodes, interface_nodes=interface_nodes, quads=quads
        )
        parallel_print(f"Snapped {n_snapped} water-surface nodes onto {n_cols} spline-X columns.", comm=comm)
    else:
        interface_nodes = set()
        water_surface_nodes = set()

    move_all = set()
    move_X_only = set()
    move_Z_only = set()
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
            locked_surfs = [geom_params.get("water_surface"), geom_params.get("surf_pad_TL"), geom_params.get("surf_pad_TR")]
            locked_surface_nodes = get_nodes(2, locked_surfs)
            outer_middle_arcs = [geom_params.get(k) for k in ["arc_WL_ML", "arc_ML_BL45", "arc_BL45_BML", "arc_BML_BMR", "arc_BMR_BR45", "arc_BR45_MR", "arc_MR_WR"]]
            z_slide_curves = [geom_params.get(k) for k in ["rock_L_lower", "rock_L_upper", "rock_R_lower", "rock_R_upper"]]
            x_slide_curves = [geom_params.get(k) for k in ["rock_B_left", "rock_B_mid", "rock_B_right"]]
            diag_rays = [geom_params.get("ray_BL_45"), geom_params.get("ray_BR_45")]
        else:
            locked_surface_nodes = set()

            outer_middle_arcs = [
                geom_params.get(k) for k in ["arc_TL_BL", "arc_BL_BML", "arc_BML_BMR", "arc_BMR_BR", "arc_BR_TR"]
            ]

            z_slide_curves = [geom_params.get("line_left"), geom_params.get("line_right")]
            x_slide_curves = [geom_params.get("line_bot_left"), geom_params.get("line_bot_mid"), geom_params.get("line_bot_right")]
            diag_rays = [geom_params.get("ray_BL"), geom_params.get("ray_BR")]

        outer_arc_nodes = get_nodes(1, outer_middle_arcs)
        z_slide_nodes = get_nodes(1, z_slide_curves)
        x_slide_nodes = get_nodes(1, x_slide_curves)
        locked_diag_nodes = get_nodes(1, diag_rays)

        corner_indices = set()
        tol = 1e-3
        x_O_TL = geom_params.get("x_O_TL")
        x_O_TR = geom_params.get("x_O_TR")

        for i, pt in enumerate(points_2d):
            x, z = pt
            if water_interface:
                if (abs(x - 0.0) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - length_x) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - 0.0) < tol and abs(z - z_water_L) < tol) or \
                   (abs(x - length_x) < tol and abs(z - z_water_R) < tol):
                    corner_indices.add(i)
            else:
                if (abs(x - 0.0) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - length_x) < tol and abs(z - depth_z) < tol) or \
                   (abs(x - 0.0) < tol and abs(z - 0.0) < tol) or \
                   (abs(x - length_x) < tol and abs(z - 0.0) < tol) or \
                   (x_O_TL is not None and abs(x - x_O_TL) < tol and abs(z - 0.0) < tol) or \
                   (x_O_TR is not None and abs(x - x_O_TR) < tol and abs(z - 0.0) < tol):
                    corner_indices.add(i)

        for i in range(len(points_2d)):
            if i in corner_indices:
                locked.add(i)
            elif i in locked_diag_nodes:
                locked.add(i)
            elif i in locked_surface_nodes:
                if i in x_slide_nodes:
                    move_X_only.add(i)
                elif i in z_slide_nodes:
                    move_Z_only.add(i)
                else:
                    locked.add(i)
            elif not water_interface and abs(points_2d[i][1] - 0.0) < tol:
                move_X_only.add(i)  # Lock top boundary Z axis, slide in X
            elif i in outer_arc_nodes:
                move_hyperellipse.add(i)
            elif i in x_slide_nodes:
                move_X_only.add(i)
            elif i in z_slide_nodes:
                move_Z_only.add(i)
            else:
                move_all.add(i)

    elif padding_type == "rectangular":
        tol = 1e-3
        if water_interface:
            corners_to_lock = [
                (0.0, depth_z), (length_x, depth_z), (pad_x_min, depth_z), (pad_x_max, depth_z),
                (0.0, pad_z_min), (length_x, pad_z_min), (pad_x_min, pad_z_min), (pad_x_max, pad_z_min)
            ]
        else:
            corners_to_lock = [
                (0.0, 0.0), (length_x, 0.0), (pad_x_min, 0.0), (pad_x_max, 0.0),
                (0.0, depth_z), (length_x, depth_z), (pad_x_min, depth_z), (pad_x_max, depth_z),
                (0.0, pad_z_min), (length_x, pad_z_min), (pad_x_min, pad_z_min), (pad_x_max, pad_z_min)
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
                if (x <= 0.0 + tol and z >= z_water_L - tol) or (x >= length_x - tol and z >= z_water_R - tol):
                    locked.add(i)
                    continue
                if i in water_surface_nodes:
                    locked.add(i)
                    continue

            if abs(x - pad_x_min) < tol or abs(x - pad_x_max) < tol or abs(x - 0.0) < tol or abs(x - length_x) < tol:
                move_Z_only.add(i)
                continue

            if abs(z - pad_z_min) < tol or abs(z - depth_z) < tol or (not water_interface and abs(z - 0.0) < tol):
                move_X_only.add(i)
                continue

            move_all.add(i)

    elif padding_type is None:
        tol = 1e-3
        corners_to_lock = [(0.0, 0.0), (length_x, 0.0), (0.0, depth_z), (length_x, depth_z)]

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
                move_Z_only.add(i)
                continue
            if abs(z - 0.0) < tol or abs(z - depth_z) < tol:
                move_X_only.add(i)
                continue

            move_all.add(i)

    parallel_print(f"Nodes Breakdown | Total: {len(points_2d)}", comm=comm)
    parallel_print(f"Move All: {len(move_all)} | X-Slide: {len(move_X_only)} | Z-Slide: {len(move_Z_only)} | hyperellipse: {len(move_hyperellipse)} | Locked: {len(locked)}", comm=comm)

    if apply_winslow:
        parallel_print("Applying Winslow smoothing...", comm=comm)
        if winslow_implementation in ("fast", "numba"):
            nx_grid, nz_grid = n_samples, n_traces
            segy_grid_x = np.linspace(domain_xmin, domain_xmax, nx_grid)
            segy_grid_z = np.linspace(domain_zmin, domain_zmax, nz_grid)
            X_grid, Z_grid = np.meshgrid(segy_grid_x, segy_grid_z, indexing='ij')
            sizes_flat = ef_segy2(X_grid.flatten(), Z_grid.flatten())
            segy_grid_vals = sizes_flat.reshape((nx_grid, nz_grid))

            if winslow_implementation == "fast":
                smoothed_points_2d = winslow_smooth_vectorized(
                    points=points_2d, quads=quads, segy_grid_x=segy_grid_x,
                    segy_grid_z=segy_grid_z, segy_grid_vals=segy_grid_vals,
                    move_all=move_all, move_X_only=move_X_only, move_Z_only=move_Z_only,
                    move_hyperellipse=move_hyperellipse, hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                    iterations=winslow_iterations, omega=winslow_omega
                )
            elif winslow_implementation == "numba":
                smoothed_points_2d = winslow_smooth_numba(
                    points=points_2d, quads=quads, segy_grid_x=segy_grid_x,
                    segy_grid_z=segy_grid_z, segy_grid_vals=segy_grid_vals,
                    move_all=move_all, move_X_only=move_X_only, move_Z_only=move_Z_only,
                    move_hyperellipse=move_hyperellipse, hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                    iterations=winslow_iterations, omega=winslow_omega
                )
        else:
            smoothed_points_2d = winslow_smooth_default(
                points=points_2d, quads=quads, sizing_fn=ef_segy2,
                move_all=move_all, move_X_only=move_X_only, move_Z_only=move_Z_only,
                move_hyperellipse=move_hyperellipse, hyperellipse_params=(a_val, b_val, xc, zc, hyper_n),
                iterations=winslow_iterations, omega=winslow_omega
            )

        smoothed_points_3d = np.zeros_like(points_3d)
        smoothed_points_3d[:, :2] = smoothed_points_2d

        parallel_print("Updating nodes in Gmsh...", comm=comm)
        for i, tag in enumerate(node_tags):
            gmsh.model.mesh.setNode(int(tag), smoothed_points_3d[i].tolist(), [])
    else:
        parallel_print("Skipping Winslow smoothing...", comm=comm)
