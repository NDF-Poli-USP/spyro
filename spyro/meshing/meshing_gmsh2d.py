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
    # Positions refer to the Gmsh (x, z) plane before the Spyro axis swap:
    # top is z=0, bottom is z=depth_z < 0, and padding lies outside the domain.
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
        x_water_interface, z_water_interface = generate_water_profile_from_segy(
            fname,
            z_min=0.0,
            z_max=depth_z,
            x_min=0.0,
            x_max=length_x,
            value=water_search_value,
            tolerance=0.001,
        )
        z_water_left, z_water_right = float(z_water_interface[0]), float(z_water_interface[-1])

        points_water_interface = [gmsh.model.occ.addPoint(x, float(z), 0.0) for x, z in zip(x_water_interface, z_water_interface)]
        curve_water_interface = gmsh.model.occ.addSpline(points_water_interface)
        gmsh.model.occ.synchronize()

        point_top_left = gmsh.model.occ.addPoint(float(x_water_interface[0]), 0.0, 0.0)
        point_top_right = gmsh.model.occ.addPoint(float(x_water_interface[-1]), 0.0, 0.0)
        line_left = gmsh.model.occ.addLine(point_top_left, points_water_interface[0])
        line_right = gmsh.model.occ.addLine(points_water_interface[-1], point_top_right)
        line_top = gmsh.model.occ.addLine(point_top_right, point_top_left)

        loop_water = gmsh.model.occ.addCurveLoop([line_left, curve_water_interface, line_right, line_top])
        surface_water = gmsh.model.occ.addPlaneSurface([loop_water])
        gmsh.model.occ.synchronize()

        if padding_type is None:
            surface_rectangle = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            fragment_result, fragment_map = gmsh.model.occ.fragment(
                [(2, surface_rectangle), (2, surface_water)], []
            )
            gmsh.model.occ.synchronize()
            water_surface_tags = [tag for dim, tag in fragment_map[1]]
            subsurface_tags = [
                tag for dim, tag in fragment_map[0] if tag not in water_surface_tags
            ]
            gmsh.model.addPhysicalGroup(2, water_surface_tags, name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, subsurface_tags, name="SubSurface")

        if padding_type == "rectangular":
            point_bottom_left = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            point_bottom_right = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            point_pad_top_left = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            point_pad_water_interface_left = gmsh.model.occ.addPoint(pad_x_min, z_water_left, 0.0)
            point_pad_left_at_domain_bottom = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            point_pad_top_right = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)
            point_pad_water_interface_right = gmsh.model.occ.addPoint(pad_x_max, z_water_right, 0.0)
            point_pad_right_at_domain_bottom = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            point_pad_bottom_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            point_pad_bottom_at_domain_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            point_pad_bottom_at_domain_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            point_pad_bottom_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)

            line_subsurface_right = gmsh.model.occ.addLine(points_water_interface[-1], point_bottom_right)
            line_subsurface_bottom = gmsh.model.occ.addLine(point_bottom_right, point_bottom_left)
            line_subsurface_left = gmsh.model.occ.addLine(point_bottom_left, points_water_interface[0])
            line_pad_water_top_right = gmsh.model.occ.addLine(point_top_right, point_pad_top_right)
            line_pad_water_right = gmsh.model.occ.addLine(point_pad_top_right, point_pad_water_interface_right)
            line_pad_water_bottom_right = gmsh.model.occ.addLine(point_pad_water_interface_right, points_water_interface[-1])
            line_pad_subsurface_right = gmsh.model.occ.addLine(point_pad_water_interface_right, point_pad_right_at_domain_bottom)
            line_pad_subsurface_bottom_right = gmsh.model.occ.addLine(point_pad_right_at_domain_bottom, point_bottom_right)
            line_pad_water_top_left = gmsh.model.occ.addLine(point_top_left, point_pad_top_left)
            line_pad_water_left = gmsh.model.occ.addLine(point_pad_top_left, point_pad_water_interface_left)
            line_pad_water_bottom_left = gmsh.model.occ.addLine(point_pad_water_interface_left, points_water_interface[0])
            line_pad_subsurface_left = gmsh.model.occ.addLine(point_pad_water_interface_left, point_pad_left_at_domain_bottom)
            line_pad_subsurface_bottom_left = gmsh.model.occ.addLine(point_pad_left_at_domain_bottom, point_bottom_left)
            line_pad_corner_left_outer = gmsh.model.occ.addLine(point_pad_left_at_domain_bottom, point_pad_bottom_left)
            line_pad_corner_left_bottom = gmsh.model.occ.addLine(point_pad_bottom_left, point_pad_bottom_at_domain_left)
            line_pad_corner_left_inner = gmsh.model.occ.addLine(point_pad_bottom_at_domain_left, point_bottom_left)
            line_pad_bottom_center = gmsh.model.occ.addLine(point_pad_bottom_at_domain_left, point_pad_bottom_at_domain_right)
            line_pad_corner_right_inner = gmsh.model.occ.addLine(point_pad_bottom_at_domain_right, point_bottom_right)
            line_pad_corner_right_bottom = gmsh.model.occ.addLine(point_pad_bottom_at_domain_right, point_pad_bottom_right)
            line_pad_corner_right_outer = gmsh.model.occ.addLine(point_pad_bottom_right, point_pad_right_at_domain_bottom)

            surface_subsurface = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([curve_water_interface, line_subsurface_right, line_subsurface_bottom, line_subsurface_left])])
            surface_pad_top_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_right, line_pad_water_top_right, line_pad_water_right, line_pad_water_bottom_right])])
            surface_pad_middle_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_pad_water_bottom_right, line_pad_subsurface_right, line_pad_subsurface_bottom_right, -line_subsurface_right])])
            surface_pad_top_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_pad_water_top_left, line_pad_water_left, line_pad_water_bottom_left, -line_left])])
            surface_pad_middle_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_pad_water_bottom_left, line_pad_subsurface_left, line_pad_subsurface_bottom_left, line_subsurface_left])])
            surface_pad_bottom_left_corner = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_pad_subsurface_bottom_left, line_pad_corner_left_outer, line_pad_corner_left_bottom, line_pad_corner_left_inner])])
            surface_pad_bottom_center = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_pad_corner_left_inner, line_pad_bottom_center, line_pad_corner_right_inner, line_subsurface_bottom])])
            surface_pad_bottom_right_corner = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_pad_corner_right_inner, line_pad_corner_right_bottom, line_pad_corner_right_outer, line_pad_subsurface_bottom_right])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surface_water], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surface_subsurface], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [
                    surface_pad_top_right,
                    surface_pad_middle_right,
                    surface_pad_top_left,
                    surface_pad_middle_left,
                    surface_pad_bottom_left_corner,
                    surface_pad_bottom_center,
                    surface_pad_bottom_right_corner,
                ],
                name="Padding",
            )

        if padding_type == "hyperelliptical":

            point_bottom_left = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)
            point_bottom_right = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            z_subsurface_middle_left, z_subsurface_middle_right = (z_water_left + depth_z) / 2.0, (z_water_right + depth_z) / 2.0
            x_bottom_middle_left, x_bottom_middle_right = length_x * 0.25, length_x * 0.75

            point_subsurface_middle_left = gmsh.model.occ.addPoint(0.0, z_subsurface_middle_left, 0.0)
            point_subsurface_middle_right = gmsh.model.occ.addPoint(length_x, z_subsurface_middle_right, 0.0)
            point_bottom_middle_left = gmsh.model.occ.addPoint(x_bottom_middle_left, depth_z, 0.0)
            point_bottom_middle_right = gmsh.model.occ.addPoint(x_bottom_middle_right, depth_z, 0.0)

            x_pad_top_left, z_pad_top_left = intersect(0.0, 0.0, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_pad_water_interface_left, z_pad_water_interface_left = intersect(0.0, z_water_left, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_pad_middle_left, z_pad_middle_left = intersect(0.0, z_subsurface_middle_left, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_left_diagonal, z_pad_bottom_left_diagonal = intersect(0.0, depth_z, -1, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_middle_left, z_pad_bottom_middle_left = intersect(x_bottom_middle_left, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_middle_right, z_pad_bottom_middle_right = intersect(x_bottom_middle_right, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_right_diagonal, z_pad_bottom_right_diagonal = intersect(length_x, depth_z, 1, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_middle_right, z_pad_middle_right = intersect(length_x, z_subsurface_middle_right, 1, 0, xc, zc, a_val, b_val, hyper_n)
            x_pad_water_interface_right, z_pad_water_interface_right = intersect(length_x, z_water_right, 1, 0, xc, zc, a_val, b_val, hyper_n)
            x_pad_top_right, z_pad_top_right = intersect(length_x, 0.0, 1, 0, xc, zc, a_val, b_val, hyper_n)

            point_pad_top_left = gmsh.model.occ.addPoint(x_pad_top_left, z_pad_top_left, 0.0)
            point_pad_water_interface_left = gmsh.model.occ.addPoint(x_pad_water_interface_left, z_pad_water_interface_left, 0.0)
            point_pad_middle_left = gmsh.model.occ.addPoint(x_pad_middle_left, z_pad_middle_left, 0.0)
            point_pad_bottom_left_diagonal = gmsh.model.occ.addPoint(x_pad_bottom_left_diagonal, z_pad_bottom_left_diagonal, 0.0)
            point_pad_bottom_middle_left = gmsh.model.occ.addPoint(x_pad_bottom_middle_left, z_pad_bottom_middle_left, 0.0)
            point_pad_bottom_middle_right = gmsh.model.occ.addPoint(x_pad_bottom_middle_right, z_pad_bottom_middle_right, 0.0)
            point_pad_bottom_right_diagonal = gmsh.model.occ.addPoint(x_pad_bottom_right_diagonal, z_pad_bottom_right_diagonal, 0.0)
            point_pad_middle_right = gmsh.model.occ.addPoint(x_pad_middle_right, z_pad_middle_right, 0.0)
            point_pad_water_interface_right = gmsh.model.occ.addPoint(x_pad_water_interface_right, z_pad_water_interface_right, 0.0)
            point_pad_top_right = gmsh.model.occ.addPoint(x_pad_top_right, z_pad_top_right, 0.0)

            line_subsurface_right_upper = gmsh.model.occ.addLine(points_water_interface[-1], point_subsurface_middle_right)
            line_subsurface_right_lower = gmsh.model.occ.addLine(point_subsurface_middle_right, point_bottom_right)
            line_subsurface_bottom_right = gmsh.model.occ.addLine(point_bottom_right, point_bottom_middle_right)
            line_subsurface_bottom_center = gmsh.model.occ.addLine(point_bottom_middle_right, point_bottom_middle_left)
            line_subsurface_bottom_left = gmsh.model.occ.addLine(point_bottom_middle_left, point_bottom_left)
            line_subsurface_left_lower = gmsh.model.occ.addLine(point_bottom_left, point_subsurface_middle_left)
            line_subsurface_left_upper = gmsh.model.occ.addLine(point_subsurface_middle_left, points_water_interface[0])

            ray_top_left = gmsh.model.occ.addLine(point_top_left, point_pad_top_left)
            ray_water_interface_left = gmsh.model.occ.addLine(points_water_interface[0], point_pad_water_interface_left)
            ray_middle_left = gmsh.model.occ.addLine(point_subsurface_middle_left, point_pad_middle_left)
            ray_bottom_left_diagonal = gmsh.model.occ.addLine(point_bottom_left, point_pad_bottom_left_diagonal)
            ray_bottom_middle_left = gmsh.model.occ.addLine(point_bottom_middle_left, point_pad_bottom_middle_left)
            ray_bottom_middle_right = gmsh.model.occ.addLine(point_bottom_middle_right, point_pad_bottom_middle_right)
            ray_bottom_right_diagonal = gmsh.model.occ.addLine(point_bottom_right, point_pad_bottom_right_diagonal)
            ray_middle_right = gmsh.model.occ.addLine(point_subsurface_middle_right, point_pad_middle_right)
            ray_water_interface_right = gmsh.model.occ.addLine(points_water_interface[-1], point_pad_water_interface_right)
            ray_top_right = gmsh.model.occ.addLine(point_top_right, point_pad_top_right)

            arc_pad_water_left = make_arc(point_pad_top_left, point_pad_water_interface_left, x_pad_top_left, z_pad_top_left,
                                          x_pad_water_interface_left, z_pad_water_interface_left, xc, zc, a_val, b_val, hyper_n)
            arc_pad_left_upper = make_arc(point_pad_water_interface_left, point_pad_middle_left, x_pad_water_interface_left, z_pad_water_interface_left,
                                          x_pad_middle_left, z_pad_middle_left, xc, zc, a_val, b_val, hyper_n)
            arc_pad_left_lower = make_arc(point_pad_middle_left, point_pad_bottom_left_diagonal, x_pad_middle_left, z_pad_middle_left,
                                          x_pad_bottom_left_diagonal, z_pad_bottom_left_diagonal, xc, zc, a_val, b_val, hyper_n)
            arc_pad_bottom_left_diagonal_to_middle = make_arc(point_pad_bottom_left_diagonal, point_pad_bottom_middle_left, x_pad_bottom_left_diagonal, z_pad_bottom_left_diagonal,
                                                              x_pad_bottom_middle_left, z_pad_bottom_middle_left, xc, zc, a_val, b_val, hyper_n)
            arc_pad_bottom_center = make_arc(point_pad_bottom_middle_left, point_pad_bottom_middle_right, x_pad_bottom_middle_left, z_pad_bottom_middle_left,
                                             x_pad_bottom_middle_right, z_pad_bottom_middle_right, xc, zc, a_val, b_val, hyper_n)
            arc_pad_bottom_middle_to_right_diagonal = make_arc(point_pad_bottom_middle_right, point_pad_bottom_right_diagonal, x_pad_bottom_middle_right, z_pad_bottom_middle_right,
                                                               x_pad_bottom_right_diagonal, z_pad_bottom_right_diagonal, xc, zc, a_val, b_val, hyper_n)
            arc_pad_right_lower = make_arc(point_pad_bottom_right_diagonal, point_pad_middle_right, x_pad_bottom_right_diagonal, z_pad_bottom_right_diagonal,
                                           x_pad_middle_right, z_pad_middle_right, xc, zc, a_val, b_val, hyper_n)
            arc_pad_right_upper = make_arc(point_pad_middle_right, point_pad_water_interface_right, x_pad_middle_right, z_pad_middle_right,
                                           x_pad_water_interface_right, z_pad_water_interface_right, xc, zc, a_val, b_val, hyper_n)
            arc_pad_water_right = make_arc(point_pad_water_interface_right, point_pad_top_right, x_pad_water_interface_right, z_pad_water_interface_right,
                                           x_pad_top_right, z_pad_top_right, xc, zc, a_val, b_val, hyper_n)

            surface_subsurface = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([curve_water_interface, line_subsurface_right_upper, line_subsurface_right_lower, line_subsurface_bottom_right, line_subsurface_bottom_center, line_subsurface_bottom_left, line_subsurface_left_lower, line_subsurface_left_upper])])
            surface_pad_top_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_left, ray_water_interface_left, -arc_pad_water_left, -ray_top_left])])
            surface_pad_left_upper = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_left_upper, ray_middle_left, -arc_pad_left_upper, -ray_water_interface_left])])
            surface_pad_left_lower = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_left_lower, ray_bottom_left_diagonal, -arc_pad_left_lower, -ray_middle_left])])
            surface_pad_bottom_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_bottom_left, ray_bottom_middle_left, -arc_pad_bottom_left_diagonal_to_middle, -ray_bottom_left_diagonal])])
            surface_pad_bottom_center = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_bottom_center, ray_bottom_middle_right, -arc_pad_bottom_center, -ray_bottom_middle_left])])
            surface_pad_bottom_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_bottom_right, ray_bottom_right_diagonal, -arc_pad_bottom_middle_to_right_diagonal, -ray_bottom_middle_right])])
            surface_pad_right_lower = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_right_lower, ray_middle_right, -arc_pad_right_lower, -ray_bottom_right_diagonal])])
            surface_pad_right_upper = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([-line_subsurface_right_upper, ray_water_interface_right, -arc_pad_right_upper, -ray_middle_right])])
            surface_pad_top_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_right, ray_top_right, -arc_pad_water_right, -ray_water_interface_right])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surface_water], name="WaterSurface")
            gmsh.model.addPhysicalGroup(2, [surface_subsurface], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [
                    surface_pad_top_left,
                    surface_pad_left_upper,
                    surface_pad_left_lower,
                    surface_pad_bottom_left,
                    surface_pad_bottom_center,
                    surface_pad_bottom_right,
                    surface_pad_right_lower,
                    surface_pad_right_upper,
                    surface_pad_top_right,
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
                len_z_rock_upper, len_z_rock_lower = abs(z_subsurface_middle_left - z_water_left), abs(
                    depth_z - z_subsurface_middle_left
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
                    ray_top_left,
                    ray_water_interface_left,
                    ray_middle_left,
                    ray_bottom_left_diagonal,
                    ray_bottom_middle_left,
                    ray_bottom_middle_right,
                    ray_bottom_right_diagonal,
                    ray_middle_right,
                    ray_water_interface_right,
                    ray_top_right,
                ]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_radial)
                for curve in [line_left, line_right, arc_pad_water_left, arc_pad_water_right]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z_water)
                for curve in [line_subsurface_left_upper, line_subsurface_right_upper, arc_pad_left_upper, arc_pad_right_upper]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z_rock_upper)
                for curve in [line_subsurface_left_lower, line_subsurface_right_lower, arc_pad_left_lower, arc_pad_right_lower]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z_rock_lower)
                for curve in [line_subsurface_bottom_left, arc_pad_bottom_left_diagonal_to_middle]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_left)
                for curve in [line_subsurface_bottom_center, arc_pad_bottom_center]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_mid)
                for curve in [line_subsurface_bottom_right, arc_pad_bottom_middle_to_right_diagonal]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_right)
                for curve in [line_top, curve_water_interface]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_total)

                padding_surfaces = [
                    surface_pad_top_left,
                    surface_pad_left_upper,
                    surface_pad_left_lower,
                    surface_pad_bottom_left,
                    surface_pad_bottom_center,
                    surface_pad_bottom_right,
                    surface_pad_right_lower,
                    surface_pad_right_upper,
                    surface_pad_top_right,
                ]
                for surface in padding_surfaces:
                    gmsh.model.mesh.setTransfiniteSurface(surface)
                    gmsh.model.mesh.setRecombine(2, surface)

                gmsh.model.mesh.setTransfiniteSurface(
                    surface_water, cornerTags=[point_top_left, point_top_right, points_water_interface[-1], points_water_interface[0]]
                )
                gmsh.model.mesh.setRecombine(2, surface_water)
                gmsh.model.mesh.setTransfiniteSurface(
                    surface_subsurface, cornerTags=[points_water_interface[0], points_water_interface[-1], point_bottom_right, point_bottom_left]
                )
                gmsh.model.mesh.setRecombine(2, surface_subsurface)

    if not water_interface:
        if padding_type is None:
            surface_rectangle = gmsh.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface_rectangle], name="SubSurface")

        if padding_type == "rectangular":
            point_top_left = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            point_top_right = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            point_bottom_right = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            point_bottom_left = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            point_pad_top_left = gmsh.model.occ.addPoint(pad_x_min, 0.0, 0.0)
            point_pad_left_at_domain_bottom = gmsh.model.occ.addPoint(pad_x_min, depth_z, 0.0)
            point_pad_bottom_left = gmsh.model.occ.addPoint(pad_x_min, pad_z_min, 0.0)
            point_pad_bottom_at_domain_left = gmsh.model.occ.addPoint(0.0, pad_z_min, 0.0)
            point_pad_bottom_at_domain_right = gmsh.model.occ.addPoint(length_x, pad_z_min, 0.0)
            point_pad_bottom_right = gmsh.model.occ.addPoint(pad_x_max, pad_z_min, 0.0)
            point_pad_right_at_domain_bottom = gmsh.model.occ.addPoint(pad_x_max, depth_z, 0.0)
            point_pad_top_right = gmsh.model.occ.addPoint(pad_x_max, 0.0, 0.0)

            line_top = gmsh.model.occ.addLine(point_top_left, point_top_right)
            line_right = gmsh.model.occ.addLine(point_top_right, point_bottom_right)
            line_bottom = gmsh.model.occ.addLine(point_bottom_right, point_bottom_left)
            line_left = gmsh.model.occ.addLine(point_bottom_left, point_top_left)

            line_pad_top_left = gmsh.model.occ.addLine(point_pad_top_left, point_top_left)
            line_pad_left = gmsh.model.occ.addLine(point_pad_left_at_domain_bottom, point_pad_top_left)
            line_pad_left_at_domain_bottom = gmsh.model.occ.addLine(point_bottom_left, point_pad_left_at_domain_bottom)

            line_pad_corner_left_outer = gmsh.model.occ.addLine(point_pad_bottom_left, point_pad_left_at_domain_bottom)
            line_pad_corner_left_bottom = gmsh.model.occ.addLine(point_pad_bottom_at_domain_left, point_pad_bottom_left)
            line_pad_corner_left_inner = gmsh.model.occ.addLine(point_bottom_left, point_pad_bottom_at_domain_left)

            line_pad_bottom_center = gmsh.model.occ.addLine(point_pad_bottom_at_domain_right, point_pad_bottom_at_domain_left)
            line_pad_corner_right_inner = gmsh.model.occ.addLine(point_bottom_right, point_pad_bottom_at_domain_right)

            line_pad_corner_right_bottom = gmsh.model.occ.addLine(point_pad_bottom_right, point_pad_bottom_at_domain_right)
            line_pad_corner_right_outer = gmsh.model.occ.addLine(point_pad_right_at_domain_bottom, point_pad_bottom_right)

            line_pad_right_at_domain_bottom = gmsh.model.occ.addLine(point_pad_right_at_domain_bottom, point_bottom_right)
            line_pad_right = gmsh.model.occ.addLine(point_pad_top_right, point_pad_right_at_domain_bottom)
            line_pad_top_right = gmsh.model.occ.addLine(point_top_right, point_pad_top_right)

            loop_domain = gmsh.model.occ.addCurveLoop([line_top, line_right, line_bottom, line_left])
            surface_domain = gmsh.model.occ.addPlaneSurface([loop_domain])
            loop_pad_left = gmsh.model.occ.addCurveLoop(
                [line_pad_top_left, -line_left, -line_pad_left_at_domain_bottom, line_pad_left])
            surface_pad_left = gmsh.model.occ.addPlaneSurface([loop_pad_left])
            loop_pad_bottom_left_corner = gmsh.model.occ.addCurveLoop(
                [line_pad_left_at_domain_bottom, -line_pad_corner_left_outer, -line_pad_corner_left_bottom, -line_pad_corner_left_inner])
            surface_pad_bottom_left_corner = gmsh.model.occ.addPlaneSurface([loop_pad_bottom_left_corner])
            loop_pad_bottom_center = gmsh.model.occ.addCurveLoop(
                [line_pad_corner_left_inner, -line_pad_bottom_center, -line_pad_corner_right_inner, line_bottom])
            surface_pad_bottom_center = gmsh.model.occ.addPlaneSurface([loop_pad_bottom_center])
            loop_pad_bottom_right_corner = gmsh.model.occ.addCurveLoop(
                [line_pad_corner_right_inner, -line_pad_corner_right_bottom, -line_pad_corner_right_outer, -line_pad_right_at_domain_bottom])
            surface_pad_bottom_right_corner = gmsh.model.occ.addPlaneSurface([loop_pad_bottom_right_corner])
            loop_pad_right = gmsh.model.occ.addCurveLoop(
                [line_right, -line_pad_right_at_domain_bottom, -line_pad_right, -line_pad_top_right])
            surface_pad_right = gmsh.model.occ.addPlaneSurface([loop_pad_right])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surface_domain], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [surface_pad_left, surface_pad_bottom_left_corner, surface_pad_bottom_center, surface_pad_bottom_right_corner, surface_pad_right],
                name="Padding",
            )

        if padding_type == "hyperelliptical":

            point_top_left = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
            point_top_right = gmsh.model.occ.addPoint(length_x, 0.0, 0.0)
            point_bottom_right = gmsh.model.occ.addPoint(length_x, depth_z, 0.0)
            point_bottom_left = gmsh.model.occ.addPoint(0.0, depth_z, 0.0)

            x_bottom_middle_left, x_bottom_middle_right = length_x * 0.25, length_x * 0.75
            point_bottom_middle_left = gmsh.model.occ.addPoint(x_bottom_middle_left, depth_z, 0.0)
            point_bottom_middle_right = gmsh.model.occ.addPoint(x_bottom_middle_right, depth_z, 0.0)

            x_pad_top_left, z_pad_top_left = intersect(0.0, 0.0, -1, 0, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_left, z_pad_bottom_left = intersect(0.0, depth_z, -1, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_middle_left, z_pad_bottom_middle_left = intersect(x_bottom_middle_left, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_middle_right, z_pad_bottom_middle_right = intersect(x_bottom_middle_right, depth_z, 0, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_bottom_right, z_pad_bottom_right = intersect(length_x, depth_z, 1, -1, xc, zc, a_val, b_val, hyper_n)
            x_pad_top_right, z_pad_top_right = intersect(length_x, 0.0, 1, 0, xc, zc, a_val, b_val, hyper_n)

            point_pad_top_left = gmsh.model.occ.addPoint(x_pad_top_left, z_pad_top_left, 0.0)
            point_pad_bottom_left = gmsh.model.occ.addPoint(x_pad_bottom_left, z_pad_bottom_left, 0.0)
            point_pad_bottom_middle_left = gmsh.model.occ.addPoint(x_pad_bottom_middle_left, z_pad_bottom_middle_left, 0.0)
            point_pad_bottom_middle_right = gmsh.model.occ.addPoint(x_pad_bottom_middle_right, z_pad_bottom_middle_right, 0.0)
            point_pad_bottom_right = gmsh.model.occ.addPoint(x_pad_bottom_right, z_pad_bottom_right, 0.0)
            point_pad_top_right = gmsh.model.occ.addPoint(x_pad_top_right, z_pad_top_right, 0.0)

            line_top = gmsh.model.occ.addLine(point_top_left, point_top_right)
            line_right = gmsh.model.occ.addLine(point_top_right, point_bottom_right)
            line_bottom_right = gmsh.model.occ.addLine(point_bottom_right, point_bottom_middle_right)
            line_bottom_center = gmsh.model.occ.addLine(point_bottom_middle_right, point_bottom_middle_left)
            line_bottom_left = gmsh.model.occ.addLine(point_bottom_middle_left, point_bottom_left)
            line_left = gmsh.model.occ.addLine(point_bottom_left, point_top_left)

            ray_top_left = gmsh.model.occ.addLine(point_top_left, point_pad_top_left)
            ray_bottom_left = gmsh.model.occ.addLine(point_bottom_left, point_pad_bottom_left)
            ray_bottom_middle_left = gmsh.model.occ.addLine(point_bottom_middle_left, point_pad_bottom_middle_left)
            ray_bottom_middle_right = gmsh.model.occ.addLine(point_bottom_middle_right, point_pad_bottom_middle_right)
            ray_bottom_right = gmsh.model.occ.addLine(point_bottom_right, point_pad_bottom_right)
            ray_top_right = gmsh.model.occ.addLine(point_top_right, point_pad_top_right)

            arc_pad_left = make_arc(point_pad_top_left, point_pad_bottom_left, x_pad_top_left, z_pad_top_left,
                                    x_pad_bottom_left, z_pad_bottom_left, xc, zc, a_val, b_val, hyper_n)
            arc_pad_bottom_left_to_middle = make_arc(point_pad_bottom_left, point_pad_bottom_middle_left, x_pad_bottom_left, z_pad_bottom_left,
                                                     x_pad_bottom_middle_left, z_pad_bottom_middle_left, xc, zc, a_val, b_val, hyper_n)
            arc_pad_bottom_center = make_arc(point_pad_bottom_middle_left, point_pad_bottom_middle_right, x_pad_bottom_middle_left, z_pad_bottom_middle_left,
                                             x_pad_bottom_middle_right, z_pad_bottom_middle_right, xc, zc, a_val, b_val, hyper_n)
            arc_pad_bottom_middle_to_right = make_arc(point_pad_bottom_middle_right, point_pad_bottom_right, x_pad_bottom_middle_right, z_pad_bottom_middle_right,
                                                      x_pad_bottom_right, z_pad_bottom_right, xc, zc, a_val, b_val, hyper_n)
            arc_pad_right = make_arc(point_pad_bottom_right, point_pad_top_right, x_pad_bottom_right, z_pad_bottom_right,
                                     x_pad_top_right, z_pad_top_right, xc, zc, a_val, b_val, hyper_n)

            surface_subsurface = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_top, line_right, line_bottom_right, line_bottom_center, line_bottom_left, line_left])])
            surface_pad_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([ray_bottom_left, -arc_pad_left, -ray_top_left, -line_left])])
            surface_pad_bottom_left = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_bottom_left, ray_bottom_left, arc_pad_bottom_left_to_middle, -ray_bottom_middle_left])])
            surface_pad_bottom_center = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_bottom_center, ray_bottom_middle_left, arc_pad_bottom_center, -ray_bottom_middle_right])])
            surface_pad_bottom_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_bottom_right, ray_bottom_middle_right, arc_pad_bottom_middle_to_right, -ray_bottom_right])])
            surface_pad_right = gmsh.model.occ.addPlaneSurface(
                [gmsh.model.occ.addCurveLoop([line_right, ray_bottom_right, arc_pad_right, -ray_top_right])])

            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            gmsh.model.addPhysicalGroup(2, [surface_subsurface], name="SubSurface")
            gmsh.model.addPhysicalGroup(
                2,
                [
                    surface_pad_left,
                    surface_pad_bottom_left,
                    surface_pad_bottom_center,
                    surface_pad_bottom_right,
                    surface_pad_right,
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

                for curve in [ray_top_left, ray_bottom_left, ray_bottom_middle_left, ray_bottom_middle_right, ray_bottom_right, ray_top_right]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_radial)

                for curve in [line_left, line_right, arc_pad_left, arc_pad_right]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_z)

                for curve in [line_bottom_left, arc_pad_bottom_left_to_middle]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_left)
                for curve in [line_bottom_center, arc_pad_bottom_center]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_mid)
                for curve in [line_bottom_right, arc_pad_bottom_middle_to_right]:
                    gmsh.model.mesh.setTransfiniteCurve(curve, n_x_right)

                gmsh.model.mesh.setTransfiniteCurve(line_top, n_x_total)

                gmsh.model.mesh.setTransfiniteSurface(
                    surface_pad_left, cornerTags=[point_bottom_left, point_pad_bottom_left, point_pad_top_left, point_top_left]
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surface_pad_bottom_left,
                    cornerTags=[point_bottom_middle_left, point_bottom_left, point_pad_bottom_left, point_pad_bottom_middle_left],
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surface_pad_bottom_center,
                    cornerTags=[point_bottom_middle_right, point_bottom_middle_left, point_pad_bottom_middle_left, point_pad_bottom_middle_right],
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surface_pad_bottom_right,
                    cornerTags=[point_bottom_right, point_bottom_middle_right, point_pad_bottom_middle_right, point_pad_bottom_right],
                )
                gmsh.model.mesh.setTransfiniteSurface(
                    surface_pad_right, cornerTags=[point_top_right, point_bottom_right, point_pad_bottom_right, point_pad_top_right]
                )

                padding_surfaces = [
                    surface_pad_left,
                    surface_pad_bottom_left,
                    surface_pad_bottom_center,
                    surface_pad_bottom_right,
                    surface_pad_right,
                ]
                for surface in padding_surfaces:
                    gmsh.model.mesh.setRecombine(2, surface)

                gmsh.model.mesh.setTransfiniteSurface(
                    surface_subsurface, cornerTags=[point_top_left, point_top_right, point_bottom_right, point_bottom_left]
                )
                gmsh.model.mesh.setRecombine(2, surface_subsurface)

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
        "surface_water": locals().get("surface_water"),
        "surface_pad_top_left": locals().get("surface_pad_top_left"),
        "surface_pad_top_right": locals().get("surface_pad_top_right"),
        "surface_pad_left": locals().get("surface_pad_left"),
        # Arcs
        "arc_pad_left_upper": locals().get("arc_pad_left_upper"),
        "arc_pad_left_lower": locals().get("arc_pad_left_lower"),
        "arc_pad_bottom_left_diagonal_to_middle": locals().get("arc_pad_bottom_left_diagonal_to_middle"),
        "arc_pad_bottom_center": locals().get("arc_pad_bottom_center"),
        "arc_pad_bottom_middle_to_right_diagonal": locals().get("arc_pad_bottom_middle_to_right_diagonal"),
        "arc_pad_right_lower": locals().get("arc_pad_right_lower"),
        "arc_pad_right_upper": locals().get("arc_pad_right_upper"),
        "arc_pad_left": locals().get("arc_pad_left"),
        "arc_pad_bottom_left_to_middle": locals().get("arc_pad_bottom_left_to_middle"),
        "arc_pad_bottom_middle_to_right": locals().get("arc_pad_bottom_middle_to_right"),
        "arc_pad_right": locals().get("arc_pad_right"),
        # Lines / Rock Boundaries
        "line_subsurface_left_lower": locals().get("line_subsurface_left_lower"),
        "line_subsurface_left_upper": locals().get("line_subsurface_left_upper"),
        "line_subsurface_right_lower": locals().get("line_subsurface_right_lower"),
        "line_subsurface_right_upper": locals().get("line_subsurface_right_upper"),
        "line_subsurface_bottom_left": locals().get("line_subsurface_bottom_left"),
        "line_subsurface_bottom_center": locals().get("line_subsurface_bottom_center"),
        "line_subsurface_bottom_right": locals().get("line_subsurface_bottom_right"),
        "line_left": locals().get("line_left"),
        "line_right": locals().get("line_right"),
        "line_bottom_left": locals().get("line_bottom_left"),
        "line_bottom_center": locals().get("line_bottom_center"),
        "line_bottom_right": locals().get("line_bottom_right"),
        # Rays
        "ray_bottom_left_diagonal": locals().get("ray_bottom_left_diagonal"),
        "ray_bottom_right_diagonal": locals().get("ray_bottom_right_diagonal"),
        "ray_top_left": locals().get("ray_top_left"),
        "ray_bottom_left": locals().get("ray_bottom_left"),
        "ray_bottom_right": locals().get("ray_bottom_right"),
        "ray_top_right": locals().get("ray_top_right"),
        "x_pad_top_left": locals().get("x_pad_top_left"),
        "x_pad_top_right": locals().get("x_pad_top_right"),
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
            locked_surfaces = [
                geom_params.get("surface_water"),
                geom_params.get("surface_pad_top_left"),
                geom_params.get("surface_pad_top_right"),
            ]
            locked_surface_nodes = get_nodes(2, locked_surfaces)
            outer_middle_arcs = [
                geom_params.get(k)
                for k in [
                    "arc_pad_left_upper",
                    "arc_pad_left_lower",
                    "arc_pad_bottom_left_diagonal_to_middle",
                    "arc_pad_bottom_center",
                    "arc_pad_bottom_middle_to_right_diagonal",
                    "arc_pad_right_lower",
                    "arc_pad_right_upper",
                ]
            ]
            z_slide_curves = [
                geom_params.get(k)
                for k in [
                    "line_subsurface_left_lower",
                    "line_subsurface_left_upper",
                    "line_subsurface_right_lower",
                    "line_subsurface_right_upper",
                ]
            ]
            x_slide_curves = [
                geom_params.get(k)
                for k in ["line_subsurface_bottom_left", "line_subsurface_bottom_center", "line_subsurface_bottom_right"]
            ]
            diagonal_rays = [geom_params.get("ray_bottom_left_diagonal"), geom_params.get("ray_bottom_right_diagonal")]
        else:
            locked_surface_nodes = set()

            outer_middle_arcs = [
                geom_params.get(k)
                for k in [
                    "arc_pad_left",
                    "arc_pad_bottom_left_to_middle",
                    "arc_pad_bottom_center",
                    "arc_pad_bottom_middle_to_right",
                    "arc_pad_right",
                ]
            ]

            z_slide_curves = [
                geom_params.get("line_left"),
                geom_params.get("line_right"),
            ]
            x_slide_curves = [
                geom_params.get("line_bottom_left"),
                geom_params.get("line_bottom_center"),
                geom_params.get("line_bottom_right"),
            ]
            diagonal_rays = [geom_params.get("ray_bottom_left"), geom_params.get("ray_bottom_right")]

        outer_arc_nodes = get_nodes(1, outer_middle_arcs)
        z_slide_nodes = get_nodes(1, z_slide_curves)
        x_slide_nodes = get_nodes(1, x_slide_curves)
        locked_diag_nodes = get_nodes(1, diagonal_rays)

        corner_indices = set()
        tol = 1e-3
        x_pad_top_left = geom_params.get("x_pad_top_left")
        x_pad_top_right = geom_params.get("x_pad_top_right")

        for i, point in enumerate(points_2d):
            x, z = point
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
                        x_pad_top_left is not None
                        and abs(x - x_pad_top_left) < tol
                        and abs(z - 0.0) < tol
                    )
                    or (
                        x_pad_top_right is not None
                        and abs(x - x_pad_top_right) < tol
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

        for i, point in enumerate(points_2d):
            x, z = point
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

        for i, point in enumerate(points_2d):
            x, z = point
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
            nx_grid, nz_grid = n_traces, n_samples
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
