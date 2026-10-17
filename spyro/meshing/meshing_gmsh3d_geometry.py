import numpy as np


def boundary_faces_of_volume(gmsh, vol_tag):
    """Return the unique surface tags bounding a Gmsh volume.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    vol_tag : int
        Tag of the three-dimensional Gmsh volume.

    Returns
    -------
    list of int
        Two-dimensional surface tags bounding the volume.
    """
    boundary = gmsh.model.getBoundary(
        [(3, int(vol_tag))],
        oriented=False,
        recursive=False,
        combined=True,
    )
    return [tag for dim, tag in boundary if dim == 2]


def _signed_power(x, p):
    """Evaluate a sign power used by the hyperellipsoid parameterization.

    Parameters
    ----------
    x : float
        Value whose signed power is evaluated.
    p : float
        Exponent applied while preserving the sign of the input.

    Returns
    -------
    float or numpy.ndarray
        Sign-preserving powered value.
    """
    if abs(x) < 1.0e-10:
        return 0.0
    return np.sign(x) * np.abs(x) ** p


def hyperellipsoid_point(u, v, a, b, c, n, xc, yc, zc):
    """Evaluate one Cartesian point on a centered hyperellipsoid.

    Parameters
    ----------
    u : float
        Azimuthal hyperellipsoid parameter.
    v : float
        Polar hyperellipsoid parameter.
    a : float
        Hyperellipsoid semi-axis in the x direction.
    b : float
        Hyperellipsoid semi-axis in the y direction.
    c : float
        Hyperellipsoid semi-axis in the z direction.
    n : float
        Hyperellipsoid shape exponent.
    xc : float
        Hyperellipsoid center coordinate in x.
    yc : float
        Hyperellipsoid center coordinate in y.
    zc : float
        Hyperellipsoid center coordinate in z.

    Returns
    -------
    tuple of float
        Cartesian ``(x, y, z)`` point on the hyperellipsoid.
    """
    cv, sv = np.cos(v), np.sin(v)
    cu, su = np.cos(u), np.sin(u)
    p = 2.0 / n

    return (
        xc + a * _signed_power(cv, p) * _signed_power(cu, p),
        yc + b * _signed_power(cv, p) * _signed_power(su, p),
        zc + c * _signed_power(sv, p),
    )


def create_closed_surface(
    gmsh,
    a,
    b,
    c,
    n,
    xc,
    yc,
    zc,
    point_func=hyperellipsoid_point,
    u_res=60,
    v_res=60,
    z_cut=0.0,
    *,
    comm,
    parallel_print,
):
    """Create a closed B-spline hyperellipsoid volume and optionally cut it at a z plane.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    a : float
        Hyperellipsoid semi-axis in the x direction.
    b : float
        Hyperellipsoid semi-axis in the y direction.
    c : float
        Hyperellipsoid semi-axis in the z direction.
    n : float
        Hyperellipsoid shape exponent.
    xc : float
        Hyperellipsoid center coordinate in x.
    yc : float
        Hyperellipsoid center coordinate in y.
    zc : float
        Hyperellipsoid center coordinate in z.
    point_func : callable
        Function used to evaluate hyperellipsoid surface points.
    u_res : int
        Number of azimuthal sampling intervals used by the B-spline surface.
    v_res : int
        Number of polar sampling points used by the B-spline surface.
    z_cut : float or None
        Z coordinate above which the generated volume is cut.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    int or None
        Generated Gmsh volume tag, or ``None`` if the z cut removes the volume.
    """
    occ = gmsh.model.occ
    point_tags = []

    for j in range(v_res):
        v = np.pi * (j / (v_res - 1) - 0.5)
        for i in range(u_res + 1):
            u = 0.0 if i == u_res else 2.0 * np.pi * i / u_res

            if j == 0:
                xyz = point_func(0.0, -np.pi / 2.0, a, b, c, n, xc, yc, zc)
            elif j == v_res - 1:
                xyz = point_func(0.0, np.pi / 2.0, a, b, c, n, xc, yc, zc)
            else:
                xyz = point_func(u, v, a, b, c, n, xc, yc, zc)

            point_tags.append(occ.addPoint(*xyz))

    try:
        surface_tag = occ.addBSplineSurface(
            pointTags=point_tags,
            numPointsU=u_res + 1,
            tag=-1,
            degreeU=min(3, u_res),
            degreeV=min(3, v_res - 1),
        )
        occ.synchronize()

        volume_tag = occ.addVolume([occ.addSurfaceLoop([surface_tag])])
        occ.synchronize()

        if volume_tag is not None and z_cut is not None:
            parallel_print(f"Applying z-cut at z = {z_cut}", comm=comm)
            domain_size = 2.0 * max(a, b, c)
            cutting_box = occ.addBox(
                xc - domain_size,
                yc - domain_size,
                z_cut,
                2.0 * domain_size,
                2.0 * domain_size,
                domain_size,
            )
            occ.synchronize()

            result = occ.cut(
                [(3, volume_tag)],
                [(3, cutting_box)],
                removeObject=True,
                removeTool=True,
            )

            if result[0]:
                volume_tag = result[0][0][1]
                parallel_print("Z-cut applied successfully", comm=comm)
            else:
                parallel_print("Warning: Z-cut removed entire volume", comm=comm)
                volume_tag = None

        return volume_tag

    except Exception as exc:
        raise RuntimeError("B-spline surface creation failed.") from exc


def create_hyperellipsoid_volume(
    gmsh,
    a=1.0,
    b=1.0,
    c=1.0,
    n=2.0,
    xc=0.0,
    yc=0.0,
    zc=0.0,
    *,
    comm,
    parallel_print,
):
    """Create a hyperellipsoid volume for three-dimensional padding.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    a : float
        Hyperellipsoid semi-axis in the x direction.
    b : float
        Hyperellipsoid semi-axis in the y direction.
    c : float
        Hyperellipsoid semi-axis in the z direction.
    n : float
        Hyperellipsoid shape exponent.
    xc : float
        Hyperellipsoid center coordinate in x.
    yc : float
        Hyperellipsoid center coordinate in y.
    zc : float
        Hyperellipsoid center coordinate in z.
    comm : mpi4py.MPI.Comm or None
        MPI communicator.
    parallel_print : callable
        Print function.

    Returns
    -------
    int or None
        Generated hyperellipsoid volume tag.
    """
    volume_tag = create_closed_surface(
        gmsh,
        a,
        b,
        c,
        n,
        xc,
        yc,
        zc,
        comm=comm,
        parallel_print=parallel_print,
    )
    if volume_tag:
        parallel_print("Successfully created volume using closed B-spline surface", comm=comm)
    return volume_tag


def _add_padding_box(occ, x0, y0, z0, lx, ly, lz):
    """Create a non-degenerate OpenCASCADE rectangular padding block.

    Parameters
    ----------
    occ : gmsh.model.occ
        Gmsh OpenCASCADE geometry interface.
    x0 : float
        Lower x coordinate of the box.
    y0 : float
        Lower y coordinate of the box.
    z0 : float
        Lower z coordinate of the box.
    lx : float
        Box extent in the x direction.
    ly : float
        Box extent in the y direction.
    lz : float
        Box extent in the z direction.

    Returns
    -------
    int or None
        Created volume tag, or ``None`` for a zero-thickness block.
    """
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        return None
    return occ.addBox(
        float(x0), float(y0), float(z0), float(lx), float(ly), float(lz)
    )


def create_rectangular_padding_boxes(
    gmsh, length_x, length_y, depth_z, padding_x, padding_y, padding_z
):
    """Create the central domain and rectangular side and bottom padding blocks.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.

    Returns
    -------
    tuple
        Central subsurface volume tag and list of padding-volume tags.
    """
    occ = gmsh.model.occ

    vol_subsoil = occ.addBox(
        0.0, 0.0, -abs(depth_z), length_x, length_y, abs(depth_z)
    )

    x_cells = [
        (-padding_x, padding_x),
        (0.0, length_x),
        (length_x, padding_x),
    ]
    y_cells = [
        (-padding_y, padding_y),
        (0.0, length_y),
        (length_y, padding_y),
    ]

    side_padding = [
        _add_padding_box(occ, x0, y0, -abs(depth_z), lx, ly, abs(depth_z))
        for ix, (x0, lx) in enumerate(x_cells)
        for iy, (y0, ly) in enumerate(y_cells)
        if (ix, iy) != (1, 1)
    ]

    bottom_z = -abs(depth_z) - padding_z
    bottom_padding = [
        _add_padding_box(occ, x0, y0, bottom_z, lx, ly, padding_z)
        for x0, lx in x_cells
        for y0, ly in y_cells
    ]

    pad_vols = [v for v in side_padding + bottom_padding if v is not None]
    return vol_subsoil, pad_vols


def generate_rectangular_padding_no_water(
    gmsh, length_x, length_y, depth_z, padding_x, padding_y, padding_z
):
    """Build the undelimited rectangular padded domain and its physical groups.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.

    Returns
    -------
    tuple
        Central subsurface volume tag and ``None`` for the water volume.
    """
    if min(length_x, length_y, abs(depth_z)) <= 0.0:
        raise ValueError("length_x, length_y, and depth_z must be positive.")
    if min(padding_x, padding_y, padding_z) < 0.0:
        raise ValueError("padding_x, padding_y, and padding_z cannot be negative.")

    occ = gmsh.model.occ
    vol_subsoil, pad_vols = create_rectangular_padding_boxes(
        gmsh, length_x, length_y, depth_z, padding_x, padding_y, padding_z
    )

    if pad_vols:
        _, fragment_map = occ.fragment(
            [(3, tag) for tag in [vol_subsoil] + pad_vols],
            [],
            removeObject=True,
            removeTool=False,
        )
        occ.synchronize()

        core_fragments = sorted({tag for dim, tag in fragment_map[0] if dim == 3})
        padding_fragments = sorted(
            {
                tag
                for mapping in fragment_map[1:]
                for dim, tag in mapping
                if dim == 3
            }
            - set(core_fragments)
        )

        if not core_fragments:
            raise RuntimeError(
                "The central water+subsurface rectangle was lost during "
                "the rectangular padding fragment operation."
            )

        vol_subsoil = core_fragments[0]
        pad_vols = padding_fragments
    else:
        occ.synchronize()
        core_fragments = [vol_subsoil]

    gmsh.model.addPhysicalGroup(3, core_fragments, name="SubsurfaceAndWater")
    if pad_vols:
        gmsh.model.addPhysicalGroup(3, pad_vols, name="Padding")

    return vol_subsoil, None


def find_outer_volume_face(gmsh, volume, axis, side):
    """Find the outermost boundary face of a volume along a Cartesian axis.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    volume : int
        Gmsh volume tag.
    axis : int
        Cartesian axis index: 0 for x, 1 for y, or 2 for z.
    side : int
        Side selector; positive chooses the maximum side and negative the minimum.

    Returns
    -------
    int
        Tag of the selected outer boundary face.
    """
    occ = gmsh.model.occ
    faces = [
        tag
        for dim, tag in gmsh.model.getBoundary(
            [(3, int(volume))],
            oriented=False,
            recursive=False,
        )
        if dim == 2
    ]

    if not faces:
        raise RuntimeError(f"Volume {volume} has no boundary surfaces.")

    centers = {
        face: occ.getCenterOfMass(2, face)[axis]
        for face in faces
    }
    return (max if side > 0 else min)(faces, key=centers.get)


def sweep_volume_faces(gmsh, volumes, axis, side, distance):
    """Extrude selected outer faces and map each source volume to its generated volume.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    volumes : sequence of int
        Gmsh volume tags to process.
    axis : int
        Cartesian axis index: 0 for x, 1 for y, or 2 for z.
    side : int
        Side selector; positive chooses the maximum side and negative the minimum.
    distance : float
        Extrusion distance.

    Returns
    -------
    dict
        Mapping from source volume tags to generated swept-volume tags.
    """
    occ = gmsh.model.occ
    volumes = list(
        dict.fromkeys(
            int(volume)
            for volume in volumes
            if volume is not None
        )
    )

    if not volumes or distance <= 0.0:
        return {}

    source_faces = {
        volume: find_outer_volume_face(gmsh, volume, axis, side)
        for volume in volumes
    }

    direction = [0.0, 0.0, 0.0]
    direction[axis] = float(side) * float(distance)

    result = occ.extrude(
        [(2, face) for face in source_faces.values()],
        *direction,
    )
    occ.synchronize()

    generated_volumes = [tag for dim, tag in result if dim == 3]
    if not generated_volumes:
        raise RuntimeError(
            "OCC extrusion generated no volumes "
            f"(axis={axis}, side={side}, distance={distance})."
        )

    generated_boundaries = {
        volume: {
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(3, volume)],
                oriented=False,
                recursive=False,
            )
            if dim == 2
        }
        for volume in generated_volumes
    }

    mapping = {}
    for source_volume, source_face in source_faces.items():
        matches = [
            generated_volume
            for generated_volume in generated_volumes
            if source_face in generated_boundaries[generated_volume]
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely identify swept volume.\n"
                f"  source volume = {source_volume}\n"
                f"  source face   = {source_face}\n"
                f"  axis          = {axis}\n"
                f"  side          = {side}\n"
                f"  matches       = {matches}"
            )

        mapping[source_volume] = matches[0]

    return mapping


def generate_structured_rectangular_padding_water(
    gmsh,
    vol_water,
    vol_subsoil,
    padding_x,
    padding_y,
    padding_z,
    *,
    comm,
    parallel_print,
):
    """Create conformal rectangular padding around delimited water and subsurface volumes.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    vol_water : int
        Tag of the central water volume.
    vol_subsoil : int
        Tag of the central subsurface volume.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    comm : mpi4py.MPI.Comm or None
        MPI communicator forwarded to rank-aware output.
    parallel_print : callable
        Rank-aware print function accepting a ``comm`` keyword argument.

    Returns
    -------
    tuple
        Water-volume tags, soil-volume tags, and padding-volume tags.

    Notes
    -----
    Padding is generated by OpenCASCADE face sweeps rather than Boolean fragmentation so
    that the structured multiblock topology remains suitable for hexahedral meshing.
    """
    parallel_print("Generating structured rectangular padding by OCC sweeps...", comm=comm)

    occ = gmsh.model.occ
    occ.synchronize()

    vol_water = int(vol_water)
    vol_subsoil = int(vol_subsoil)

    water = {(0, 0): vol_water}
    soil = {(0, 0): vol_subsoil}

    # X padding
    if padding_x > 0.0:
        for side in (-1, +1):
            mapping = sweep_volume_faces(
                gmsh,
                [vol_water, vol_subsoil],
                axis=0,
                side=side,
                distance=padding_x,
            )
            water[(side, 0)] = mapping[vol_water]
            soil[(side, 0)] = mapping[vol_subsoil]

    # Y padding
    if padding_y > 0.0:
        water_row = sorted(water.items(), key=lambda item: item[0][0])
        soil_row = sorted(soil.items(), key=lambda item: item[0][0])
        sources = (
            [volume for _, volume in water_row]
            + [volume for _, volume in soil_row]
        )

        for side in (-1, +1):
            mapping = sweep_volume_faces(
                gmsh,
                sources,
                axis=1,
                side=side,
                distance=padding_y,
            )

            for (xposition, _), volume in water_row:
                water[(xposition, side)] = mapping[volume]
            for (xposition, _), volume in soil_row:
                soil[(xposition, side)] = mapping[volume]

    # Bottom padding
    soil_xy = [
        volume
        for _, volume in sorted(
            soil.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]

    if padding_z > 0.0:
        bottom_map = sweep_volume_faces(
            gmsh,
            soil_xy,
            axis=2,
            side=-1,
            distance=padding_z,
        )
        soil_bottom = [bottom_map[volume] for volume in soil_xy]
    else:
        soil_bottom = []

    water_vols = [
        volume
        for _, volume in sorted(
            water.items(),
            key=lambda item: (item[0][1], item[0][0]),
        )
    ]
    soil_vols = soil_xy + soil_bottom

    pad_vols = (
        [volume for volume in water_vols if volume != vol_water]
        + [volume for volume in soil_vols if volume != vol_subsoil]
    )

    water_vols = list(dict.fromkeys(water_vols))
    soil_vols = list(dict.fromkeys(soil_vols))
    pad_vols = list(dict.fromkeys(pad_vols))

    gmsh.model.addPhysicalGroup(3, [vol_subsoil], name="Subsurface")
    gmsh.model.addPhysicalGroup(3, [vol_water], name="Water")
    if pad_vols:
        gmsh.model.addPhysicalGroup(3, pad_vols, name="Padding")
    gmsh.model.addPhysicalGroup(3, water_vols, name="Water_with_padding")
    gmsh.model.addPhysicalGroup(3, soil_vols, name="Subsurface_with_padding")

    occ.synchronize()

    return water_vols, soil_vols, pad_vols
