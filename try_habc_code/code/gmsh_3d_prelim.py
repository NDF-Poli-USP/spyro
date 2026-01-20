def mesh_data_3D(self, hyp_par):
    '''
    Generate mesh data for the hyperellipsoidal domain

    Parameters
    ----------
    hyp_par : `tuple`
        Hyperellipsoid parameters.
        Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
        - n_hyp : `float`
            Degree of the hyperellipsoid
        - surface : `float`
            Surface area of the full hyperellipsoid (3D)
        - a_hyp : `float`
            Hyperellipsoid semi-axis in direction x
        - b_hyp : `float`
            Hyperellipsoid semi-axis in direction z
        - c_hyp : `float`
            Hyperellipsoid semi-axis in direction y

    Returns
    -------
    rec_box : `tuple`
        Box defined by original domain dimensions
        Structure: (xmin, xmax, ymin, ymax, zmin, zmax)
    hyp_box : `tuple`
        Box that envelopes the hyperellipsoid
        Structure: (xmin, xmax, ymin, ymax, zmin, zmax)
    centroid : `tuple`
        Hyperellipsoid centroid according to reference system (z, x, y)
        Structure: (xc, yc, zc)
    semi_axes : `tuple`
        Hyperellipsoid semi-axes according to reference system (z, x, y)
        Structure: (a_hyp, c_hyp, b_hyp)
    resol : `int`
        Resolution for the generation of the hyperellipsoid surface
    '''

    # Domain dimensions
    Lx, Lz, Ly = self.dom_dim

    # Hyperellipsoid parameters
    n_hyp, surface, a_hyp, b_hyp, c_hyp = hyp_par

    # Box defined by original domain dimensions
    rec_box = (0., Lx, 0., Ly, -Lz, 0.)

    # Hyperellipsoid centroid
    xc = Lx / 2.
    yc = Ly / 2.
    zc = -Lz / 2.

    # Box that envelopes the hyperellipsoid
    hyp_xmin = xc - a_hyp
    hyp_xmax = xc + a_hyp
    hyp_ymin = yc - c_hyp
    hyp_ymax = yc + c_hyp
    hyp_zmin = zc - b_hyp
    hyp_zmax = zc + b_hyp
    hyp_box = (hyp_xmin, hyp_xmax, hyp_ymin, hyp_ymax, hyp_zmin, hyp_zmax)

    # Hyperellipsoid centroid according to reference system (x, y, z)
    centroid = (xc, yc, zc)

    # Hyperellipsoid semi-axes according to reference system (x, z, y)
    semi_axes = (a_hyp, c_hyp, b_hyp)

    # Expected elements in a full hyperellipsoid surface
    r_asp = max(semi_axes) / min(semi_axes)
    resol = int(np.ceil(surface * r_asp / ((2.5 * self.lmin)**2)) + 1)

    return rec_box, hyp_box, centroid, semi_axes, resol


def merge_mesh_3D(self, hyp_par):
    '''
    Build a merged mesh from a box mesh and a hyperellipsoidal mesh

    Parameters
    ----------
    hyp_par : `tuple`
        Hyperellipsoid parameters.
        Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
        - n_hyp : `float`
            Degree of the hyperellipsoid
        - surface : `float`
            Surface area of the full hyperellipsoid (3D)
        - a_hyp : `float`
            Hyperellipsoid semi-axis in direction x
        - b_hyp : `float`
            Hyperellipsoid semi-axis in direction z
        - c_hyp : `float`
            Hyperellipsoid semi-axis in direction y

    Returns
    -------
    final_mesh : `firedrake mesh`
        Merged final mesh
    '''

    # Hyperellipsoid degree
    n_hyp = hyp_par[0]

    # Get mesh data
    rec_box, hyp_box, centr, semi_axes, resol = self.mesh_data_3D(hyp_par)

    # Initialize Gmsh
    try:
        gmsh.finalize()

    except Exception as e:
        print(f"Finalization failed: {e}", flush=True)

    gmsh.initialize()
    # -  0: disables all output messages
    # -  1: minimal output
    # -  2: default verbosity
    # - 99: maximum verbosity
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("hyp_mesh_3D")

    # Create hyperellipsoid volume
    hyp_vol_tag = self.create_hyp_vol_3D(semi_axes, centr, n_hyp, resol)

    # Create box volume
    box_vol_tag = self.create_box_vol_3D(rec_box)

    # Fragment the geometries
    if not (hyp_vol_tag and box_vol_tag):
        print("Error Generating Merged Mesh", flush=True)

    try:
        print("Fragmenting Box and Hyperellipsoid", flush=True)

        # Fragment operation - create proper subdomain
        fragment_result = gmsh.model.occ.fragment(
            [(3, hyp_vol_tag), (3, box_vol_tag)],  # Object volumes
            [],  # (empty for self-fragmentation)
            removeObject=True, removeTool=False)
        gmsh.model.occ.synchronize()

        # Get all 3D entities and volume tags from the model
        volumes = gmsh.model.getEntities(3)
        vol_tags = [tag for dim, tag in volumes if dim == 3]
        box_vol = [vol_tags[0]] if len(vol_tags) > 0 else []
        hyp_vol = [vol_tags[1]] if len(vol_tags) > 1 else []

        # Create physical groups for the volumes
        box = gmsh.model.addPhysicalGroup(3, [vol_tags[0]], name="Box")
        hyp = gmsh.model.addPhysicalGroup(3, [vol_tags[1]], name="Hyp")

        # Set mesh size in the hypershape
        if hyp_vol:
            hyp_f = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(hyp_f, "VIn", 0.9 * self.lmin)
            gmsh.model.mesh.field.setNumbers(hyp_f, "VolumesList", hyp_vol)
            hyp_r = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(hyp_r, "InField", hyp_f)
            gmsh.model.mesh.field.setNumbers(hyp_r, "VolumesList", hyp_vol)
            field_list = [hyp_r]
            gmsh.model.mesh.field.setAsBackgroundMesh(field_list[0])

        # Free mesh in the volume mesh
        # gmsh.model.mesh.setSize(gmsh.model.getBoundary(
        #     [(3, vol_tags[1])], oriented=False,
        #     recursive=True), self.lmin)

        # Settings for the mesh generation
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        gmsh.option.setNumber("Mesh.SaveWithoutOrphans", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)
        self.report_quality(dim=3, quality_type=2)

        # Get mesh info
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, \
            element_node_tags = gmsh.model.mesh.getElements(3)
        total_nodes = len(node_tags)
        total_elements = sum(len(tags) for tags in element_tags)

        # Mesh data
        print(f"Mesh Created with {total_nodes} Nodes "
              f"and {total_elements} Volume Elements", flush=True)

        with NamedTemporaryFile(suffix='.msh') as tmp:

            # Save to temporary file
            gmsh.write(tmp.name)
            gmsh.clear()

            # Load mesh directly from temporary file
            q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
            final_mesh = fire.Mesh(
                tmp.name, distribution_parameters=q, comm=self.comm.comm)
        print("Merged Mesh Generated Successfully", flush=True)

        gmsh.finalize()

    except Exception as e:
        print(f"Error Generating Merged Mesh: {e}. Exiting.", flush=True)

    # Adjusting coordinates: Swap (x, y, z) -> (z, x ,y)
    final_mesh.coordinates.dat.data_with_halos[:, [0, 1, 2]] = \
        final_mesh.coordinates.dat.data_with_halos[:, [2, 0, 1]]

    return final_mesh


def mesh_data_3D(self, hyp_par):
    '''
    Generate mesh data for the hyperellipsoidal domain

    Parameters
    ----------
    hyp_par : `tuple`
        Hyperellipsoid parameters.
        Structure: (n_hyp, surface, a_hyp, b_hyp, c_hyp)
        - n_hyp : `float`
            Degree of the hyperellipsoid
        - surface : `float`
            Surface area of the full hyperellipsoid (3D)
        - a_hyp : `float`
            Hyperellipsoid semi-axis in direction x
        - b_hyp : `float`
            Hyperellipsoid semi-axis in direction z
        - c_hyp : `float`
            Hyperellipsoid semi-axis in direction y

    Returns
    -------
    rec_box : `tuple`
        Box defined by original domain dimensions
        Structure: (xmin, xmax, ymin, ymax, zmin, zmax)
    hyp_box : `tuple`
        Box that envelopes the hyperellipsoid
        Structure: (xmin, xmax, ymin, ymax, zmin, zmax)
    centroid : `tuple`
        Hyperellipsoid centroid according to reference system (z, x, y)
        Structure: (xc, yc, zc)
    semi_axes : `tuple`
        Hyperellipsoid semi-axes according to reference system (z, x, y)
        Structure: (a_hyp, c_hyp, b_hyp)
    resol : `int`
        Resolution for the generation of the hyperellipsoid surface
    '''

    # Domain dimensions
    Lx, Lz, Ly = self.dom_dim

    # Hyperellipsoid parameters
    n_hyp, surface, a_hyp, b_hyp, c_hyp = hyp_par

    # Box defined by original domain dimensions
    rec_box = (0., Lx, 0., Ly, -Lz, 0.)

    # Hyperellipsoid centroid
    xc = Lx / 2.
    yc = Ly / 2.
    zc = -Lz / 2.

    # Box that envelopes the hyperellipsoid
    hyp_xmin = xc - a_hyp
    hyp_xmax = xc + a_hyp
    hyp_ymin = yc - c_hyp
    hyp_ymax = yc + c_hyp
    hyp_zmin = zc - b_hyp
    hyp_zmax = zc + b_hyp
    hyp_box = (hyp_xmin, hyp_xmax, hyp_ymin, hyp_ymax, hyp_zmin, hyp_zmax)

    # Hyperellipsoid centroid according to reference system (x, y, z)
    centroid = (xc, yc, zc)

    # Hyperellipsoid semi-axes according to reference system (x, y, z)
    semi_axes = (a_hyp, c_hyp, b_hyp)

    # Expected elements in a full hyperellipsoid surface
    r_asp = max(semi_axes) / min(semi_axes)
    resol = int(np.ceil(surface * r_asp / ((2.5 * self.lmin)**2)) + 1)

    return rec_box, hyp_box, centroid, semi_axes, resol


@staticmethod
def create_hyp_pnt_3D(u, v, semi_axes, centroid, n):
    '''
    Create a point on the hyperellipsoid surface

    Parameters
    ----------
    u : `float`
        Longitude parameter [0, 2π]
    v : `float`
        Latitude parameter [-π/2, π/2]
    semi_axes : `tuple`
        Semi-axes of the hyperellipsoid (a, b, c)
    centroid : `tuple`
        Centroid of the hyperellipsoid (xc, yc, zc)
    n : `float`
        Degree of the hyperellipsoid

    Returns
    -------
    x, y, z : `float`
        Coordinates of the point on the hyperellipsoid surface
    '''

    # Hyperellipsoid semi-axes
    a, b, c = semi_axes

    # Hyperellipsoid centroid
    xc, yc, zc = centroid

    #  Trigonometric function evaluation with special cases
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)

    # Power and sign function
    def sign_power(x, p):
        return 0. if abs(x) < 1e-10 else np.sign(x) * (abs(x) ** p)

    # Calculate point relative to origin
    x = a * sign_power(cos_v, 2. / n) * sign_power(cos_u, 2. / n)
    y = b * sign_power(cos_v, 2. / n) * sign_power(sin_u, 2. / n)
    z = c * sign_power(sin_v, 2. / n)

    # Translate to center coordinates
    x += xc
    y += yc
    z += zc

    return x, y, z


def create_hyp_srf_3D(self, semi_axes, centroid, n, u_res=800, v_res=800):
    '''
    Create a closed B-spline surface for the hypershape using OpenCASCADE

    Parameters
    ----------
    u : `float`
        Longitude parameter [0, 2π]
    v : `float`
        Latitude parameter [-π/2, π/2]
    semi_axes : `tuple`
        Semi-axes of the hyperellipsoid (a, b, c)
    centroid : `tuple`
        Centroid of the hyperellipsoid (xc, yc, zc)
    n : `float`
        Degree of the hyperellipsoid
    point_func : `function`
        Function to compute points on the hypershape surface
    u_res : `int`, optional
        Resolution in the u direction (longitude). Default is 800
    v_res : `int`, optional
        Resolution in the v direction (latitude). Default is 800

    Returns:
    -------
    hyp_srf_tag : int
        OpenCASCADE surface tag for the hyperellipsoid
    '''

    # gmsh.initialize()
    gmsh.clear()

    # Create a new model using OpenCASCADE kernel
    gmsh.model.add("hyper_ellipsoid_occ")

    # Generate point grid
    print("Generating Hyperellipsoid Boundary Points", flush=True)
    point_tags = []
    for j in range(v_res):
        for i in range(u_res + 1):  # +1 to include closure point at u=2π

            # At poles, all u values should give the same point
            if j == 0:  # South pole
                u = 0.
                v = -np.pi / 2.

            elif j == v_res - 1:  # North pole
                u = 0.
                v = np.pi / 2.

            else:
                # Handle u-direction closure: last point same as first
                u = 0. if i == u_res else 2. * np.pi * i / u_res

                # Handle v direction including exact poles
                v = np.pi * (j / (v_res - 1.) - 0.5)  # From -π/2 to π/2

            # Create point on the hyperellipsoid surface
            x, y, z = self.create_hyp_pnt_3D(u, v, semi_axes, centroid, n)
            point_tag = gmsh.model.occ.addPoint(x, y, z)
            point_tags.append(point_tag)

    # Create B-spline surface
    print("Generating Hyperellipsoid Surface", flush=True)
    hyp_srf_tag = gmsh.model.occ.addBSplineSurface(
        pointTags=point_tags,
        numPointsU=u_res + 1,  # Include closure point
        tag=-1, degreeU=min(3, u_res), degreeV=min(3, v_res-1))
    gmsh.model.occ.synchronize()

    return hyp_srf_tag


def create_hyp_vol_3D(self, semi_axes, centroid, n, resol):
    '''
    Create a 3D hyperellipsoid volume using OpenCASCADE B-spline surfaces

    Parameters
    ----------
    semi_axes : `tuple`
        Semi-axes of the hyperellipsoid (a, b, c)
    centroid : `tuple`
        Centroid of the hyperellipsoid (xc, yc, zc)
    n : `float`
        Degree of the hyperellipsoid
    resol : `int`
        Resolution for the generation of the hyperellipsoid surface

    Returns:
    --------
    hyp_vol_tag : int
        OpenCASCADE volume tag for the hyperellipsoid
    '''

    print("Creating Hyperellipsoid", flush=True)

    hyp_vol_tag = None

    try:

        # Create surface
        u_res = resol
        v_res = int(np.ceil(min(semi_axes) / max(semi_axes) * resol) + 1)
        hyp_srf_tag = self.create_hyp_srf_3D(semi_axes, centroid, n,
                                             u_res=u_res, v_res=v_res)

        # Create volume
        print("Generating Hyperellipsoid Volume", flush=True)
        surface_loop = gmsh.model.occ.addSurfaceLoop([hyp_srf_tag])
        hyp_vol_tag = gmsh.model.occ.addVolume([surface_loop])
        gmsh.model.occ.synchronize()

        if hyp_vol_tag is None:
            return None

        # Hyperellipsoid centroid
        xc, yc, zc = centroid

        # Apply z-cut to remove upper part above free surface
        z_cut = 0.
        d_cut = 2 * max(semi_axes)

        # Create cutting box above z_cut
        print("Applying Cut at Free Surface", flush=True)
        cutting_box = gmsh.model.occ.addBox(xc - d_cut, yc - d_cut, z_cut,
                                            2 * d_cut, 2 * d_cut, d_cut)
        gmsh.model.occ.synchronize()

        # Remove everything above z_cut
        result = gmsh.model.occ.cut(
            [(3, hyp_vol_tag)], [(3, cutting_box)],
            removeObject=True, removeTool=True)

        # Verify if the resulting volume is valid
        if result[0]:
            print("Cut Applied Successfully", flush=True)
            hyp_vol_tag = result[0][0][1]
        else:
            print("Cut Removed Entire Volume", flush=True)
            hyp_vol_tag = None

        print("Successfully Created Volume Using Closed B-Spline Surface",
              flush=True)

        return hyp_vol_tag

    except Exception as e:
        print(f"B-spline Surface Creation Failed: {e}", flush=True)

        return None


def create_box_vol_3D(self, rec_box):
    '''
    Create a structured mesh for a box volume using OpenCASCADE

    Parameters
    ----------
    rec_box : `tuple`
        Box defined by original domain dimensions
        Structure: (xmin, xmax, ymin, ymax, zmin, zmax)

    Returns
    -------
    box_vol_tag : `int`
        OpenCASCADE volume tag for the box
    '''

    print("Creating Original Box Domain", flush=True)

    # Create ox volume
    xmin, xmax, ymin, ymax, zmin, zmax = rec_box
    box_vol_tag = gmsh.model.occ.addBox(xmin, ymin, zmin,  # corner
                                        xmax - xmin,       # width in x
                                        ymax - ymin,       # width in y
                                        zmax - zmin)       # width in z
    gmsh.model.occ.synchronize()

    # Structured mesh for cube only
    # Compute number of divisions along each axis based on edge size and h
    nx = max(1, int(round((xmax - xmin) / self.lmin)))
    ny = max(1, int(round((ymax - ymin) / self.lmin)))
    nz = max(1, int(round((zmax - zmin) / self.lmin)))

    # Get cube surfaces
    box_surfaces = gmsh.model.getBoundary(
        [(3, box_vol_tag)], oriented=False, recursive=False)
    box_surfaces = [s[1] for s in box_surfaces if s[0] == 2]

    # Apply transfinite meshing to cube edges and surfaces
    for s in box_surfaces:
        edges = gmsh.model.getBoundary([(2, s)],
                                       oriented=False,
                                       recursive=False)
        for e in edges:
            if e[0] == 1:  # line entity
                # Get curve bounding box to detect direction
                xmin, ymin, zmin, xmax, \
                    ymax, zmax = gmsh.model.getBoundingBox(1, e[1])
                dx = abs(xmax - xmin)
                dy = abs(ymax - ymin)
                dz = abs(zmax - zmin)

                if dx > dy and dx > dz:
                    gmsh.model.mesh.setTransfiniteCurve(e[1], nx + 1)
                elif dy > dx and dy > dz:
                    gmsh.model.mesh.setTransfiniteCurve(e[1], ny + 1)
                else:
                    gmsh.model.mesh.setTransfiniteCurve(e[1], nz + 1)
        gmsh.model.mesh.setTransfiniteSurface(s)

    # Apply transfinite volume
    gmsh.model.mesh.setTransfiniteVolume(box_vol_tag)
    gmsh.model.occ.synchronize()

    return box_vol_tag
