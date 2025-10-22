class Sign_Distance_Hyp3D():
    '''
    Class for the signed distance function of a truncated
    hyperellipsoid with a truncation plane at x = x_cut

    Attributes
    ----------
    semi_axes : `tuple`
        Semi-axes of the hyperellipsoid (a, b, c)
    centroid : `tuple`
        Centroid of the hyperellipsoid (xc, yc, zc)
     n : `float`
        Degree of the hyperellipsoid
    x_cut : `float`, optional
        x-coordinate of the truncation plane. Default is 0

    Methods
    -------
    hyperellipsoid_sdf()
        Compute the signed distance function for the hyperellipsoid
    sign_dist_hyptrunc
            Compute the combined signed distance function to
            create the boundary of a truncated hypererellipsoid
    top_cut_sdf()
        Compute the signed distance function for the top cut
    '''

    def __init__(self, semi_axes, centroid, n, x_cut=0.):
        '''
        Initialize the Sign_Distance_Hyp3D class.
        Create a signed distance function for a truncated
        hyperellipsoid with a truncation plane at x = x_cut

        Parameters
        ----------
        semi_axes : `tuple`
            Semi-axes of the hyperellipsoid (a, b, c)
        centroid : `tuple`
            Centroid of the hyperellipsoid (xc, yc, zc)
        n : `float`
            Degree of the hyperellipsoid
        x_cut : `float`, optional
            x-coordinate of the truncation plane. Default is 0

        Returns
        -------
        None
        '''

        # Validate input parameters
        if len(semi_axes) != 3 or any(sa <= 0 for sa in semi_axes):
            raise ValueError("semi_axes must be a tuple of "
                             "three positive values (a, b, c).")

        if len(centroid) != 3:
            raise ValueError("centroid must be a tuple of "
                             "three values (xc, yc, zc).")

        if n <= 0:
            raise ValueError("n must be a positive value.")

        if not isinstance(x_cut, (int, float)):
            raise ValueError("x_cut must be a numeric value.")

        # Assign attributes
        self.semi_axes = semi_axes
        self.centroid = centroid
        self.n = n
        self.x_cut = x_cut

    def hyperellipsoid_sdf(self, pts):
        '''
        Compute the signed distance function for the hyperellipsoid

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the points
            where the signed distance function is evaluated

        Returns
        -------
        sdf : `array`
            Array containing the signed distance function values
        '''

        # Semi-axes and centroid
        a, b, c = self.semi_axes
        xc, yc, zc = self.centroid

        # Relative coordinates to center at (xc, yc, zc)
        x_h = abs((pts[:, 0] - xc) / a)**self.n
        y_h = abs((pts[:, 1] - yc) / b)**self.n
        z_h = abs((pts[:, 2] - zc) / c)**self.n

        # Compute radius in hyperellipsoid space
        r = (x_h + y_h + z_h)**(1. / self.n)

        # Signed distance
        sign_dist = r - 1.

        # Rescale back to original space
        scale = min(a, b, c)

        return sign_dist * scale

    def top_cut_sdf(self, pts):
        '''
        Compute the signed distance function for the top cut

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the points
            where the signed distance function is evaluated

        Returns
        -------
        sdf : `array`
            Array containing the signed distance function values
        '''
        return (pts[:, 0] - self.x_cut)

    def sign_dist_hyptrunc(self, pts):
        '''
        Compute the combined signed distance function to
        create the boundary of a truncated hypererellipsoid

        Parameters
        ----------
        pts : `array`
            Array containing the coordinates of the points
            where the signed distance function is evaluated

        Returns
        -------
        sdf : `array`
            Array containing the signed distance function values
        '''
        return np.maximum(self.hyperellipsoid_sdf(pts), self.top_cut_sdf(pts))


def mesh_data_3D(self, hyp_par):
    '''
    Generate mesh data for the hyperellipsoidal absorbing layer

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
        Structure: (zmin, zmax, xmin, xmax, ymin, ymax)
    hyp_box : `tuple`
        Box that envelopes the hyperellipsoid
        Structure: (zmin, zmax, xmin, xmax, ymin, ymax)
    fixed_pnts : `array`
        Array containing the coordinates of the original mesh points
    centroid : `tuple`
        Hyperellipsoid centroid according to reference system (z, x, y)
        Structure: (zc, xc, yc)
    semi_axes : `tuple`
        Hyperellipsoid semi-axes according to reference system (z, x, y)
        Structure: (b_hyp, a_hyp, c_hyp)
    surf_ele_expt : `int`
        Expected elements in a full hyperellipsoid surface
    '''

    # Domain dimensions
    Lx, Lz, Ly = self.dom_dim

    # Hyperellipsoid parameters
    n_hyp, surface, a_hyp, b_hyp, c_hyp = hyp_par

    # Box defined by original domain dimensions
    rec_box = (-Lz, 0., 0., Lx, 0., Ly)

    # Hyperellipsoid centroid
    zc = -Lz / 2.
    xc = Lx / 2.
    yc = Ly / 2.

    # Box that envelopes the hyperellipsoid
    hyp_zmin = zc - b_hyp
    hyp_zmax = zc + b_hyp
    hyp_xmin = xc - a_hyp
    hyp_xmax = xc + a_hyp
    hyp_ymin = yc - c_hyp
    hyp_ymax = yc + c_hyp
    hyp_box = (hyp_zmin, hyp_zmax, hyp_xmin, hyp_xmax, hyp_ymin, hyp_ymax)

    # Extract node positions
    fixed_pnts = self.mesh_original.coordinates.dat.data_with_halos

    # Clip fixed_pnts points to ensure they are within the bounding box
    fixed_pnts[:, 0] = np.clip(fixed_pnts[:, 0], hyp_zmin, hyp_zmax)
    fixed_pnts[:, 1] = np.clip(fixed_pnts[:, 1], hyp_xmin, hyp_xmax)
    fixed_pnts[:, 2] = np.clip(fixed_pnts[:, 2], hyp_ymin, hyp_ymax)

    # Hyperellipsoid centroid according to reference system (z, x, y)
    centroid = (zc, xc, yc)

    # Hyperellipsoid semi-axes according to reference system (z, x, y)
    semi_axes = (b_hyp, a_hyp, c_hyp)

    # Expected elements in a full hyperellipsoid surface
    surf_ele_expt = np.ceil(surface / self.lmin**2) + 1

    return rec_box, hyp_box, fixed_pnts, centroid, semi_axes, surf_ele_expt


def get_clean_mesh_pnts3D(self, all_pnts, dom_pnts, box=None):
    '''
    Remove points outside the original mesh and too close to the box

    Parameters
    ----------
    all_pnts : `array`
        Array containing the coordinates of all the mesh points
    dom_pnts : `array`
        Array containing the coordinates of the original mesh points
    box : `tuple`, optional
        Original box bounds. Default is None
        Structure: (xmin, xmax, ymin, ymax, zmin, zmax).

    Returns
    -------
    clean_pnts : `array`
        Array containing the coordinates of the filtered mesh points
    '''

    if len(all_pnts) == 0:
        return all_pnts.copy()

    # Convert to numpy arrays
    all_pnts = np.asarray(all_pnts)
    dom_pnts = np.asarray(dom_pnts)

    # Determine box bounds
    if box is None:
        if len(dom_pnts) == 0:
            return all_pnts.copy()

        # If not provided
        xmin = np.min(dom_pnts[:, 0])
        xmax = np.max(dom_pnts[:, 0])
        ymin = np.min(dom_pnts[:, 1])
        ymax = np.max(dom_pnts[:, 1])
        zmin = np.min(dom_pnts[:, 2])
        zmax = np.max(dom_pnts[:, 2])
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = box

    # # Vectorized check if point is clearly outside cube -> keep it
    # x, y, z = all_pnts[:, 0], all_pnts[:, 1], all_pnts[:, 2]
    # cond_out = ((x < xmin - self.tol) | (x > xmax + self.tol)
    #             | (y < ymin - self.tol) | (y > ymax + self.tol)
    #             | (z < zmin - self.tol) | (z > zmax + self.tol))

    # # Get inside points only
    # in_pnts = all_pnts[~cond_out]

    # if len(in_pnts):
    #     # Build k-d tree for fast nearest neighbor queries
    #     dom_tree = cKDTree(dom_pnts)

    #     # Batch query all inside points at once
    #     distances = dom_tree.query(
    #         in_pnts, distance_upper_bound=self.tol, workers=-1)[0]

    #     # Get the points are close to domain points
    #     clean_in_pnts = in_pnts[distances <= self.tol]
    # else:
    #     clean_in_pnts = np.array([]).reshape(0, 3)

    # # Get outside points only
    # out_pnts = all_pnts[cond_out]

    # if len(out_pnts):

    #     # Vectorized minimum distance to box calculation
    #     x, y, z = out_pnts[:, 0], out_pnts[:, 1], out_pnts[:, 2]

    #     # Calculate distance components
    #     dx = np.maximum(0, np.maximum(xmin - x, x - xmax))
    #     dy = np.maximum(0, np.maximum(ymin - y, y - ymax))
    #     dz = np.maximum(0, np.maximum(zmin - z, z - zmax))

    #     # Euclidean distance
    #     distances = np.sqrt(dx**2 + dy**2 + dz**2)

    #     # Keep points that are far enough from structured box mesh
    #     clean_out_pnts = out_pnts[distances > self.lmin / 1.25]
    # else:
    #     clean_out_pnts = np.array([]).reshape(0, 3)

    # # Combine outside points and inside clean points
    # clean_pnts = np.vstack([clean_out_pnts, clean_in_pnts])

    # import ipdb; ipdb.set_trace()

    # return clean_pnts if len(clean_pnts) else np.array([]).reshape(0, 3)

    clean_pnts = []

    for point in all_pnts:
        x, y, z = point

        # Check if point is clearly outside cube -> keep it
        cond_out = (x < xmin - self.tol or x > xmax + self.tol
                    or y < ymin - self.tol or y > ymax + self.tol
                    or z < zmin - self.tol or z > zmax + self.tol)

        if cond_out:
            clean_pnts.append(point)
        else:
            # Point is inside or on boundary - check if it's a dom_pnt
            is_dom_pnt = False
            for dom_pnt in dom_pnts:
                if np.linalg.norm(point - dom_pnt) < self.tol:
                    is_dom_pnt = True
                    break

            if is_dom_pnt:
                clean_pnts.append(point)
            # If inside/boundary and not dom_pnt, don't add (remove it)

    return np.array(clean_pnts) if clean_pnts else np.array([]).reshape(0, 3)


@staticmethod
def remove_elements_outside_hyp3D(pts, connect, geps, hyp3D_sdf):
    '''
    Remove elements outside the hyperellipsoid.
    Based on a post-processing function used by SeismicMesh.

    Parameters
    ----------
    pts : `array`
        Array containing the coordinates of the mesh points
    connect : `array`
        Array containing the connectivity of the mesh points
    geps : `float`
        Geometric tolerance for the hyperellipsoid surface
    hyp3D_sdf : `function`
        Function to evaluate the signed distance to the hyperellipsoid

    Returns
    -------
    connect : `array`
        Filtered array of shape (num_tets_filtered, 4) containing the
        indices of the mesh points that form each tetrahedral element
        3D array
    '''

    dim = pts.shape[1]
    pmid = pts[connect].sum(1) / (dim + 1)  # Compute centroids

    return connect[hyp3D_sdf(pmid) < -geps]  # Keep interior elements


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

    # Hyperellipse parameters
    n_hyp, surface, a_hyp, b_hyp, c_hyp = hyp_par

    rec_box, hyp_box, pfix, centr, s_axs, \
        surf_ele_expt = self.mesh_data_3D(hyp_par)

    sdfunct = Sign_Distance_Hyp3D(s_axs, centr, n_hyp)

    # Generate a initial 3D mesh
    print("Generating Initial Mesh with Fixed Points in Original Box")
    orig_dom = geometry.Cube(rec_box)
    points, cells = generation.generate_mesh(
        domain=sdfunct.sign_dist_hyptrunc, edge_length=self.lmin,
        bbox=hyp_box, max_iter=400, pfix=pfix, subdomains=[orig_dom])

    # Get clean points to generate final mesh
    print("Getting Clean Points to Generate Final Mesh")
    points = self.get_clean_mesh_pnts3D(points, pfix, rec_box)

    # Create new Delaunay with structured mesh part
    fixed_triangulation = Delaunay(points)
    cells = fixed_triangulation.simplices

    # SeismicMesh filter for boundary elements
    print("Removing Elements Outside the Hypershape")
    geps = (surface / (2e3 * surf_ele_expt))**0.5
    cells = self.remove_elements_outside_hyp3D(points, cells, geps,
                                               sdfunct.sign_dist_hyptrunc)

    # import meshio
    # meshio.write_points_cells(
    #     "hyp3d.msh",
    #     points,
    #     [("tetra", cells)],
    #     file_format="gmsh22",
    #     binary=False)

    # final_mesh = fire.Mesh(
    #     "fragmented_geometry.msh",
    #     distribution_parameters={"overlap_type": (
    #         fire.DistributedMeshOverlapType.NONE, 0)},
    #     comm=self.comm.comm)

    # Create the final mesh
    final_mesh = Mesh()
    final_mesh.dim = self.dimension
    final_mesh.SetMaterial(1, "habc")

    # Add points
    point_map = {}
    for i, coord in enumerate(points):
        z, x, y = coord
        point_map[i] = final_mesh.Add(MeshPoint((z, x, y)))

    # Add elements
    for cell in cells:
        netgen_points = [point_map[cell[i]] for i in range(len(cell))]
        final_mesh.Add(Element3D(1, netgen_points))  # 1: Volume marker

    try:
        # Mesh data
        # final_mesh.Compress()
        print(f"Mesh created with {len(final_mesh.Points())} points "
              f"and {len(final_mesh.Elements3D())} elements")

        # Mesh is transformed into a firedrake mesh
        final_mesh = fire.Mesh(final_mesh, comm=self.comm.comm)
        print("Merged Mesh Generated Successfully")

    except Exception as e:
        UserWarning(f"Error Generating Merged Mesh: {e}. Exiting.")

    return final_mesh
