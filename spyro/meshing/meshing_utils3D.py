import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..io.basicio import _read_velocity_binary3D

try:
    import SeismicMesh
except ImportError:
    SeismicMesh = None


def sizing_function_xyz(
    X,
    Y,
    Z,
    ef_segy,
    length_x,
    length_y,
    depth_z,
):
    """Evaluate structured Winslow sizing using nearest-edge extension.

    Parameters
    ----------
    X : numpy.ndarray
        X coordinates of the structured mesh nodes.
    Y : numpy.ndarray
        Y coordinates of the structured mesh nodes.
    Z : numpy.ndarray
        Z coordinates of the structured mesh nodes.
    ef_segy : callable
        Mesh-sizing function evaluated in ``(z, x, y)`` coordinates.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth.

    Returns
    -------
    numpy.ndarray
        Positive sizing values with the same shape as the input coordinate arrays.

    Notes
    -----
    Queries outside the physical model are projected to the nearest model edge before
    evaluating the sizing field.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            "Winslow sizing coordinates X, Y and Z must have matching shapes."
        )

    finite_coordinates = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    if not np.all(finite_coordinates):
        bad_count = int(
            finite_coordinates.size - np.count_nonzero(finite_coordinates)
        )
        raise FloatingPointError(
            "Winslow generated NaN or infinite coordinates before "
            f"sizing evaluation ({bad_count} invalid nodes). Reduce "
            "winslow_omega or inspect the structured topology."
        )

    X_edge = np.clip(X.reshape(-1), 0.0, length_x)
    Y_edge = np.clip(Y.reshape(-1), 0.0, length_y)
    Z_edge = np.clip(Z.reshape(-1), -abs(depth_z), 0.0)

    queries_zxy = np.column_stack((Z_edge, X_edge, Y_edge))

    sizes = np.asarray(
        ef_segy(queries_zxy),
        dtype=float,
    ).reshape(X.shape)

    if not np.all(np.isfinite(sizes)):
        invalid = np.flatnonzero(~np.isfinite(sizes))
        first = int(invalid[0])
        raise ValueError(
            "The structured edge-extended sizing function returned "
            "NaN or infinity. First projected query "
            f"(z, x, y)={queries_zxy[first].tolist()}. Check the "
            "velocity binary metadata and sizing-function construction."
        )

    if np.any(sizes <= 0.0):
        minimum = float(np.min(sizes))
        raise ValueError(
            "The structured Winslow sizing function must be positive; "
            f"minimum value is {minimum}."
        )

    return sizes


def create_sizing_function3D(
    fname,
    hmin,
    bbox,
    wl,
    freq,
    pad_type=None,
    pad_size_x=0.0,
    pad_size_y=0.0,
    pad_size_z=0.0,
    grade=0.15,
    vp_water=None,
    nz=None,
    nx=None,
    ny=None,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
):
    """Create a finite positive wavelength-based sizing function for three-dimensional meshing.

    Parameters
    ----------
    fname : str or pathlib.Path
        Path to the binary velocity model.
    hmin : float or None
        Requested minimum element size.
    bbox : sequence of float
        Bounding box ordered as ``(zmin, zmax, xmin, xmax, ymin, ymax)``.
    wl : float
        Number of mesh points per wavelength.
    freq : float
        Reference frequency used to convert velocity to element size.
    pad_type : str or None
        Padding type forwarded to the sizing-field construction.
    pad_size_x : float
        Padding extent in x used by the sizing function.
    pad_size_y : float
        Padding extent in y used by the sizing function.
    pad_size_z : float
        Padding extent in z used by the sizing function.
    grade : float
        Maximum sizing-field grading parameter.
    vp_water : float or None
        Velocity assigned to water or zero-valued cells.
    nz : int
        Number of velocity samples in the z direction.
    nx : int
        Number of velocity samples in the x direction.
    ny : int
        Number of velocity samples in the y direction.
    byte_order : {"big", "little"}
        Byte order of the velocity-model binary file.
    axes_order : tuple of int
        Permutation mapping binary axes to Spyro ``(z, x, y)`` order.
    axes_order_sort : {"C", "F"}
        Memory order used to reshape the binary velocity data.
    dtype : str or numpy.dtype
        Numeric data type stored in the velocity-model file.

    Returns
    -------
    tuple
        Sizing callable, minimum size, maximum size when available, and grid dimensions.
    """
    if any(value is None for value in (nz, nx, ny)):
        raise ValueError("3-D sizing requires nz, nx and ny.")
    if wl is None or float(wl) <= 0.0:
        raise ValueError("wl must be positive.")
    if freq is None or float(freq) <= 0.0:
        raise ValueError("freq must be positive.")

    zmin, zmax, xmin, xmax, ymin, ymax = map(float, bbox)
    if not (zmin < zmax and xmin < xmax and ymin < ymax):
        raise ValueError(
            "bbox must be ordered as "
            "(zmin, zmax, xmin, xmax, ymin, ymax)."
        )

    requested_hmin = None
    if hmin is not None and float(hmin) > 0.0:
        requested_hmin = float(hmin)

    if vp_water is not None and float(vp_water) > 0.0:
        physical_positive_floor = (
            float(vp_water) / (float(freq) * float(wl))
        )
    elif requested_hmin is not None:
        physical_positive_floor = requested_hmin
    else:
        physical_positive_floor = np.finfo(np.float64).eps

    if requested_hmin is not None:
        physical_positive_floor = max(
            physical_positive_floor,
            requested_hmin,
        )

    def edge_extended_callable(base_callable):
        """Wrap a core interpolator with nearest-edge extension outside the model bounds.

        Parameters
        ----------
        base_callable : callable
            Core sizing interpolator defined inside the velocity-model bounds.

        Returns
        -------
        callable
            Sizing function that clamps queries to the velocity-model bounds.
        """
        def evaluate(coordinates):
            """Evaluate the edge-extended sizing function at one or more coordinates.

            Parameters
            ----------
            coordinates : numpy.ndarray
                Coordinates with shape ``(N, 3)`` in ``(z, x, y)`` order.

            Returns
            -------
            float or numpy.ndarray
                Positive sizing values corresponding to the supplied coordinates.
            """
            points = np.asarray(coordinates, dtype=np.float64)
            scalar_input = points.ndim == 1

            if scalar_input:
                points = points.reshape(1, 3)

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    "3-D sizing coordinates must have shape (N, 3) "
                    "in (z, x, y) order."
                )
            if not np.all(np.isfinite(points)):
                raise ValueError(
                    "3-D sizing coordinates contain NaN or infinity."
                )

            projected = points.copy()
            projected[:, 0] = np.clip(
                projected[:, 0], zmin, zmax
            )
            projected[:, 1] = np.clip(
                projected[:, 1], xmin, xmax
            )
            projected[:, 2] = np.clip(
                projected[:, 2], ymin, ymax
            )

            values = np.asarray(
                base_callable(projected),
                dtype=np.float64,
            ).reshape(-1)

            if values.size != projected.shape[0]:
                raise ValueError(
                    "The 3-D sizing callable returned an unexpected "
                    f"number of values: {values.size} for "
                    f"{projected.shape[0]} coordinates."
                )
            if not np.all(np.isfinite(values)):
                first = int(
                    np.flatnonzero(~np.isfinite(values))[0]
                )
                raise ValueError(
                    "The 3-D sizing function returned NaN or infinity "
                    "inside the velocity-model box. First projected "
                    f"coordinate (z, x, y)="
                    f"{projected[first].tolist()}."
                )
            if np.any(values < 0.0):
                first = int(np.flatnonzero(values < 0.0)[0])
                raise ValueError(
                    "The 3-D sizing function returned a negative size "
                    f"{values[first]} at projected coordinate "
                    f"(z, x, y)={projected[first].tolist()}."
                )

            values = np.maximum(
                values,
                physical_positive_floor,
            )

            if scalar_input:
                return values[0]
            return values

        return evaluate

    if SeismicMesh is not None:
        try:
            seismic_kwargs = {
                "vp_water": vp_water,
                "freq": float(freq),
                "wl": float(wl),
                "grade": float(grade),
                "domain_pad": max(
                    float(pad_size_x),
                    float(pad_size_y),
                    float(pad_size_z),
                    abs(xmax - xmin),
                    abs(ymax - ymin),
                    abs(zmax - zmin),
                ),
                "pad_style": "edge",
                "nz": int(nz),
                "nx": int(nx),
                "ny": int(ny),
                "byte_order": byte_order,
                "axes_order": tuple(axes_order),
                "axes_order_sort": axes_order_sort,
            }
            if requested_hmin is not None:
                seismic_kwargs["hmin"] = requested_hmin

            base_ef = SeismicMesh.get_sizing_function_from_segy(
                str(fname),
                bbox,
                **seismic_kwargs,
            )
            ef = edge_extended_callable(base_ef)

            return (
                ef,
                physical_positive_floor,
                None,
                int(nz),
                int(nx),
                int(ny),
            )
        except (TypeError, ValueError, RuntimeError):
            pass

    velocity = _read_velocity_binary3D(
        fname=fname,
        nz=nz,
        nx=nx,
        ny=ny,
        byte_order=byte_order,
        axes_order=axes_order,
        axes_order_sort=axes_order_sort,
        dtype=dtype,
    )

    if vp_water is not None:
        velocity = np.where(
            velocity == 0.0,
            float(vp_water),
            velocity,
        )

    if np.any(~np.isfinite(velocity)):
        raise ValueError(
            "The velocity model contains NaN or infinity."
        )
    if np.any(velocity <= 0.0):
        raise ValueError(
            "The velocity model contains non-positive values. Set vp_water "
            "or preprocess the model before constructing the sizing field."
        )

    sizes = velocity / (float(freq) * float(wl))
    if requested_hmin is not None:
        sizes = np.maximum(sizes, requested_hmin)

    sizes = np.maximum(sizes, physical_positive_floor)

    z_axis = np.linspace(zmax, zmin, int(nz))[::-1]
    x_axis = np.linspace(xmin, xmax, int(nx))
    y_axis = np.linspace(ymin, ymax, int(ny))
    sizes_for_interpolation = sizes[::-1, :, :]

    interpolator = RegularGridInterpolator(
        (z_axis, x_axis, y_axis),
        sizes_for_interpolation,
        method="linear",
        bounds_error=True,
    )

    ef = edge_extended_callable(interpolator)

    return (
        ef,
        float(np.min(sizes)),
        float(np.max(sizes)),
        int(nz),
        int(nx),
        int(ny),
    )


def define_winslow_points_3d(
    gmsh,
    points_3d,
    tag_to_index,
    length_x,
    length_y,
    depth_z,
    padding_type,
    padding_x,
    padding_y,
    padding_z,
    water_interface,
    tol=2.0,
):
    """Classify structured-mesh nodes by the coordinates they may move during Winslow smoothing.

    Parameters
    ----------
    gmsh : module
        Initialized Gmsh Python module used to build or query the mesh.
    points_3d : numpy.ndarray
        Mesh-node coordinates with shape ``(N, 3)``.
    tag_to_index : dict
        Mapping from Gmsh node tags to zero-based point indices.
    length_x : float
        Physical domain length in the x direction.
    length_y : float
        Physical domain length in the y direction.
    depth_z : float
        Physical domain depth; the model occupies negative z.
    padding_type : str or None
        Padding geometry, either ``None`` or ``"rectangular"`` for Winslow.
    padding_x : float
        Padding thickness in the x direction.
    padding_y : float
        Padding thickness in the y direction.
    padding_z : float
        Bottom padding thickness in the z direction.
    water_interface : bool
        Whether the water/subsurface interface is geometrically delimited.
    tol : float
        Coordinate tolerance used to identify boundary planes.

    Returns
    -------
    dict
        Sets of locked, directionally movable, fully movable, water, and union movable
        nodes.

    Notes
    -----
    Water nodes are locked when a delimited water volume is present. Boundary
    intersections are constrained so corners are fixed, edges move tangentially, faces
    move in-plane, and interior nodes move freely.
    """
    points_3d = np.asarray(points_3d, dtype=float)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3).")

    if padding_type not in (None, "rectangular"):
        raise ValueError(
            "3-D Winslow point selection supports only padding_type=None "
            "or padding_type='rectangular'; 'hyperelliptical' is not "
            "supported for structured Winslow smoothing."
        )

    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Winslow point-selection tolerance must be positive.")

    locked = set()
    move_X_only = set()
    move_Y_only = set()
    move_Z_only = set()
    move_all = set()
    water_nodes = set()

    if water_interface:
        water_group_name = (
            "Water_with_padding"
            if padding_type == "rectangular"
            else "Water"
        )

        group_found = False
        for dim, physical_tag in gmsh.model.getPhysicalGroups(dim=3):
            if gmsh.model.getPhysicalName(dim, physical_tag) != water_group_name:
                continue

            group_found = True
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag)
            for entity_tag in entities:
                node_tags, _, _ = gmsh.model.mesh.getNodes(
                    dim,
                    entity_tag,
                    includeBoundary=True,
                )
                for node_tag in node_tags:
                    node_index = tag_to_index.get(int(node_tag))
                    if node_index is not None:
                        water_nodes.add(node_index)

        if not group_found:
            raise RuntimeError(
                f"Physical volume group {water_group_name!r} was not found "
                "while selecting 3-D Winslow nodes."
            )

        if not water_nodes:
            raise RuntimeError(
                f"Physical volume group {water_group_name!r} contains no "
                "mesh nodes for 3-D Winslow smoothing."
            )

        if len(water_nodes) == len(points_3d):
            raise RuntimeError(
                f"Every mesh node was classified as {water_group_name}. "
                "No subsurface nodes remain available for Winslow smoothing."
            )

    x_planes = [0.0, float(length_x)]
    y_planes = [0.0, float(length_y)]
    z_planes = [-abs(float(depth_z))]

    if padding_type == "rectangular":
        x_planes.extend(
            [
                -float(padding_x),
                float(length_x) + float(padding_x),
            ]
        )
        y_planes.extend(
            [
                -float(padding_y),
                float(length_y) + float(padding_y),
            ]
        )
        z_planes.append(
            -abs(float(depth_z)) - float(padding_z)
        )

    if not water_interface:
        z_planes.append(0.0)

    def on_any_plane(value, planes):
        """Test whether a coordinate lies on any constrained plane within tolerance.

        Parameters
        ----------
        value : float
            Coordinate to classify.
        planes : sequence of float
            Candidate plane coordinates.

        Returns
        -------
        bool
            ``True`` when the coordinate is within tolerance of any supplied plane.
        """
        return any(abs(value - plane) < tol for plane in planes)

    for node_index, (x_coord, y_coord, z_coord) in enumerate(points_3d):
        if node_index in water_nodes:
            locked.add(node_index)
            continue

        on_x_plane = on_any_plane(x_coord, x_planes)
        on_y_plane = on_any_plane(y_coord, y_planes)
        on_z_plane = on_any_plane(z_coord, z_planes)

        contact_count = int(on_x_plane) + int(on_y_plane) + int(on_z_plane)

        if contact_count == 3:
            locked.add(node_index)
            continue

        if contact_count == 2:
            if on_x_plane and on_y_plane:
                move_Z_only.add(node_index)
            elif on_x_plane and on_z_plane:
                move_Y_only.add(node_index)
            elif on_y_plane and on_z_plane:
                move_X_only.add(node_index)
            continue

        if contact_count == 1:
            if on_x_plane:
                move_Y_only.add(node_index)
                move_Z_only.add(node_index)
            elif on_y_plane:
                move_X_only.add(node_index)
                move_Z_only.add(node_index)
            elif on_z_plane:
                move_X_only.add(node_index)
                move_Y_only.add(node_index)
            continue

        move_all.add(node_index)

    movable_nodes = (
        move_all
        | move_X_only
        | move_Y_only
        | move_Z_only
    )

    if not movable_nodes:
        raise RuntimeError(
            "No movable nodes were found for 3-D Winslow smoothing."
        )

    return {
        "locked": locked,
        "move_all": move_all,
        "move_X_only": move_X_only,
        "move_Y_only": move_Y_only,
        "move_Z_only": move_Z_only,
        "water_nodes": water_nodes,
        "movable_nodes": movable_nodes,
    }
