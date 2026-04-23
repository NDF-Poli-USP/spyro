"""Methods to extend the material property in an absorbing layer."""

import firedrake as fire
from numpy import clip, where
from firedrake.__future__ import interpolate
from spyro.domains.space import create_function_space
from spyro.utils.error_management import value_parameter_error
from spyro.utils.eval_functions_to_ufl import generate_ufl_functions
fire.interpolate = interpolate


def generate_conditional_value_for_layer(domain_dim, mesh, dimension,
                                         ufl_coordinates_habc, type_marker='mask'):
    """Generate the conditional value for the absorbing layer.

    Parameters
    ----------
    domain_dim : `tuple`
        Domain dimensions: (length_z, length_x) for 2
        or (length_z, length_x, length_y) for 3D
    mesh : `Firedrake.Mesh`
        Current mesh
    dimension : `int`
        Model dimension (2D or 3D)
    ufl_coordinates_habc : `ufl.geometry.SpatialCoordinate`
        Domain Coordinates including the absorbing layer
    type_marker : `string`, optional
        Type of marker for the absorbing layer. Default is 'mask'.
        - 'damping' : Get the reference distance to the original boundary
        - 'mask' : Define a mask to filter the layer boundary domain

    Returns
    -------
    ref_conditional : `ufl.conditional.Conditional`
        Conditional expression to identify the layer domain or reference
        distance for the damping function inside the layer
    """

    # Domain dimensions
    Lz, Lx = domain_dim[:2]

    # UFL coordinates
    z, x = ufl_coordinates_habc[0], ufl_coordinates_habc[1]
    if dimension == 3:  # 3D
        y = ufl_coordinates_habc[2]

    # Conditional expression
    condz = z < -Lz
    condx1 = x < 0.
    condx2 = x > Lx

    # Conditional value
    exprz = f"(z + {Lz})**2" if type_marker == 'damping' else "1"
    exprx1 = "x**2" if type_marker == 'damping' else "1"
    exprx2 = f"(x - {Lx})**2" if type_marker == 'damping' else "1"
    valz = generate_ufl_functions(mesh, exprz, dimension)
    valx1 = generate_ufl_functions(mesh, exprx1, dimension)
    valx2 = generate_ufl_functions(mesh, exprx2, dimension)

    # Conditional expressions for the mask
    z_pd = fire.conditional(condz, valz, 0.)
    x_pd = fire.conditional(condx1, valx1, 0.) + fire.conditional(condx2, valx2, 0.)
    ref_conditional = z_pd + x_pd

    if dimension == 3:  # 3D

        # 3D dimension
        Ly = domain_dim[2]

        # Conditional expression
        condy1 = y < 0.
        condy2 = y > Ly

        # Conditional value
        expry1 = "y**2" if type_marker == 'damping' else "1"
        expry2 = f"(y - {Ly})**2" if type_marker == 'damping' else "1"
        valy1 = generate_ufl_functions(mesh, expry1, dimension)
        valy2 = generate_ufl_functions(mesh, expry2, dimension)

        # Conditional expressions for the mask
        y_pd = fire.conditional(condy1, valy1, 0.) + fire.conditional(condy2, valy2, 0.)
        ref_conditional += y_pd

    return ref_conditional


def layer_mask_field(domain_dim, mesh, dimension, ufl_coordinates_habc, V,
                     damp_par=None, type_marker='damping', name_mask=None):
    """Generate a mask function inside for the absorbing layer.

    The mask is defined for conditional expressions to identify the domain
    of the layer (option: 'mask') or the reference to the original boundary
    (option: 'damping') used to compute the damping profile.

    Parameters
    ----------
    domain_dim : `tuple`
        Domain dimensions: (length_z, length_x) for 2
        or (length_z, length_x, length_y) for 3D
    mesh : `Firedrake.Mesh`
        Current mesh
    dimension : `int`, optional
        Model dimension (2D or 3D)
    ufl_coordinates_habc : `ufl.geometry.SpatialCoordinate`
        Domain Coordinates including the absorbing layer
    V : `Firedrake.FunctionSpace`
        Function space for the mask field
    damp_par : `tuple`, optional
        Damping parameters for the absorbing layer.
        Structure: (pad_len, eta_crt, aq, bq)
        - pad_len : `float`
            Size of the absorbing layer
        - eta_crt : `float`
            Critical damping coefficient (1/s)
        - aq : `float`
            Coefficient for quadratic term in the damping function
        - bq : `float`
            Coefficient bq for linear term in the damping function
    type_marker : `string`, optional
        Type of marker. Default is 'mask'.
        - 'damping' : Get the reference distance to the original boundary
        - 'mask' : Define a mask to filter the layer boundary domain
    name_mask : `string`, optional
        Name for the mask field. Default is None

    Returns
    -------
    layer_mask : `Firedrake.Function`
        Mask for the absorbing layer
        - 'damping' : `ufl.conditional.Conditional`
            Reference distance to the original boundary
        - 'mask' : `ufl.algebra.Division`
            Conditional expression to identify the layer domain
    """

    # Reference function for the layer mask
    ref_funct = generate_conditional_value_for_layer(domain_dim, mesh,
                                                     dimension, ufl_coordinates_habc,
                                                     type_marker=type_marker)

    if type_marker == 'damping':  # Damping profile for the absorbing layer

        if damp_par is None:
            raise ValueError("Damping parameters must be provided "
                             "when 'type_marker' is 'damping'.")

        # Damping parameters
        pad_len, eta_crt, aq, bq = damp_par

        if pad_len <= 0:
            raise ValueError(f"Invalid value for 'pad_len': {pad_len}. "
                             "'pad_len' must be greater than zero "
                             "when 'type_marker' is 'damping'.")
        if eta_crt <= 0:
            raise ValueError(f"Invalid value for 'eta_crt': {eta_crt}. "
                             "'eta_crt' must be greater than zero "
                             "when 'type_marker' is 'damping'.")

        # Reference distance to the original boundary
        ref_funct = fire.sqrt(ref_funct) / fire.Constant(pad_len)

        # Quadratic damping profile
        if bq == 0.:
            ref_funct = fire.Constant(eta_crt) * fire.Constant(aq) * ref_funct**2
        else:
            ref_funct = fire.Constant(eta_crt) * (fire.Constant(aq) * ref_funct**2
                                                  + fire.Constant(bq) * ref_funct)

    elif type_marker == 'mask':  # Mask filter for layer boundary domain

        ref_funct = fire.conditional(ref_funct > 0, 1., 0.)

    else:
        value_parameter_error('type_marker', type_marker, ['damping', 'mask'])

    layer_mask = fire.Function(V, name=name_mask)
    layer_mask.assign(fire.assemble(fire.interpolate(ref_funct, V)))

    return layer_mask


def clipping_coordinates_lay_field(domain_dim, mesh, dimension,
                                   ufl_coordinates_habc, V, quadrilateral=False):
    """Generate a field with clipping coordinates to the original boundary.

    Parameters
    ----------
    domain_dim : `tuple`
        Domain dimensions: (length_z, length_x) for 2
        or (length_z, length_x, length_y) for 3D
    mesh : `Firedrake.Mesh`
        Current mesh
    dimension : `int`
        Model dimension (2D or 3D)
    ufl_coordinates_habc : `ufl.geometry.SpatialCoordinate`
        Domain Coordinates including the absorbing layer
    V : `Firedrake.FunctionSpace`
        Function space for the mask field
    quadrilateral : bool, optional
        Flag to indicate whether to use quadrilateral/hexahedral elements

    Returns
    -------
    lay_field : `Firedrake.Function`
        Field with clipped coordinates only in the absorbing layer
    layer_mask : `Firedrake.Function`
        Mask for the absorbing layer
    """

    print("Clipping Coordinates Inside Layer", flush=True)

    # Domain dimensions
    Lz, Lx = domain_dim[:2]

    # Vectorial space for auxiliar field of clipped coordinates
    method_element = "DQ" if quadrilateral else "DG"
    W = create_function_space(mesh, method_element, 0, dim=dimension)

    # Clipping coordinates
    lay_field = fire.Function(W)
    lay_field.assign(fire.assemble(fire.interpolate(ufl_coordinates_habc, W)))
    lay_arr = lay_field.dat.data_with_halos[:]
    lay_arr[:, 0] = clip(lay_arr[:, 0], -Lz, 0.)
    lay_arr[:, 1] = clip(lay_arr[:, 1], 0., Lx)

    if dimension == 3:  # 3D

        # 3D dimension
        Ly = domain_dim[2]

        # Clipping coordinates
        lay_arr[:, 2] = clip(lay_arr[:, 2], 0., Ly)

    # Mask function to identify the absorbing layer domain
    layer_mask = layer_mask_field(domain_dim, mesh, dimension,
                                  ufl_coordinates_habc, V, type_marker='mask')

    # Field with clipped coordinates only in the absorbing layer
    lay_field.assign(fire.assemble(fire.interpolate(lay_field * layer_mask, W)))

    return lay_field, layer_mask


def point_cloud_field(parent_mesh, pts_cloud, parent_field, tolerance):
    """Create a field on a point cloud from a parent mesh and field.

    Parameters
    ----------
    parent_mesh : `firedrake mesh`
        Parent mesh containing the original field
    pts_cloud : `array`
        Array of shape (num_pts, dim) containing the coordinates
        of the point cloud
    parent_field : `Firedrake.Function`
        Parent field defined on the parent mesh
    tolerance : `float`
        Tolerance for searching nodes in the mesh

    Returns
    -------
    cloud_field : `Firedrake.Function`
        Field defined on the point cloud
    """

    # Creating a point cloud field from the parent mesh
    pts_mesh = fire.VertexOnlyMesh(
        parent_mesh, pts_cloud, reorder=True, tolerance=tolerance,
        missing_points_behaviour='error', redundant=False)
    del pts_cloud

    # Cloud field
    V0 = fire.FunctionSpace(pts_mesh, "DG", 0)
    f_pts = fire.assemble(fire.interpolate(parent_field, V0))

    # Ensuring correct assemble
    V1 = fire.FunctionSpace(pts_mesh.input_ordering, "DG", 0)
    del pts_mesh
    cloud_field = fire.Function(V1)
    cloud_field.assign(fire.assemble(fire.interpolate(f_pts, V1)))
    del f_pts

    return cloud_field


def extend_scalar_field_profile(mesh_original, field_to_extend, lay_field, layer_mask,
                                tolerance, method='point_cloud', name_prop="Property"):
    """Extend the profile of a scalar field inside the absorbing layer.

    Parameters
    ----------
    mesh_original : `Firedrake.Mesh`
        Original mesh without absorbing layer
    field_to_extend : `Firedrake.Function`
        Scalar field defined on the original mesh to be extended inside the layer
    lay_field : `Firedrake.Function`
        Field with clipped coordinates only in the absorbing layer
    layer_mask : `Firedrake.Function`
        Mask for the absorbing layer
    tolerance : `float`
        Tolerance for searching nodes in the mesh
    method : `str`, optional
        Method to extend the velocity profile. Options:
        'point_cloud' or 'nearest_point'. Default is 'point_cloud'
    name_prop : `str`, optional
        Name for the property field. Default is "Property"

    Returns
    -------
    extended_field : `Firedrake.Function`
        Extended scalar field defined on the same function space as `lay_field`.
    """

    print("Extending Profile Inside Layer", flush=True)

    # Extracting the nodes from the layer field
    lay_nodes = lay_field.dat.data_with_halos[:]

    # Nodes to extend the property field
    ind_nodes = where(layer_mask.dat.data_with_halos)[0]
    pts_to_extend = lay_nodes[ind_nodes]

    if method == 'point_cloud':

        print(f"Using Cloud Points Method to Extend {name_prop} Profile", flush=True)

        # Set the property of the nearest point on the original boundary
        vel_to_extend = \
            point_cloud_field(mesh_original, pts_to_extend,
                              field_to_extend, tolerance).dat.data_with_halos[:]

    elif method == 'nearest_point':

        print(f"Using Nearest Point Method to Extend {name_prop} Profile", flush=True)

        # Set the property of the nearest point on the original boundary
        vel_to_extend = field_to_extend.at(pts_to_extend, dont_raise=True)
        del pts_to_extend

    else:
        value_parameter_error('method', method, ['point_cloud', 'nearest_point'])

    # Velocity profile inside the layer
    lay_field.dat.data_with_halos[ind_nodes, 0] = vel_to_extend
    del vel_to_extend, ind_nodes

    extended_field = lay_field.sub(0)

    return extended_field
