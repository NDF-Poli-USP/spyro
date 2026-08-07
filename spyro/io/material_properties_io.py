import warnings
from os import getcwd

import firedrake as fire
from numpy import log10, ones
from numpy.random import uniform

from .interpolators import interpolate
from ..utils import error_management
from ..utils import eval_functions_to_ufl
from ..utils.utils import write_hdf5_velocity_model
from ..domains.space import check_function_space_type, create_function_space


def define_property_function_space(wave, func_space_type, dg_property,
                                   shape_func_space=None):
    """
    Define the function space for a material property.

    Parameters:
    -----------
    func_space_type, `str`
        Type of function space for the material property.
        Options: 'scalar', 'vector' or 'tensor'
    dg_property: `bool`
        If True, uses a DG0 function space for conditional and
        expression inputs
    shape_func_space: `tuple`, optional
        Shape of the function space for only tensorial material property.
        Default is None

    Returns:
    --------
    V: `firedrake function space`
        Function space for the material property
    """

    # Checking input arguments
    opts_func_space_type = ["scalar", "vector", "tensor"]
    error_management.value_parameter_error("func_space_type", func_space_type,
                                           opts_func_space_type)

    if dg_property is False and func_space_type == "scalar":
        return point_to_scalar_wave_function_space(wave)
    elif dg_property is False and func_space_type == "vector":
        return point_to_vector_wave_function_space(wave)

    if dg_property and func_space_type == "scalar":
        return point_to_dg_scalar_wave_function_space(wave)
    elif dg_property and func_space_type == "vector":
        return point_to_dg_vector_wave_function_space(wave)

    if func_space_type == "tensor":
        return point_to_correct_tensor_space(wave, shape_func_space, dg_property)


def _initialize_material_property_from_ufl(wave, property_name,
                                           func_space_type, V,
                                           shape_func_space=None,
                                           constant=None,
                                           conditional=None,
                                           expression=None):
    """
    Initialize material property from a UFL input. This method is
    used when the material property is defined by a constant value,
    a conditional or an expression.

    Parameters:
    -----------
    property_name: `str`
        Name of the material property to be set
    func_space_type, `str`
        Type of function space for the material property.
        Options: 'scalar', 'vector' or 'tensor'
    V: `firedrake function space`
        Function space for the material property
    shape_func_space: `tuple`, optional
        Shape of the function space for only tensorial material property.
        Default is None
    constant: `float`, optional
        Constant value for the material property. Default is None
    conditional:  `firedrake conditional`, optional
        Firedrake conditional object. Default is None
    expression: `str`, optional
        If you use an expression, you can use the following variables:
        x, y, z, pi, tanh, sqrt. Ex: "2. + 0.5 * tanh((x - 2.) / 0.1)".
        It will be interpoalte into either the same function space as
        the object or a DG0 function space in the same mesh.
        Default is None

    Returns:
    --------
    mat_property: `firedrake function`
        Material property
    """

    if constant is not None:
        value = 1 if constant == 0.0 else abs(constant)
        col = int(abs(log10(abs(value)))) + 2

        print(f"Assigning {property_name} with a "
              f"constant value of {constant:>{col}}", flush=True)

        if func_space_type == "vector":
            ufl_input = fire.as_vector((constant,) * wave.dimension)

        elif func_space_type == "tensor":
            ufl_input = fire.as_tensor(constant * ones((shape_func_space)))
        else:
            ufl_input = fire.Constant(constant)

    if conditional is not None:
        print(f"Assigning {property_name} with a conditional "
              f"field given by {conditional}", flush=True)
        ufl_input = conditional

    if expression is not None:
        print(f"Assigning {property_name} with an expression "
              f"field given by f = {expression} ", flush=True)
        ufl_input = eval_functions_to_ufl.generate_ufl_functions(wave.mesh,
                                                                 expression,
                                                                 wave.dimension)

    mat_property = fire.Function(V, name=property_name).interpolate(ufl_input)

    return mat_property


def _initialize_material_property_from_func(wave, property_name,
                                            fire_function, V):
    """
    Initialize material property from a firedrake function.

    Parameters:
    -----------
    property_name: `str`
        Name of the material property to be set.
    fire_function: `firedrake function`
        Firedrake function based on the input object
    V: `firedrake function space`
        Function space for the material property

    Returns:
    --------
    mat_property: `firedrake function`
        Material property
    """

    original_family = wave.function_space.ufl_element().family()
    original_degree = wave.function_space.ufl_element().degree()
    element_family = V.ufl_element().family()
    element_degree = V.ufl_element().degree()

    print(f"Assigning {property_name} with a firedrake function",
          ("in the same" if element_family == original_family
           and element_degree == original_degree else "in another"),
          f"function space: {element_family} {element_degree}.", flush=True)

    if element_family == original_family and element_degree == original_degree:
        # Same function space
        mat_property = fire_function
        mat_property.rename(property_name)

    else:  # Different function space
        mat_property = fire.Function(
            V, name=property_name).interpolate(fire_function)

    return mat_property


def _initialize_random_material_prop(property_name, random, V):
    """
    Initialize material property from a random distribution.

    Parameters:
    -----------
    property_name: `str`
        Name of the material property to be set.
    random: `tuple`
        If you want to set a random material property, specify the range of
        values as a tuple (min, max)
    V: `firedrake function space`
        Function space for the material property

    Returns:
    --------
    mat_property: `firedrake function`
        Material property
    """

    col0 = int(abs(log10(abs(random[0])))) + 2
    col1 = int(abs(log10(abs(random[1])))) + 2
    print(f"Assigning {property_name} with a random field "
          f"between ({random[0]:>{col0}},{random[1]:>{col1}})",
          flush=True)

    mat_property = fire.Function(V, name=property_name)
    mat_property.dat.data[:] = uniform(random[0], random[1],
                                       mat_property.dat.data.shape)

    return mat_property


def _initialize_material_property_from_file(
    wave,
    property_name,
    from_file,
    V,
    fast_interpolate=False,
):
    """
    Initialize a scalar material property from a file or grid data.

    The source may be a gridded dictionary (already in memory), an HDF5/H5
    file, or a SEGY file. SEGY inputs are first converted to HDF5 with
    :func:`write_hdf5_velocity_model`. The model is then transferred onto the
    finite-element space ``V`` by :func:`interpolate`.

    Parameters
    ----------
    wave : spyro Wave object
        The wave solver that owns the mesh, communicator and mesh parameters
        used while reading and transferring the model.
    property_name : `str`
        Name of the material property being set (used only in log messages).
    from_file : `str` or `dict`
        Path to a ``*.segy``, ``*.hdf5`` or ``*.h5`` file, or a grid
        dictionary holding the model values already loaded in memory.
    V : `firedrake.FunctionSpace`
        Function space onto which the property is transferred.
    fast_interpolate : `bool`, optional
        Selects how the gridded model is transferred onto ``V``. When
        ``False`` (default) the model is read and L2-projected onto ``V``,
        which is robust and smooths between the model grid and the mesh. When
        ``True`` the model values are sampled directly at the node coordinates
        of ``V``, which skips the projection solve and is faster, but relies on
        point sampling of the grid. Only meaningful for file/grid inputs.

    Returns
    -------
    mat_property : `firedrake.Function`
        The material property interpolated onto ``V``.

    Raises
    ------
    ValueError
        If ``from_file`` is a path that is not a ``*.segy``, ``*.hdf5`` or
        ``*.h5`` file.
    """

    original_family = wave.function_space.ufl_element().family()
    original_degree = wave.function_space.ufl_element().degree()
    element_family = V.ufl_element().family()
    element_degree = V.ufl_element().degree()

    source_label = from_file if isinstance(from_file, str) else "grid velocity data"
    print(f"Assigning {property_name} from {source_label}",
          ("in the same" if element_family == original_family
           and element_degree == original_degree else "in another"),
          f"function space: {element_family} {element_degree}.", flush=True)

    if isinstance(from_file, dict):
        return interpolate(
            wave,
            from_file,
            V,
            fast_interpolate=fast_interpolate,
        )

    if from_file.endswith(".segy"):
        from_file = write_hdf5_velocity_model(wave, from_file)

    if not from_file.endswith((".hdf5", ".h5")):
        raise ValueError(
            f"Cannot initialize {property_name} from {from_file!r}: "
            "expected a '*.segy', '*.hdf5' or '*.h5' file, or grid data.",
        )

    return interpolate(
        wave,
        from_file,
        V,
        fast_interpolate=fast_interpolate,
    )


def _saving_property_to_file(wave, mat_property, property_name,
                             foldername="default"):
    """
    Save a material property to a pvd file for visualization.

    Parameters:
    -----------
    mat_property: `firedrake function`
        Material property
    property_name: `str`
        Name of the material property to be set.
    foldername : `string`, optional
        Name of the folder where the material property is saved.
        If default is 'default', property is saved in '/property_fields/'

    Returns:
    --------
    None
    """

    # Path to save data
    wave.path_save_matprop = getcwd() + ("/property_fields/"
                                         if foldername == "default"
                                         else foldername)
    pth_prop = wave.path_save_matprop + property_name + ".pvd"
    print(f"Saving {property_name} to {foldername} for visualization.")
    fire.VTKFile(pth_prop).write(mat_property, name=property_name)


def _check_material_property_inputs(val_lst, func_space_type,
                                    shape_func_space, output):
    """
    Validate that exactly one value source was given for a material property.

    A material property must be defined by a single source. This guard rejects
    two invalid cases before any function space is built:

    - **zero sources** (all entries of ``val_lst`` are ``None``): previously
      this silently fell through and later failed with an obscure
      ``UnboundLocalError`` on ``mat_property``; it now raises a clear
      ``ValueError`` naming the accepted sources;
    - **more than one source**: raises a ``ValueError`` listing the sources
      that were supplied together.

    Parameters
    ----------
    val_lst : `list`
        Values for ``constant``, ``conditional``, ``expression``, ``random``,
        ``fire_function`` and ``from_file``, in this order. Unused sources are
        ``None``.
    func_space_type : `str`
        Type of function space for the material property. One of ``'scalar'``,
        ``'vector'`` or ``'tensor'``.
    shape_func_space : `tuple`, optional
        Shape of the function space, only for tensorial properties. Default is
        ``None``.
    output : `bool`, optional
        If ``True`` the property is later written to a PVD file for
        visualization. Default is ``False``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no source or more than one source is provided.
    """

    number_of_sources = sum(value is not None for value in val_lst)
    if number_of_sources == 0:
        raise ValueError(
            "A material property requires exactly one value source: "
            "constant, conditional, expression, random, fire_function or "
            "from_file.",
        )
    if number_of_sources > 1:
        name_lst = ["constant", "conditional", "expression",
                    "random", "fire_function", "from_file"]
        name_lst[-1] += " (*.segy or *.hdf5)"
        error_management.mutually_exclusive_parameter_error(name_lst,
                                                            val_lst)
    if shape_func_space is not None:
        if func_space_type != "tensor":
            raise ValueError("'shape_func_space' can only be specified "
                             "for tensorial material properties.")
        if shape_func_space[0] * shape_func_space[1] > 9 and output:
            raise ValueError("Output of tensorial material "
                             "properties with more than 9 "
                             "components is not supported.")


def set_material_property(wave, property_name, func_space_type,
                          shape_func_space=None, constant=None,
                          conditional=None, expression=None,
                          random=None, fire_function=None,
                          from_file=None, dg_property=False,
                          output=False, foldername="default",
                          fast_interpolate=False):
    """
    Set a material property(e.g., density, etc.) in the model.

    Parameters:
    -----------
    wave: spyro Wave object
        The wave solver that owns the mesh and function spaces onto which the
        property is defined.
    property_name: `str`
        Name of the material property to be set.
    func_space_type, `str`
        Type of function space for the material property.
        Options: 'scalar', 'vector' or 'tensor'
    shape_func_space: `tuple`, optional
        Shape of the function space for only tensorial material property.
        Default is None
    from_file: `str` or `dict`, optional
        Name of the file containing the material property, or grid data.
        Default is None.
    constant: `float`, optional
        Constant value for the material property. Default is None
    conditional: `firedrake conditional`, optional
        Firedrake conditional object. Default is None
    expression: `str`, optional
        If you use an expression, you can use the following variables:
        x, y, z, pi, tanh, sqrt. Ex: "2. + 0.5 * tanh((x - 2.) / 0.1)".
        It will be interpoalte into either the same function space as
        the object or a DG0 function space in the same mesh.
        Default is None
    random: `tuple`, optional
        If you want to set a random material property, specify the range of
        values as a tuple(min, max). Default is None
    fire_function: `firedrake function`, optional
        Firedrake function based on the input object. Default is None.
    dg_property: `bool`, optional
        If True, uses a DG0 function space for conditional and
        expression inputs. Default is False
    output: `bool`, optional
        If True, outputs the material property to a pvd file for
        visualization. Default is False
    foldername: `string`, optional
        Name of the folder where the material property is saved.
        If default is 'default', property is saved in '/property_fields/'
    fast_interpolate: `bool`, optional
        Only used for ``from_file`` (grid/HDF5) inputs; ignored otherwise.
        Selects how the gridded model is transferred onto the target space.
        When False (default), the model is read and L2-projected onto the
        space, which is robust and smooths differences between the model grid
        and the mesh. When True, the model values are sampled directly at the
        node coordinates of the space, skipping the projection solve: this is
        faster but relies on point sampling of the grid. Default is False.

    Returns:
    -----------
    mat_property: `firedrake function`
        Material property
    """

    # Checking input arguments
    val_lst = [constant, conditional, expression,
               random, fire_function, from_file]

    _check_material_property_inputs(val_lst, func_space_type,
                                    shape_func_space, output)

    V = define_property_function_space(
        wave, func_space_type, dg_property, shape_func_space=shape_func_space)

    # If no mesh is set, we have to do it beforehand
    if wave.mesh is None:
        wave.set_mesh()

    if func_space_type == "scalar":

        if any(v is not None for v in val_lst[:3]):  # UFL
            mat_property = _initialize_material_property_from_ufl(
                wave, property_name, func_space_type, V, constant=constant,
                conditional=conditional, expression=expression)

        if random is not None:  # Random
            mat_property = _initialize_random_material_prop(
                property_name, random, V)

        if fire_function is not None:
            mat_property = _initialize_material_property_from_func(
                wave, property_name, fire_function, V)

        if from_file is not None:
            mat_property = _initialize_material_property_from_file(
                wave,
                property_name,
                from_file,
                V,
                fast_interpolate=fast_interpolate,
            )

    else:

        print("Vectorial and Tensorial material properties are "
              "defined only either by constants or\nby firedrake "
              "functions. If use 'constant', define its components "
              "as scalar material\nproperties using the same property "
              "name followed by the component at the end.")

        if constant is not None:
            mat_property = _initialize_material_property_from_ufl(
                wave, property_name, func_space_type, V,
                shape_func_space=shape_func_space,
                constant=constant, conditional=None,
                expression=None)

        if fire_function is not None:
            mat_property = _initialize_material_property_from_func(
                wave, property_name, fire_function, V)

    if output:
        _saving_property_to_file(wave, mat_property, property_name,
                                 foldername=foldername)

    return mat_property


def set_material_properties(wave, *args, **kwargs):
    """Deprecated alias for :func:`set_material_property`.

    .. deprecated::
        Use :func:`set_material_property` instead. This wrapper forwards every
        argument unchanged and will be removed in a future release.
    """
    warnings.warn(
        "set_material_properties() is deprecated; use "
        "set_material_property() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return set_material_property(wave, *args, **kwargs)


def point_to_scalar_wave_function_space(wave):
    # Check if wave.function_space is a generates vector os scalar fields:
    original_function_space_type = check_function_space_type(wave.function_space)
    if wave.scalar_function_space is not None:
        return wave.scalar_function_space
    elif original_function_space_type == "vector":
        vector_element = wave.function_space.ufl_element()
        element = vector_element.sub_elements[0]
        wave.scalar_function_space = create_function_space(
            wave.function_space.mesh(), element)
        return wave.scalar_function_space
    else:
        raise ValueError(f"Should not create a new FunctionSpace from {original_function_space_type}")


def point_to_vector_wave_function_space(wave):
    # Check if wave.function_space is a generates vector os scalar fields:
    original_function_space_type = check_function_space_type(wave.function_space)
    if wave.vector_function_space is not None:
        return wave.vector_function_space
    elif original_function_space_type == "scalar":
        wave.vector_function_space = fire.VectorFunctionSpace(
            wave.function_space.mesh(), wave.function_space.ufl_element())
        return wave.vector_function_space
    else:
        raise ValueError(f"Should not create a new VectorFunctionSpace from {original_function_space_type}")


def point_to_dg_scalar_wave_function_space(wave):
    if wave.dg0_scalar_function_space is not None:
        return wave.dg0_scalar_function_space
    else:
        wave.dg0_scalar_function_space = create_function_space(
            wave.function_space.mesh(), "DG0", 0)
        return wave.dg0_scalar_function_space


def point_to_dg_vector_wave_function_space(wave):
    if wave.dg0_vector_function_space is not None:
        return wave.dg0_vector_function_space
    else:
        wave.dg0_vector_function_space = fire.VectorFunctionSpace(
            wave.function_space.mesh(), "DG", 0)
        return wave.dg0_vector_function_space


def point_to_correct_tensor_space(wave, shape_func_space, is_dg):
    if wave.tensor_function_space0 is None:
        wave.tensor_function_space0_shape = shape_func_space
        wave.tensor_function_space0 = set_tensor_function_space(wave, shape_func_space, is_dg)
        return wave.tensor_function_space0
    elif wave.tensor_function_space0_shape == shape_func_space:
        return wave.tensor_function_space0
    elif wave.tensor_function_space1 is None:
        wave.tensor_function_space1_shape = shape_func_space
        wave.tensor_function_space1 = set_tensor_function_space(wave, shape_func_space, is_dg)
        return wave.tensor_function_space1
    elif wave.tensor_function_space1_shape == shape_func_space:
        return wave.tensor_function_space1
    else:
        tensor_function_spaces_lst = [wave.tensor_function_space0, wave.tensor_function_space1]
        tensor_space_shapes_lst = [wave.tensor_function_space0_shape, wave.tensor_function_space1_shape]
        for space_id in range(len(tensor_function_spaces_lst)):
            if shape_func_space == tensor_space_shapes_lst[space_id]:
                return tensor_function_spaces_lst[space_id]
        raise ValueError("More than 2 tensor function spaces not yet supported")


def set_tensor_function_space(wave, shape_func_space, is_dg):
    if wave.mesh_parameters.quadrilateral:  # Q_Elements
        base_mesh = wave.mesh._base_mesh
        base_cell = base_mesh.ufl_cell()
        element = wave.function_space.ufl_element()

        if is_dg:
            # DG case: build a degree-0 DQ ⊗ DG element from scratch.
            element_zx = fire.FiniteElement("DQ", base_cell, 0)
            element_y = fire.FiniteElement("DG", fire.interval, 0)
            tensor_element = fire.TensorProductElement(element_zx, element_y)
        else:
            # Non-DG case: VectorElement.sub_elements[0] is the scalar
            # TensorProductElement that was used to build this function space.
            # In new Firedrake/UFL, TensorProductElement.sub_elements is empty
            # so we can no longer reconstruct the element from its nested
            # factor elements. Instead, reuse the element directly — it already
            # has the correct family, degree and variant baked in.
            tensor_element = element.sub_elements[0]

        # Function space for the property
        V = fire.TensorFunctionSpace(
            wave.mesh,
            tensor_element,
            shape=shape_func_space,
        )
        return V

    else:  # T_Elements
        element_family = "DG" if is_dg else \
            wave.function_space.ufl_element().family()
        element_degree = 0 if is_dg else \
            wave.function_space.ufl_element().degree()

        # Function space for the property
        V = fire.TensorFunctionSpace(
            wave.mesh,
            element_family,
            element_degree,
            shape=shape_func_space,
        )
        return V


class VelocityModelFileIO:
    def __init__(self):
        """
        Initialize VelocityModelFileIO.
        """
        self.velocity_model_file = None
        if self.input_dictionary is not None:
            synthetic_data = self.input_dictionary.get("synthetic_data", {})
            self.initial_velocity_model_file = synthetic_data.get("real_velocity_file")
