"""Methods to label the case study of the ABC scheme."""
from numpy import pi
from os import getcwd
from ..io.basicio import parallel_print as pprint
from ..utils.error_management import (mutually_exclusive_parameter_error, validate_enum,
                                      validate_numeric, validate_parameter, validate_string)
from ..utils.typing import (AbsorbingBCsType, BoundaryConditionsType, LayerShapeType,
                            LayerSizeRefFrequency, NRBCBoundaryType)


def formatting_abc_layer_type(str_to_format, abc_type, for_prints=True):
    """Format a string for the ABC layer type.

    The formatted string can be used for printing on screen or to generate paths for
    output files. The `for_prints` parameter determines whether the formatted string
    is intended for printing or for labeling purposes.

    Parameters
    ----------
    str_to_format : `str`
        The string to format.
    abc_type : `typing.AbsorbingBCsType`
        Type of the absorbing boundary condition. Options: `AbsorbingBCsType.HYBRID` or
        `AbsorbingBCsType.PML`. Option `AbsorbingBCsType.HYBRID` is based on paper of
        Salas et al. (2022). doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation
    for_prints : `bool`, optional
        Flag to indicate whether the formatted string is for
        printing (`True`) or for labeling (`False`). Default is `True`.

    Returns
    -------
    formatted_str : `str`
            The formatted string for the ABC layer type.
    """

    # Validate input parameters
    validate_string("string to format", str_to_format, accept_parameter_as_none=True)

    # Checking ABC type (Only ABCs based on absorbing layers are included)
    validate_parameter("abc_type", abc_type, [AbsorbingBCsType.HYBRID, AbsorbingBCsType.PML])

    # Layer type
    if abc_type == AbsorbingBCsType.HYBRID:
        abc_layer_str = "Absorbing" if for_prints else "habc"
    elif abc_type == AbsorbingBCsType.PML:
        abc_layer_str = "PML" if for_prints else "pml"

    formatted_str = str_to_format.format(abc_layer_str)

    return formatted_str


def identify_abc_layer_case(abc_type, abc_boundary_layer_shape,
                            abc_deg_layer, abc_reference_freq):
    """Generate an identifier for the current layer geometry of the ABC.

    The identifier includes the layer shape ("REC" for rectangular layers or "HN"
    followed by the degree for hypershape layers) and the reference frequency for
    sizing the absorbing layer ('SOU': source frequency or 'BND': boundary frequency).
    The identifier can be used for labeling output files and directories.
    Examples: "REC_SOU", "REC_BND", "HN2.4_SOU" or "HN2.4_BND".

    Parameters
    ----------
    abc_type : `typing.AbsorbingBCsType`
        Type of the absorbing boundary condition. Options: `AbsorbingBCsType.HYBRID` or
        `AbsorbingBCsType.PML`. Option `AbsorbingBCsType.HYBRID` is based on paper of
        Salas et al. (2022). doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation
    abc_boundary_layer_shape : `typing.LayerShapeType`
        Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
        `LayerShapeType.HYPERSHAPE`.
    abc_deg_layer : `int` or `float` or `None`
        Hypershape degree. For hypershape layers, the degree must be greater than or
        equal to 2. `None` is used only for rectangular layers
    abc_reference_freq : `typing.LayerSizeRefFrequency`
        Reference frequency for sizing the absorbing layer.
        Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.

    Returns
    -------
    case_absl : `str`
        Label for the output files that includes the layer shape and degree for
        hypershape layers ("REC", "HNX.Y" with X.Y as the hypershape degree with one
        decimal place precision) and the reference frequency ('SOU' or 'BND').
        Examples: "REC_SOU", "REC_BND", "HN2.4_SOU" or "HN2.4_BND".
    """

    # Checking input parameters
    validate_enum("abc_boundary_layer_shape", abc_boundary_layer_shape, LayerShapeType)
    validate_numeric(
        'abc_deg_layer', abc_deg_layer, float_num=True, integer_num=True,
        accept_parameter_as_none=True, lower_bound=2., include_lower_bound=True)
    validate_enum("abc_reference_freq", abc_reference_freq, LayerSizeRefFrequency)

    # Labeling for the layer shape
    if abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:
        case_absl = "REC"

    elif abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
        case_absl = "HN" + f"{abc_deg_layer:.1f}"

    # Labeling for the reference frequency for the absorbing layer
    if abc_reference_freq == LayerSizeRefFrequency.SOURCE:
        case_absl += "_SOU"

    elif abc_reference_freq == LayerSizeRefFrequency.BOUNDARY:
        case_absl += "_BND"

    # Printing layer info on screen
    layer_str = formatting_abc_layer_type("\n{} Layer Shape: ", abc_type) + \
        f"{abc_boundary_layer_shape.value.capitalize()}" + (
        f" - Degree: {abc_deg_layer}"
        if abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE else "")
    pprint(layer_str)

    return case_absl


def identify_nrbc_case(non_reflect_bc, angle_max, abc_boundary_type):
    """Generate an identifier for the current type of the NRBC.

    The identifier includes the NRBC type ("HIG" for Higdon BCs or "SOM" for
    Sommerfeld BCs) and the boundary type where the NRBCs are applied ("STB"
    for straight boundaries or "HYP" for hypershape boundaries).
    The identifier can be used for labeling output files and directories.
    Examples: "HIG_STB", "HIG_HYP", "SOM_STB" or "SOM_HYB".

    Parameters
    ----------
    non_reflect_bc : `typing.BoundaryConditionsType`
        Type of boundary condition to apply on the domain boundaries (for only NRBCs)
        or the outer absorbing layer boundaries (HABCs: Absorbing Layer aand NRBCs).
        Options: `BoundaryConditionsType.HIGDON` or `BoundaryConditionsType.SOMMERFELD`.
    angle_max : `float`, optional
        Maximum incidence angle considered in the NRBC.
    abc_boundary_type : `typing.NRBCBoundaryType`
        Boundary type where NRBCs are applied . Options: `NRBCBoundaryType.STRAIGHT`
        or `NRBCBoundaryType.HYPERSHAPE`.

    Returns
    -------
    case_nrbc : `str`
        Label for the output files that includes the NRBC type ("HIG" or "SOM")
        and the boundary type where the NRBCs are applied ("STB" or "HYP").
        Examples: "HIG_STB", "HIG_HYP", "SOM_STB" or "SOM_HYB".
    """

    # Checking input parameters (TraditionaL BCs are not inlcuded)
    validate_parameter('non_reflect_bc', non_reflect_bc,
                       [BoundaryConditionsType.HIGDON, BoundaryConditionsType.SOMMERFELD])
    validate_numeric("angle_max", angle_max, float_num=True, integer_num=False,
                     lower_bound=0., include_lower_bound=True)
    validate_enum("abc_boundary_type", abc_boundary_type, NRBCBoundaryType)

    # Labeling for the NRBC type
    if non_reflect_bc == BoundaryConditionsType.HIGDON:
        case_nrbc = "HIG"

    elif non_reflect_bc == BoundaryConditionsType.SOMMERFELD:
        case_nrbc = "SOM"

    # Labeling for the boundary type
    if abc_boundary_type == NRBCBoundaryType.STRAIGHT:
        case_nrbc += "_STB"  # Straight Boundary

    elif abc_boundary_type == NRBCBoundaryType.HYPERSHAPE:
        case_nrbc += "_HYB"  # HyperShape Boundary

    # Printing NRBC info on screen
    nrbc_str = f"\nNRBC Type: {non_reflect_bc.value.capitalize()}" + \
        f"\nBoundary Type: {abc_boundary_type.value.capitalize()}" + \
        (f"\nMaximum Incidence Angle: {180. * angle_max / pi:.1f}°"
         if non_reflect_bc == BoundaryConditionsType.HIGDON else "")
    pprint(nrbc_str)

    return case_nrbc


def path_to_save_abc_case(abc_type, abc_boundary_layer_shape=None, abc_deg_layer=None,
                          abc_reference_freq=None, non_reflect_bc=None, angle_max=None,
                          abc_boundary_type=None, output_folder=None):
    """Create the path to save data for the current case study of the ABC scheme.

    Parameters
    ----------
    abc_type : `typing.AbsorbingBCsType`
        Type of the absorbing boundary condition. Options: `AbsorbingBCsType.NRBC`,
        `AbsorbingBCsType.HYBRID` or `AbsorbingBCsType.PML`.
        Option `AbsorbingBCsType.HYBRID` is based on paper of Salas et al. (2022).
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation
    abc_boundary_layer_shape : `typing.LayerShapeType`, optional
        Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
        `LayerShapeType.HYPERSHAPE`. Default is `None`.
    abc_reference_freq : `typing.LayerSizeRefFrequency`
        Reference frequency for sizing the absorbing layer.
        Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
        Default is 'None'.
    abc_deg_layer : `int` or `float` or `None`, optional
            Hypershape degree. For hypershape layers, the degree must be greater than or
            equal to 2. `None` is used only for rectangular layers. Default is `None`.
    non_reflect_bc : `typing.BoundaryConditionsType`, optional
        Type of boundary condition to apply on the domain boundaries (for only NRBCs)
        or the outer absorbing layer boundaries (HABCs: Absorbing Layer aand NRBCs).
        Options: `BoundaryConditionsType.HIGDON` or `BoundaryConditionsType.SOMMERFELD`.
        Dafault is `None`.
    angle_max : `float`, optional
        Maximum incidence angle considered in the NRBC. Default is `None`.
    abc_boundary_type : `typing.NRBCBoundaryType`, optional
        Boundary type where NRBCs are applied . Options: `NRBCBoundaryType.STRAIGHT`
        or `NRBCBoundaryType.HYPERSHAPE`. Default is `None`.
        output_folder : `str`, optional
        The folder where output data will be saved. Default is `None`.

    Returns
    -------
    path_save : `string`
        Path to save data.
    case_absl : `str`
        Label for the output files that includes the layer shape and degree for
        hypershape layers ("REC", "HNX.Y" with X.Y as the hypershape degree with one
        decimal place precision) and the reference frequency ('SOU' or 'BND').
        Examples: "REC_SOU", "REC_BND", "HN2.4_SOU" or "HN2.4_BND"
    path_case_absl : `string`
        Path to save data for the current case study of ABCs based on absorbing layers.
    case_nrbc : `str`
        Label for the output files that includes the NRBC type ("HIG" or "SOM")
        and the boundary type where the NRBCs are applied ("STB" or "HYP").
        Examples: "HIG_STB", "HIG_HYP", "SOM_STB" or "SOM_HYB".
    path_case_nrbc : `string`
        Path to save data for the current case study of NRBCs.
    """

    # Checking ABC type (Case `AbsorbingBCsType.NOABCS` is not included)
    validate_parameter("abc_type", abc_type, [AbsorbingBCsType.HYBRID,
                                              AbsorbingBCsType.PML,
                                              AbsorbingBCsType.NRBC])

    # Validate the output folder parameter
    validate_string("output_folder", output_folder, accept_parameter_as_none=True)

    # Path to save data
    path_save = getcwd() + "/output/" if output_folder is None \
        else getcwd() + "/" + output_folder + "/"

    if abc_type in [AbsorbingBCsType.PML, AbsorbingBCsType.HYBRID]:

        # For absorbing layer cases, ensure NRBC parameters are not provided
        mutually_exclusive_parameter_error(
            ["non_reflect_bc", "angle_max", "abc_boundary_type"],
            [non_reflect_bc, angle_max, abc_boundary_type])

        # Identify the Absorbing Layer (HABC ou PML) scheme for output labeling
        case_absl = identify_abc_layer_case(abc_type, abc_boundary_layer_shape,
                                            abc_deg_layer, abc_reference_freq)
        path_case_absl = path_save + case_absl + "/"

        return path_save, case_absl, path_case_absl

    elif abc_type == AbsorbingBCsType.NRBC:

        # For NRBC case, ensure ABC layer parameters are not provided
        mutually_exclusive_parameter_error(
            ["abc_boundary_layer_shape", "abc_reference_freq", "abc_deg_layer"],
            [abc_boundary_layer_shape, abc_reference_freq, abc_deg_layer])

        # Identify the type of the NRBC for output labeling
        case_nrbc = identify_nrbc_case(non_reflect_bc, angle_max, abc_boundary_type)
        path_case_nrbc = path_save + case_nrbc + "/"

        return path_save, case_nrbc, path_case_nrbc
