from ..io.basicio import parallel_print as pprint
from ..utils.error_management import (enum_parameter_error, value_numerical_error,
                                      value_parameter_error)
from ..utils.typing import HyperLayerDegreeType, LayerShapeType, LayerSizeRefFrequency


class Read_boundary_layer:
    """
    Read the boundary layer dictionary

    Attributes
    ----------
    abc_boundary_layer_shape : `typing.LayerShapeType`, optional
        Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
        `LayerShapeType.HYPERSHAPE`. Default is `LayerShapeType.RECTANGULAR`.
    abc_boundary_layer_type : `str`
        Type of the boundary layer. Options: 'hybrid' or 'PML'.
        Option 'hybrid' is based on paper of Salas et al. (2022).
        doi: https://doi.org/10.1016/j.apm.2022.09.014
    abc_deg_eikonal : `int`
        Finite element order for the Eikonal analysis
    abc_deg_layer : `int` or `float`
        Hypershape degree
    abc_degree_type : `typing.HyperLayerDegreeType`, optional
        Type of the hypereshape degree. Options: 'HyperLayerDegreeType.REAL' or
        'HyperLayerDegreeType.INTEGER'. Default is 'HyperLayerDegreeType.REAL'
    abc_extend_properties : `str`
        Mode to extend the properties into the absorbing layer.
        Options: 'abc_driven'  (performed by a specific method) or
        'builtin' (automatic at field definition)
    abc_get_ref_model : `bool`
        If True, the infinite model is created
    abc_pad_length : `float`
        Thickness of the PML in the z-direction (km) - always positive
    abc_pml_cmax: float
        Maximum propagation speed (km/s) in the PML layer. Default is 4.7 km/s.
    abc_pml_exponent: int
        Exponent for the polynomial damping profile of the PML layer. Default is 2.
    abc_pml_R: float
        Theoretical reflection coefficient of the PML layer. Default is 1e-6.
    abc_reference_freq : `typing.LayerSizeRefFrequency`, optional
        Reference frequency for sizing the absorbing layer.
        Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
        Default is 'LayerSizeRefFrequency.SOURCE'.
    abc_user_pad_length : `bool`
        If True, the pad length is provided by the user. If False,
        the pad length is determined with the HABC criterion.
    abc_user_pml_cmax : `bool`
        If True, the maximum propagation speed in the PML layer is provided by the user.
    damping_type : `str`
        Type of the boundary layer
    dictionary : `dict`
        Dictionary containing the boundary layer information
    """

    def __init__(self, comm=None):
        """
        Initialize the Read_boundary_layer class

        Parameters
        ----------
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # General parameters
        self.input_dictionary.setdefault("absorving_boundary_conditions", {})
        self.input_dictionary[
            "absorving_boundary_conditions"].setdefault("status", False)
        self.abc_active = self.input_dictionary[
            "absorving_boundary_conditions"]["status"]
        self.input_dictionary[
            "absorving_boundary_conditions"].setdefault("damping_type", None)
        self.abc_boundary_layer_type = self.input_dictionary[
            "absorving_boundary_conditions"]["damping_type"]
        self.input_dictionary[
            "absorving_boundary_conditions"].setdefault("pad_length", None)
        self.abc_pad_length = self.input_dictionary[
            "absorving_boundary_conditions"]["pad_length"]

        # Absorbing boundaries
        self.absorb_top = self.input_dictionary[
            "absorving_boundary_conditions"].get("absorb_top", False)
        self.absorb_bottom = self.input_dictionary[
            "absorving_boundary_conditions"].get("absorb_bottom", True)
        self.absorb_right = self.input_dictionary[
            "absorving_boundary_conditions"].get("absorb_right", True)
        self.absorb_left = self.input_dictionary[
            "absorving_boundary_conditions"].get("absorb_left", True)
        self.absorb_front = self.input_dictionary[
            "absorving_boundary_conditions"].get("absorb_front", True)
        self.absorb_back = self.input_dictionary[
            "absorving_boundary_conditions"].get("absorb_back", True)

    @property
    def abc_boundary_layer_shape(self):
        if not hasattr(self, "_abc_boundary_layer_shape"):
            self._abc_boundary_layer_shape = LayerShapeType.NOLAYER
        return self._abc_boundary_layer_shape

    @abc_boundary_layer_shape.setter
    def abc_boundary_layer_shape(self, value):
        """Set boundary layer shape with enum validation."""
        shape_enum = enum_parameter_error("abc_boundary_layer_shape",
                                          value, LayerShapeType)

        if shape_enum == LayerShapeType.NOLAYER:
            raise ValueError("NOLAYER not allowed for active ABC.")

        self._abc_boundary_layer_shape = shape_enum

    @property
    def abc_reference_freq(self):
        return self._abc_reference_freq

    @abc_reference_freq.setter
    def abc_reference_freq(self, value):
        """Set reference frequency for sizing the absorbing layer with enum validation."""
        reference_freq_enum = enum_parameter_error("abc_reference_freq", value,
                                                   LayerSizeRefFrequency)
        self._abc_reference_freq = reference_freq_enum

    @property
    def abc_degree_type(self):
        return self._abc_degree_type

    @abc_degree_type.setter
    def abc_degree_type(self, value):
        """Set hypershape degree type for hypershape layers with enum validation."""
        degree_type_enum = enum_parameter_error("abc_degree_type", value,
                                                HyperLayerDegreeType)
        self._abc_degree_type = degree_type_enum

    @property
    def abc_pml_exponent(self):
        return self._abc_pml_exponent

    @abc_pml_exponent.setter
    def abc_pml_exponent(self, value):
        """Set the exponent for the polynomial damping profile in PML with validation."""
        pml_exponent = value_numerical_error("abc_pml_exponent", value, integer_num=True,
                                             lower_bound=1, include_lower_bound=True)
        self._abc_pml_exponent = pml_exponent

    @property
    def abc_pml_R(self):
        return self._abc_pml_R

    @abc_pml_R.setter
    def abc_pml_R(self, value):
        """Set the theoretical reflection coefficient in the PML layer with validation."""
        pml_R = value_numerical_error("abc_pml_R", value, float_num=True, lower_bound=0.)
        self._abc_pml_R = pml_R

    @property
    def abc_pml_cmax(self):
        return self._abc_pml_cmax

    @abc_pml_cmax.setter
    def abc_pml_cmax(self, value):
        """Set the maximum propagation speed in the PML layer with validation."""
        self.abc_user_pml_cmax = True
        if value is None:
            pprint("Maximum propagation speed will get from model", comm=self.comm)
            self.abc_user_pml_cmax = False
            pml_cmax = value
        else:
            pml_cmax = value_numerical_error("abc_pml_cmax", value, float_num=True,
                                             integer_num=True, lower_bound=0.)
        self._abc_pml_cmax = value

    @property
    def abc_boundary_layer_type(self):
        return self._abc_boundary_layer_type

    @abc_boundary_layer_type.setter
    def abc_boundary_layer_type(self, value):
        """Set the type of absorbing boundary layer with validation."""
        abc_dictionary = self.input_dictionary['absorving_boundary_conditions']
        accepted_damping_types = [
            "PML",
            "local",
            "hybrid",
            None,
        ]

        # Cheking damping type input
        self._abc_boundary_layer_type = value_parameter_error(
            'abc_boundary_layer_type', value, accepted_damping_types)

        if value == "PML":
            # PML forces rectangular shape
            self.abc_boundary_layer_shape = LayerShapeType.RECTANGULAR

            # PML-specific parameters with defaults
            abc_dictionary.setdefault("exponent", 2)
            self.abc_pml_exponent = abc_dictionary["exponent"]
            abc_dictionary.setdefault("R", 1e-6)
            self.abc_pml_R = abc_dictionary["R"]
            abc_dictionary.setdefault("cmax", None)
            self.abc_pml_cmax = abc_dictionary["cmax"]
        if value == "hybrid":
            # Get shape from dictionary, default to rectangular
            self.abc_boundary_layer_shape = abc_dictionary.get("layer_shape", "rectangular")

            # Hypershape-specific validation
            self.abc_deg_layer = None
            if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
                self.abc_deg_layer = abc_dictionary.get("degree_layer", 2.)
                value_numerical_error(
                    'abc_deg_layer', self.abc_deg_layer, float_num=True,
                    integer_num=True, lower_bound=2., include_lower_bound=True)

            self.abc_degree_type = abc_dictionary.get("degree_type", "real")

        # Common parameters for both PML and hybrid
        self.abc_reference_freq = abc_dictionary.get("abc_reference_freq", "source")
        self.abc_deg_eikonal = abc_dictionary.get("degree_eikonal", 2)
        self.abc_get_ref_model = abc_dictionary.get("get_ref_model", False)
        self.abc_extend_properties = abc_dictionary.get("extend_properties", "abc_driven")

    @property
    def abc_pad_length(self):
        return self._abc_pad_length

    @abc_pad_length.setter
    def abc_pad_length(self, value):
        """Set the pad length for the absorbing boundary condition.

        Parameters
        ----------
        value : `float` or `None`
            The pad length in kilometers. If `None`, the pad length will be determined
            using the HABC criterion.

        Returns
        -------
        None

        Notes
        -----
        For the HABC criterion see Salas et al (2022): Hybrid absorbing scheme based on
        hyperelliptical layers with non-reflecting boundary conditions in scalar wave
        equations. doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation
        """

        self.abc_user_pad_len = True
        if value is None:
            pprint("Pad length will be determined with HABC criterion", comm=self.comm)
            pad_length = 0.
            self.abc_user_pad_len = False
        else:
            pad_length = value_numerical_error("abc_pad_length", value, float_num=True,
                                               integer_num=True, lower_bound=0.)
            pprint(f"Pad length provided by user (km): {pad_length:.4f}", comm=self.comm)
        self._abc_pad_length = pad_length
