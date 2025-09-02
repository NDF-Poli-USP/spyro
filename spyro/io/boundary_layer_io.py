# # Specify a 250-m PML on the three sides of the
# # domain to damp outgoing waves.
# default_dictionary["absorving_boundary_conditions"] = {
#     "status": False,  # True or false
# #  None or non-reflective (outer boundary condition)
#     "outer_bc": "non-reflective",
# # polynomial, hyperbolic, shifted_hyperbolic
#     "damping_type": "polynomial",
#     "exponent": 2,  # damping layer has a exponent variation
#     "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
#     "R": 1e-6,  # theoretical reflection coefficient
# # thickness of the PML in the z-direction (km) - always positive
#     "lz": 0.25,
# # thickness of the PML in the x-direction (km) - always positive
#     "lx": 0.25,
# # thickness of the PML in the y-direction (km) - always positive
#     "ly": 0.0,
# }


class Read_boundary_layer:
    """
    Read the boundary layer dictionary

    Attributes
    ----------
    dictionary : dict
        Dictionary containing the boundary layer information
    abc_exponent : float
        Exponent of the polynomial damping
    abc_cmax : float
        Maximum acoustic wave velocity in PML - km/s
    abc_R : float
        Theoretical reflection coefficient
    abc_pad_length : float
        Thickness of the PML in the z-direction (km) - always positive
    damping_type : str
        Type of the boundary layer
    abc_boundary_layer_type : `str`
        Type of the boundary layer. Option 'hybrid' is based on paper
        of Salas et al. (2022). doi: https://doi.org/10.1016/j.apm.2022.09.014
    abc_boundary_layer_shape : str
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    abc_deg_layer : `int`
        Hypershape degree
    abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer
        Options: 'source' or 'boundary'
    abc_deg_eikonal : `int`
        Finite element order for the Eikonal analysis
    abc_get_ref_model : `bool`
        If True, the infinite model is created

    Methods
    -------
    read_PML_dictionary()
        Read the PML dictionary for a perfectly matched layer
    """
    def __init__(self, dictionary=None, comm=None):
        self.input_dictionary.setdefault("absorving_boundary_conditions", {})
        self.input_dictionary["absorving_boundary_conditions"].setdefault("status", False)
        self.abc_active = self.input_dictionary["absorving_boundary_conditions"]["status"]
        self.input_dictionary["absorving_boundary_conditions"].setdefault("damping_type", None)
        self.input_dictionary["absorving_boundary_conditions"].setdefault("pad_length", None)
        self.abc_boundary_layer_type = self.input_dictionary["absorving_boundary_conditions"]["damping_type"]
        self.abc_pad_length = self.input_dictionary["absorving_boundary_conditions"]["pad_length"]

        self.absorb_top = dictionary["absorving_boundary_conditions"].get("absorb_top", False)
        self.absorb_bottom = dictionary["absorving_boundary_conditions"].get("absorb_bottom", True)
        self.absorb_right = dictionary["absorving_boundary_conditions"].get("absorb_right", True)
        self.absorb_left = dictionary["absorving_boundary_conditions"].get("absorb_left", True)
        self.absorb_front = dictionary["absorving_boundary_conditions"].get("absorb_front", True)
        self.absorb_back = dictionary["absorving_boundary_conditions"].get("absorb_back", True)

    @property
    def abc_boundary_layer_type(self):
        return self._abc_boundary_layer_type
    
    @abc_boundary_layer_type.setter
    def abc_boundary_layer_type(self, value):
        abc_dictionary = self.input_dictionary['absorving_boundary_conditions']
        accepted_damping_types = [
            "PML",
            "local",
            "hybrid",
            None,
        ]
        if value not in accepted_damping_types:
            raise ValueError(f"Damping type of {value} not recognized.")
        if value == "PML":
            abc_dictionary.setdefault("exponent", 2)
            abc_dictionary.setdefault("R", 1e-6)
            abc_dictionary.setdefault("cmax", 4.7)
            self.abc_exponent = abc_dictionary["exponent"]
            self.abc_R = abc_dictionary["R"]
            self.abc_cmax = abc_dictionary["cmax"]
        if value == "hybrid":
            abc_dictionary.setdefault("layer_shape", "rectangular")
            abc_dictionary.setdefault("degree_eikonal", None)
            self.abc_boundary_layer_shape = abc_dictionary["layer_shape"]
            self.abc_deg_layer = None \
                if abc_dictionary["layer_shape"] == "rectangular" \
                else abc_dictionary.get("degree_layer", 2)
            self.abc_reference_freq = abc_dictionary["habc_reference_freq"]
            self.abc_deg_eikonal = abc_dictionary.get("degree_eikonal", None)
            self.abc_get_ref_model = abc_dictionary["get_ref_model"]

        self._abc_boundary_layer_type = value
    
    @property
    def abc_pad_length(self):
        return self._abc_pad_length
    
    @abc_pad_length.setter
    def abc_pad_length(self, value):
        if (value is None or value == 0) and self.abc_boundary_layer_type == "PML":
            raise ValueError(f"No pad not compatible with PML")
        self._abc_pad_length = value
