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

    Methods
    -------
    read_PML_dictionary()
        Read the PML dictionary for a perfectly matched layer
    """

    @property
    def damping_type(self):
        return self._damping_type
    
    @damping_type.setter
    def damping_type(self, value):
        abc_dictionary = self.input_dictionary
        accepted_damping_types = [
            "PML",
            "local",
            None,
        ]
        if value not in accepted_damping_types:
            return ValueError(f"Damping type of {value} not recognized.")
        if value == "PML":
            abc_dictionary.setdefault("exponent", 2)
            abc_dictionary.setdefault("R", 1e-6)
            abc_dictionary.setdefault("cmax", 4.7)
            self.abc_exponent = abc_dictionary["exponent"]
            self.abc_R = abc_dictionary["R"]
            self.abc_cmax = abc_dictionary["cmax"]
        self._damping_type = value
    
    @property
    def abc_pad_length(self):
        return self._abc_pad_length
    
    @abc_pad_length.setter
    def abc_pad_length(self, value):
        if (value is None or value == 0) and self.damping_type == "PML":
            raise ValueError(f"No pad not compatible with PML")
        self._abc_pad_length = value
