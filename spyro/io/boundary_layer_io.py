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


class read_boundary_layer:
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
    abc_boundary_layer_type : str
        Type of the boundary layer

    Methods
    -------
    read_PML_dictionary()
        Read the PML dictionary for a perfectly matched layer
    """

    def __init__(self, abc_dictionary):
        self.dictionary = abc_dictionary
        if self.dictionary["status"] is False:
            self.abc_exponent = None
            self.abc_cmax = None
            self.abc_R = None
            self.abc_pad_length = 0.0
            self.abc_boundary_layer_type = None
            pass
        elif self.dictionary["damping_type"] == "PML":
            self.abc_boundary_layer_type = self.dictionary["damping_type"]
            self.read_PML_dictionary()
        else:
            abc_type = self.dictionary["damping_type"]
            raise ValueError(
                f"Boundary layer type of {abc_type} not recognized"
            )

    def read_PML_dictionary(self):
        """
        Reads the PML dictionary for a perfectly matched layer
        """
        self.abc_exponent = self.dictionary["exponent"]
        self.abc_cmax = self.dictionary["cmax"]
        self.abc_R = self.dictionary["R"]
        self.abc_pad_length = self.dictionary["pad_length"]
