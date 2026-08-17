"""Class for reading option section int he input dictionary."""

from ..utils.error_management import validate_parameter


class Read_options:
    """Read the options section of the dictionary.

    Attributes
    ----------
    options_dictionary : dict
        Dictionary containing the options information.
    cell_type : str
        The cell type to be used.
    method : str
        The FEM method to be used.
    variant : str
        The quadrature variant to be used.
    degree : int
        The polynomial degree of the FEM method.
    dimension : int
        The spatial dimension of the problem.
    automatic_adjoint : bool
        Whether to automatically compute the adjoint.
    analysis : `str`
        The type of analysis to be performed. Can be 'transient', 'modal' or 'eikonal'.

    Methods
    -------
    check_valid_degree()
        Check that the degree is valid for the method.
    _check_valid_degree_for_mlt()
        Check that the degree is valid for the MLT method.
    check_mismatch_cell_type_variant_method()
        Check that the user has not specified both the method and the cell type.
    get_from_method()
        Get the method, cell type and variant from the method.
    get_from_cell_type_variant()
        Get the method, cell type and variant from the cell type and variant.
    """

    def __init__(self, dictionary={}):
        """Initialize Read_options class."""
        options_dictionary = dictionary["options"]
        options_dictionary.setdefault("method", None)
        options_dictionary.setdefault("cell_type", None)
        options_dictionary.setdefault("variant", None)
        options_dictionary.setdefault("degree", None)
        options_dictionary.setdefault("dimension", None)
        options_dictionary.setdefault("automatic_adjoint", False)
        options_dictionary.setdefault("analysis", "transient")
        self.options_dictionary = options_dictionary

        self.variant = options_dictionary["variant"]
        self.method = options_dictionary["method"]
        self.cell_type = options_dictionary["cell_type"]
        self.degree = options_dictionary["degree"]
        self.dimension = options_dictionary["dimension"]
        self.analysis = options_dictionary["analysis"]

    @property
    def variant(self):
        """Str | None: Type of element variant for each cell.

        Accepted variants are:
            "lumped": Mass lumping, works for SEM and MLT
            "equispaced": Equispaced nodes
            "DG": Discontinuous Galerkin method
            None: Default/unspecified variant
        """
        return self._variant

    @variant.setter
    def variant(self, value: str | None):
        accepted_variants = ["lumped", "equispaced", "DG", None]
        self._variant = validate_parameter("variant", value, accepted_variants)

    @property
    def method(self):
        """Str | None: FE method used for wave propagation.

        Accepted methods:
            'mass_lumped_triangle',
            'spectral_quadrilateral',
            'DG_triangle',
            'DG_quadrilateral',
            'CG', or
            None.

        Notes
        -----
        ----_
        The 'mass_lumped_triangle' has equivalents:
            -"KMV", "MLT", "mass_lumped_triangle", "mass_lumped_tetrahedra".
        The 'spectral_quadrilateral' has equivalents:
            -"spectral", "SEM", "spectral_quadrilateral".
        """
        return self._method

    @method.setter
    def method(self, value: str | None):
        mlt_equivalents = [
            "KMV",
            "MLT",
            "mass_lumped_triangle",
            "mass_lumped_tetrahedra",
        ]
        sem_equivalents = ["spectral", "SEM", "spectral_quadrilateral"]
        dg_t_equivalents = [
            "DG_triangle",
            "DGT",
            "discontinuous_galerkin_triangle",
        ]
        dg_q_equivalents = [
            "DG_quadrilateral",
            "DGQ",
            "discontinuous_galerkin_quadrilateral",
        ]

        # Check if the provided value is in any of the accepted methods
        validate_parameter(
            "method",
            value,
            mlt_equivalents
            + sem_equivalents
            + dg_t_equivalents
            + dg_q_equivalents
            + ["CG", None],
        )

        if value in mlt_equivalents:
            self._method = "mass_lumped_triangle"
            self.cell_type = "triangle"
        elif value in sem_equivalents:
            self._method = "spectral_quadrilateral"
            self.cell_type = "quadrilateral"
        elif value in dg_t_equivalents:
            self._method = "DG_triangle"
            self.cell_type = "triangle"
        elif value in dg_q_equivalents:
            self._method = "DG_quadrilateral"
            self.cell_type = "quadrilateral"
        elif value == "CG":
            options = self.input_dictionary["options"]
            if not ("variant" in options and "cell_type" in options):
                raise ValueError(
                    "Cant use CG without specifying cell type and variant."
                )
            self._method = "CG"
        elif value is None:
            self._method = None

    @property
    def cell_type(self):
        """Str | None: Cell type used for wave propagation.

        Accepted cell geometry:
            'triangle',
            'quadrilateral', or
            None.

        Notes
        -----
        The 'triangle' geometry has equivalents:
        - "T", "triangle", "triangles", "tetrahedra", "tetrahedron".
        The 'quadrilateral' geometry has equivalents:
        - "Q", "quadrilateral", "quadrilaterals", "hexahedra", "hexahedron".
        """
        return self._cell_type

    @cell_type.setter
    def cell_type(self, value: str | None):
        triangle_equivalents = [
            "T",
            "triangle",
            "triangles",
            "tetrahedra",
            "tetrahedron",
        ]
        triangle_methods = ["mass_lumped_triangle", "DG_triangle", "CG"]
        quadrilateral_equivalents = [
            "Q",
            "quadrilateral",
            "quadrilaterals",
            "hexahedra",
            "hexahedron",
        ]
        quadrilateral_methods = ["spectral_quadrilateral", "DG_quadrilateral", "CG"]

        if value is None:
            self._cell_type = None
            return

        if value in triangle_equivalents:
            canonical = "triangle"
            if self.method is not None and self.method not in triangle_methods:
                raise ValueError(
                    f"Cell type '{canonical}' is not "
                    f"compatible with method '{self.method}'."
                )
            self._cell_type = canonical
        elif value in quadrilateral_equivalents:
            canonical = "quadrilateral"
            if self.method is not None and self.method not in quadrilateral_methods:
                raise ValueError(
                    f"Cell type '{canonical}' is not "
                    f"compatible with method '{self.method}'."
                )
            self._cell_type = canonical
        else:
            raise ValueError(f"Cell type '{value}' is not supported.")

        if self.variant is not None and self.method is None:
            if self.variant == "lumped" and canonical == "triangle":
                self.method = "mass_lumped_triangle"
            elif self.variant == "DG" and canonical == "triangle":
                self.method = "DG_triangle"
            elif self.variant == "equispaced" and canonical == "triangle":
                self.method = "CG"
            elif self.variant == "lumped" and canonical == "quadrilateral":
                self.method = "spectral_quadrilateral"
            elif self.variant == "DG" and canonical == "quadrilateral":
                self.method = "DG_quadrilateral"
            elif self.variant == "equispaced" and canonical == "quadrilateral":
                self.method = "CG"
            else:
                raise ValueError(
                    f"Cell type of {canonical} not "
                    f"compatible with variant {self.variant}."
                )

    @property
    def degree(self):
        """Int: FEM polynomial basis degree."""
        return self._degree

    @degree.setter
    def degree(self, value: int):
        self._degree = value

    @property
    def dimension(self):
        """Int: Geometric dimension.

        Should only be 2 or 3.
        """
        return self._dimension

    @dimension.setter
    def dimension(self, value: int):
        self._dimension = validate_parameter("dimension", value, [2, 3])

    @property
    def analysis(self):
        """Analysis type."""
        return self._analysis

    @analysis.setter
    def analysis(self, value: str | None):
        allowed_analyses = ["transient", "modal", "eikonal"]
        self._analysis = validate_parameter("analysis", value, allowed_analyses)


class Read_outputs:
    """Read and normalize visualization/output options.

    This reader fills default values in ``self.input_dictionary["visualization"]``
    and exposes them as attributes for downstream use.

    Attributes
    ----------
    forward_output : bool
        Whether forward wavefield output should be written.
    forward_output_filename : str
        Output filename used for forward results.
    gradient_output : bool
        Whether gradient output should be written.
    gradient_filename : str
        Output filename used for gradient results.
    adjoint_output : bool
        Whether adjoint wavefield output should be written.
    adjoint_filename : str
        Output filename used for adjoint results.
    debug_output : bool
        Whether debug output should be written.

    Notes
    -----
    ``self.input_dictionary`` is expected to exist before initialization and
    contain the global input dictionary used by the I/O readers.
    """

    def __init__(self):
        """Populate visualization/output options with defaults."""
        v_str = "visualization"
        self.input_dictionary.setdefault(v_str, {})

        # Forward output
        self.input_dictionary[v_str].setdefault("forward_output", False)
        self.forward_output = self.input_dictionary[v_str]["forward_output"]

        self.input_dictionary[v_str].setdefault(
            "forward_output_filename", "results/forward.pvd"
        )
        self.forward_output_filename = self.input_dictionary[v_str][
            "forward_output_filename"
        ]

        # General output folder
        self.input_dictionary[v_str].setdefault("output_folder", "output")
        self.output_folder = self.input_dictionary[v_str]["output_folder"]

        # Gradient output
        self.input_dictionary[v_str].setdefault("gradient_output", False)
        self.gradient_output = self.input_dictionary[v_str]["gradient_output"]
        self.input_dictionary[v_str].setdefault(
            "gradient_filename",
            "results/gradient.pvd",
        )
        self.gradient_filename = self.input_dictionary[v_str]["gradient_filename"]

        # Adjoint output
        self.input_dictionary[v_str].setdefault("adjoint_output", False)
        self.adjoint_output = self.input_dictionary[v_str]["adjoint_output"]
        self.input_dictionary[v_str].setdefault(
            "adjoint_filename",
            "results/adjoint.pvd",
        )
        self.adjoint_filename = self.input_dictionary[v_str]["adjoint_filename"]

        # Debug output
        self.input_dictionary[v_str].setdefault("debug_output", False)
        self.debug_output = self.input_dictionary[v_str]["debug_output"]
