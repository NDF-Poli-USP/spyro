import warnings
import os


class Read_options:
    """
    Read the options section of the dictionary.

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
        options_dictionary = dictionary["options"]
        options_dictionary.setdefault("method", None)
        options_dictionary.setdefault("cell_type", None)
        options_dictionary.setdefault("variant", None)
        options_dictionary.setdefault("degree", None)
        options_dictionary.setdefault("dimension", None)
        options_dictionary.setdefault("automatic_adjoint", False)
        self.options_dictionary = options_dictionary

        self.method = options_dictionary["method"]
        self.variant = options_dictionary["variant"]
        self.cell_type = options_dictionary["cell_type"]
        self.degree = options_dictionary["degree"]
        self.dimension = options_dictionary["dimension"]

    @property
    def variant(self):
        return self._variant
    
    @variant.setter
    def variant(self, value):
        accepted_variants = ["lumped", "equispaced", "DG", None]
        if value not in accepted_variants:
            raise ValueError(f"Variant of {value} is not valid.")
        self._variant = value
        
    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, value):
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
        if value in mlt_equivalents:
            self._method = "mass_lumped_triangle"
        elif value in sem_equivalents:
            self._method = "spectral_quadrilateral"
        elif value in dg_t_equivalents:
            self._method = "DG_triangle"
        elif value in dg_q_equivalents:
            self._method = "DG_quadrilateral"
        elif value == "DG":
            raise ValueError(
                "DG is not a valid method. Please specify \
                either DG_triangle or DG_quadrilateral."
            )
        elif value == "CG":
            if "variant" in self.options_dictionary and "cell_type" in self.options_dictionary:
                self._method = "CG"
            else:
                raise ValueError("Cant use CG without specifying cell type and variant.")
        elif value is None:
            self._method = None
        else:
            raise ValueError(f"Method of {value} is not valid.")

    @property
    def cell_type(self):
        return self._cell_type
    
    @cell_type.setter
    def cell_type(self, value):
        triangle_equivalents = [
            "T", "triangle", "triangles", "tetrahedra", "tetrahedron"
        ]
        triangle_methods = [
            "mass_lumped_triangle", "DG_triangle", "CG"
        ]
        quadrilateral_equivalents = [
            "Q", "quadrilateral", "quadrilaterals", "hexahedra", "hexahedron"
        ]
        quadrilateral_methods = [
            "spectral_quadrilateral", "DG_quadrilateral", "CG"
        ]

        if value is None:
            self._cell_type = None
            return

        if value in triangle_equivalents:
            canonical = "triangle"
            if self.method is not None and self.method not in triangle_methods:
                raise ValueError(
                    f"Cell type '{canonical}' is not compatible with method '{self.method}'."
                )
            self._cell_type = canonical
        elif value in quadrilateral_equivalents:
            canonical = "quadrilateral"
            if self.method is not None and self.method not in quadrilateral_methods:
                raise ValueError(
                    f"Cell type '{canonical}' is not compatible with method '{self.method}'."
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
                raise ValueError(f"Cell type of {canonical} not compatible with variant {self.variant}")

    @property
    def degree(self):
        return self._degree
    
    @degree.setter
    def degree(self, value):
        if not isinstance(value, int):
            raise ValueError("Degree has to be integer")
        self._degree = value
    
    @property
    def dimension(self):
        return self._dimension
    
    @dimension.setter
    def dimension(self, value):
        if value not in {2, 3}:
            raise ValueError(f"Dimension of {value} not 2 or 3.")
        self._dimension = value
