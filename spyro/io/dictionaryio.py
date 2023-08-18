import warnings
import os


def parse_cg(dictionary):
    """
    Parse the CG method from the dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing the options information.

    Returns
    -------
    method : str
        The method to be used.
    cell_type : str
        The cell type to be used.
    variant : str
        The variant to be used.
    """
    if "variant" not in dictionary:
        raise ValueError("variant must be specified for CG method.")
    if dictionary["variant"] == "KMV":
        dictionary["cell_type"] = "T"
    if "cell_type" not in dictionary:
        raise ValueError("cell_type must be specified for CG method.")

    cell_type = None
    triangle_equivalents = ["T", "triangle", "triangles", "tetrahedra"]
    quadrilateral_equivalents = [
        "Q",
        "quadrilateral",
        "quadrilaterals, hexahedra",
    ]
    if dictionary["cell_type"] in triangle_equivalents:
        cell_type = "triangle"
    elif dictionary["cell_type"] in quadrilateral_equivalents:
        cell_type = "quadrilateral"
    elif dictionary["variant"] == "GLL":
        cell_type = "quadrilateral"
        warnings.warn(
            "GLL variant only supported for quadrilateral meshes. Assuming quadrilateral."
        )
    else:
        raise ValueError(
            f"cell_type of {dictionary['cell_type']} is not valid."
        )

    if dictionary["variant"] is None:
        warnings.warn("variant not specified for CG method. Assuming lumped.")
        dictionary["variant"] = "lumped"

    accepted_variants = ["lumped", "equispaced", "DG", "GLL", "KMV"]
    if dictionary["variant"] not in accepted_variants:
        raise ValueError(f"variant of {dictionary['variant']} is not valid.")

    variant = dictionary["variant"]

    if variant == "GLL":
        variant = "lumped"

    if variant == "KMV":
        variant = "lumped"

    if cell_type == "triangle" and variant == "lumped":
        method = "mass_lumped_triangle"
    elif cell_type == "triangle" and variant == "equispaced":
        method = "CG_triangle"
    elif cell_type == "triangle" and variant == "DG":
        method = "DG_triangle"
    elif cell_type == "quadrilateral" and variant == "lumped":
        method = "spectral_quadrilateral"
    elif cell_type == "quadrilateral" and variant == "equispaced":
        method = "CG_quadrilateral"
    elif cell_type == "quadrilateral" and variant == "DG":
        method = "DG_quadrilateral"
    return method, cell_type, variant


def check_if_mesh_file_exists(file_name):
    if file_name is None:
        return
    if os.path.isfile(file_name):
        return
    else:
        raise ValueError(f"Mesh file {file_name} does not exist.")


class read_options:
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
    def __init__(self, options_dictionary=None):
        default_dictionary = {
            # simplexes such as triangles or tetrahedra (T)
            # or quadrilaterals (Q)
            "cell_type": "T",
            # lumped, equispaced or DG, default is lumped
            "variant": "lumped",
            # (MLT/spectral_quadrilateral/DG_triangle/
            # DG_quadrilateral) You can either specify a cell_type+variant or a method
            "method": "MLT",
            # p order
            "degree": 4,
            # dimension
            "dimension": 2,
            "automatic_adjoint": False,
        }

        if options_dictionary is None:
            self.options_dictionary = default_dictionary
        else:
            self.options_dictionary = options_dictionary

        self.cell_type = None
        self.method = None
        self.overdefined_method = False
        self.variant = None
        self.degree = None
        self.dimension = None
        self.overdefined_method = self.check_mismatch_cell_type_variant_method()

        if "method" not in self.options_dictionary:
            self.options_dictionary["method"] = None

        if self.overdefined_method is True:
            self.method, self.cell_type, self.variant = self.get_from_method()
        elif self.options_dictionary["method"] is not None:
            self.method, self.cell_type, self.variant = self.get_from_method()
        else:
            (
                self.method,
                self.cell_type,
                self.variant,
            ) = self.get_from_cell_type_variant()

        if "degree" in self.options_dictionary:
            self.degree = self.options_dictionary["degree"]
        else:
            self.degree = default_dictionary["degree"]
            warnings.warn("Degree not specified, using default of 4.")

        if "dimension" in self.options_dictionary:
            self.dimension = self.options_dictionary["dimension"]
        else:
            self.dimension = default_dictionary["dimension"]
            warnings.warn("Dimension not specified, using default of 2.")

        if "automatic_adjoint" in self.options_dictionary:
            self.automatic_adjoint = self.options_dictionary[
                "automatic_adjoint"
            ]
        else:
            self.automatic_adjoint = default_dictionary["automatic_adjoint"]

        self.check_valid_degree()

    def check_valid_degree(self):
        if self.degree < 1:
            raise ValueError("Degree must be greater than 0.")
        if self.method == "mass_lumped_triangle":
            self._check_valid_degree_for_mlt()

    def _check_valid_degree_for_mlt(self):
        degree = self.degree
        dimension = self.dimension
        if dimension == 2 and degree > 5:
            raise ValueError(
                "Degree must be less than or equal to 5 for MLT in 2D."
            )
        elif dimension == 3 and degree > 4:
            raise ValueError(
                "Degree must be less than or equal to 4 for MLT in 3D."
            )
        elif dimension == 3 and degree == 4:
            warnings.warn(
                f"Degree of {self.degree} not supported by \
                    {self.dimension}D {self.method} in main firedrake."
            )

    def check_mismatch_cell_type_variant_method(self):
        dictionary = self.options_dictionary
        overdefined = False
        if "method" in dictionary and (
            "cell_type" in dictionary or "variant" in dictionary
        ):
            overdefined = True
        else:
            pass

        if overdefined:
            if dictionary["method"] is None:
                overdefined = False

        if overdefined:
            warnings.warn(
                "Both methods of specifying method and cell_type with \
                    variant used. Method specification taking priority."
            )
        return overdefined

    def get_from_method(self):
        dictionary = self.options_dictionary
        if dictionary["method"] is None:
            raise ValueError("Method input of None is invalid.")

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
        if dictionary["method"] in mlt_equivalents:
            method = "mass_lumped_triangle"
            cell_type = "triangle"
            variant = "lumped"
        elif dictionary["method"] in sem_equivalents:
            method = "spectral_quadrilateral"
            cell_type = "quadrilateral"
            variant = "lumped"
        elif dictionary["method"] in dg_t_equivalents:
            method = "DG_triangle"
            cell_type = "triangle"
            variant = "DG"
        elif dictionary["method"] in dg_q_equivalents:
            method = "DG_quadrilateral"
            cell_type = "quadrilateral"
            variant = "DG"
        elif dictionary["method"] == "DG":
            raise ValueError(
                "DG is not a valid method. Please specify \
                either DG_triangle or DG_quadrilateral."
            )
        elif dictionary["method"] == "CG":
            method, cell_type, variant = parse_cg(dictionary)
        else:
            raise ValueError(f"Method of {dictionary['method']} is not valid.")
        return method, cell_type, variant

    def get_from_cell_type_variant(self):
        triangle_equivalents = [
            "T",
            "triangle",
            "triangles",
            "tetrahedra",
            "tetrahedron",
        ]
        quadrilateral_equivalents = [
            "Q",
            "quadrilateral",
            "quadrilaterals",
            "hexahedra",
            "hexahedron",
        ]
        cell_type = self.options_dictionary["cell_type"]
        if cell_type in triangle_equivalents:
            cell_type = "triangle"
        elif cell_type in quadrilateral_equivalents:
            cell_type = "quadrilateral"
        else:
            raise ValueError(f"cell_type of {cell_type} is not valid.")

        variant = self.options_dictionary["variant"]
        accepted_variants = ["lumped", "equispaced", "DG"]
        if variant not in accepted_variants:
            raise ValueError(f"variant of {variant} is not valid.")

        if cell_type == "triangle" and variant == "lumped":
            method = "mass_lumped_triangle"
        elif cell_type == "triangle" and variant == "equispaced":
            method = "CG_triangle"
        elif cell_type == "triangle" and variant == "DG":
            method = "DG_triangle"
        elif cell_type == "quadrilateral" and variant == "lumped":
            method = "spectral_quadrilateral"
        elif cell_type == "quadrilateral" and variant == "equispaced":
            method = "CG_quadrilateral"
        elif cell_type == "quadrilateral" and variant == "DG":
            method = "DG_quadrilateral"
        else:
            raise ValueError(
                f"cell_type of {cell_type} with variant of {variant} is not valid."
            )
        return method, cell_type, variant


class read_mesh:
    """
    Read the mesh section of the dictionary.

    Attributes
    ----------
    mesh_dictionary : dict
        Dictionary containing the mesh information.
    dimension : int
        The spatial dimension of the problem.
    mesh_file : str
        The mesh file name.
    mesh_type : str
        The type of mesh.
    user_mesh : bool
        Whether the user has provided a mesh.
    firedrake_mesh : bool
        Whether the user requires a firedrake mesh.
    length_z : float
        The length in the z direction.
    length_x : float
        The length in the x direction.
    length_y : float
        The length in the y direction.

    Methods
    -------
    get_mesh_file_info()
        Get the mesh file name.
    get_mesh_type()
        Get the mesh type.
    _derive_mesh_type()
        Derive the mesh type.
    get_user_mesh()
        Get the user mesh.
    """
    def __init__(self, dimension=2, mesh_dictionary=None):
        default_dictionary = {
            # depth in km - always positive
            "Lz": 0.0,
            # width in km - always positive
            "Lx": 0.0,
            # thickness in km - always positive
            "Ly": 0.0,
            # mesh file name with .msh extension
            "mesh_file": None,
        }
        if mesh_dictionary is None:
            self.mesh_dictionary = default_dictionary
        else:
            self.mesh_dictionary = mesh_dictionary

        self.dimension = dimension
        self.mesh_file = self.get_mesh_file_info()

        check_if_mesh_file_exists(self.mesh_file)

        self.mesh_type = self.get_mesh_type()
        self.user_mesh = self.get_user_mesh()
        if self.mesh_type == "firedrake_mesh":
            self.firedrake_mesh = True
        else:
            self.firedrake_mesh = False

        if "Lz" in self.mesh_dictionary:
            self.length_z = self.mesh_dictionary["Lz"]
        else:
            self.length_z = default_dictionary["Lz"]
            warnings.warn("Lz not specified, using default of 0.0.")

        if "Lx" in self.mesh_dictionary:
            self.length_x = self.mesh_dictionary["Lx"]
        else:
            self.length_x = default_dictionary["Lx"]
            warnings.warn("Lx not specified, using default of 0.0.")

        if "Ly" in self.mesh_dictionary:
            self.length_y = self.mesh_dictionary["Ly"]
        elif dimension == 2:
            self.length_y = 0.0
        else:
            self.length_y = default_dictionary["Ly"]
            warnings.warn("Ly not specified, using default of 0.0.")

    def get_mesh_file_info(self):
        dictionary = self.mesh_dictionary
        if "mesh_file" not in dictionary:
            mesh_file = None
            return None

        mesh_file = dictionary["mesh_file"]

        if mesh_file is None:
            return None

        if mesh_file == "not_used.msh":
            mesh_file = None
        else:
            return mesh_file

    def get_mesh_type(self):
        valid_mesh_types = [
            "file",
            "firedrake_mesh",
            "user_mesh",
            "SeismicMesh",
            None,
        ]
        dictionary = self.mesh_dictionary
        if "mesh_type" not in dictionary:
            mesh_type = self._derive_mesh_type()
        elif dictionary["mesh_type"] in valid_mesh_types:
            mesh_type = dictionary["mesh_type"]
        else:
            raise ValueError(
                f"mesh_type of {dictionary['mesh_type']} is not valid."
            )

        if mesh_type is None:
            warnings.warn("No mesh yet provided.")

        return mesh_type

    def _derive_mesh_type(self):
        dictionary = self.mesh_dictionary
        user_mesh_in_dictionary = False
        if "user_mesh" not in dictionary:
            dictionary["user_mesh"] = None

        if self.mesh_file is not None:
            mesh_type = "file"
            return mesh_type
        elif dictionary["user_mesh"] is not None:
            mesh_type = "user_mesh"
            return mesh_type
        else:
            return None

    def get_user_mesh(self):
        if self.mesh_type == "user_mesh":
            return self.mesh_dictionary["user_mesh"]
        else:
            return False
