from spyro.utils.error_management import value_dimension_error, value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class RectangLayer():
    """
    Define a rectangular layer in 2D or parallelepided in 3D.

    Attributes
    ----------
    area : `float`
        Area of the domain with rectangular layer
    area_ratio : `float`
        Area ratio to the area of the original domain. area_ratio = area / a_orig
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
    domain_dim : `tuple`
        Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
    f_Ah : `float`
        Hyperelliptical area factor. f_Ah = 4 (n_hyp is considered infinite)
    f_Vh : `float`
        Hyperellipsoidal volume factor. f_Vh = 8 (n_hyp is considered infinite)
    hyper_axes : `tuple`
        Semi-axes of the rectangular layer (a, b) (2D) or (a, b, c) (3D)
    n_hyp: `float`
        Degree of the hyperelliptical pad layer. n_hyp is set to None because
        this attribute is not applicable to rectangular layers
    vol : `float`
        Volume of the domain with rectangular layer
    vol_ratio : `float`
        Volume ratio to the volume of the original domain. vol_ratio = vol / v_orig

    Methods
    -------
    calc_rec_geom_prop()
        Calculate the geometric properties for the rectangular layer
    define_hyperaxes()
        Define the rectangular semi-axes
    """

    def __init__(self, domain_dim, dimension=2):
        """Initialize the RectangLayer class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D

        Returns
        -------
        None
        """

        # Validate input arguments
        if not isinstance(domain_dim, tuple):
            raise TypeError("domain_dim must be a tuple, "
                            f"got {type(domain_dim).__name__}.")

        if dimension not in [2, 3]:
            value_parameter_error('dimension', dimension, [2, 3])

        # Original domain dimensions
        self.domain_dim = domain_dim

        # Model dimension
        self.dimension = dimension

        # Hypershape degree (Not applicable in rectangular layers)
        self.n_hyp = None

    def define_rec_hyperaxes(self, pad_len):
        """Define the rectangular semi-axes

        Parameters
        ----------
        pad_len : `float`
            Size of the absorbing layer

        Returns
        -------
        None
        """

        # Rectangular semi-axes
        Lx, Lz = self.domain_dim[:2]
        a_hyp = 0.5 * Lx + pad_len
        b_hyp = 0.5 * Lz + pad_len
        self.hyper_axes = (a_hyp, b_hyp)

        if self.dimension == 3:  # 3D
            Ly = self.domain_dim[2]
            c_hyp = 0.5 * Ly + pad_len
            self.hyper_axes += (c_hyp,)

    def calc_rec_geom_prop(self, domain_lay, pad_len):
        """Calculate the geometric properties for the rectangular layer.

        Parameters
        ----------
        domain_lay : `tuple`
            Domain dimensions with layer including truncation by free surface.
            - 2D : (Lx + 2 * pad_len, Lz + pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + pad_len, Ly + 2 * pad_len)
        pad_len : `float`
            Size of the absorbing layer

        Returns
        -------
        None
        """

        # Checking inputs
        chk_domd = len(self.domain_dim)
        chk_habc = len(domain_lay)
        if self.dimension != chk_domd or self.dimension != chk_habc:
            value_dimension_error(('domain_dim', 'domain_lay'),
                                  (chk_domd, chk_habc),
                                  self.dimension)

        # Domain dimensions w/o layer
        Lx, Lz = self.domain_dim[:2]
        Lx_habc, Lz_habc = domain_lay[:2]

        # Rectangular semi-axes
        self.define_rec_hyperaxes(pad_len)

        # Geometric properties of the rectangular layer
        if self.dimension == 2:  # 2D
            self.area = Lx_habc * Lz_habc
            self.area_ratio = self.area / (Lx * Lz)
            self.f_Ah = 4
            print("Area Ratio: {:5.3f}".format(self.area_ratio), flush=True)

        if self.dimension == 3:  # 3D
            Ly = self.domain_dim[2]
            Ly_habc = domain_lay[2]
            self.vol = Lx_habc * Lz_habc * Ly_habc
            self.vol_ratio = self.vol / (Lx * Lz * Ly)
            self.f_Vh = 8
            print("Volume Ratio: {:5.3f}".format(self.vol_ratio), flush=True)
