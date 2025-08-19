import numpy as np
from spyro.utils.error_management import value_dimension_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class RectangLayer():
    '''
    Define a rectangular layer in 2D or parallelepided in 3D.

    Attributes
    ----------
    area : `float`
        Area of the domain with hyperelliptical layer
    a_rat : `float`
        Area ratio to the area of the original domain. a_rat = area / a_orig
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
    f_Ah : `float`
        Hyperelliptical area factor. f_Ah = area / (a_hyp * b_hyp)
    f_Vh : `float`
        Hyperellipsoidal volume factor. f_Vh = vol / (a_hyp * b_hyp * c_hyp)
    n_hyp: `float`
        Degree of the hyperelliptical pad layer. n_hyp is set to None because
        this attribute is not applicable to rectangular layers
    vol : `float`
        Volume of the domain with hyperellipsoidal layer
    v_rat : `float`
        Volume ratio to the volume of the original domain. v_rat = vol / v_orig

    Methods
    -------
    calc_rec_geom_prop()
        Calculate the geometric properties for the rectangular layer

    '''

    def __init__(self, dimension=2):
        '''
        Initialize the HyperLayer class.

        Parameters
        ----------
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D

        Returns
        -------
        None
        '''

        # Model dimension
        self.dimension = dimension

        # Hypershape degree (Not applicable in rectangular layers)
        self.n_hyp = None

    def calc_rec_geom_prop(self, domain_dim, domain_lay):
        '''
        Calculate the geometric properties for the rectangular layer.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        domain_lay : `tuple`
            Domain dimensions with layer including truncation by free surface.
            - 2D : (Lx + 2 * pad_len, Lz + pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + pad_len, Ly + 2 * pad_len)

        Returns
        -------
        None
        '''

        # Domain dimensions w/o layer
        chk_domd = len(domain_dim)
        chk_habc = len(domain_lay)
        if self.dimension != chk_domd or self.dimension != chk_habc:
            value_dimension_error(('domain_dim', 'domain_lay'),
                                  (chk_domd, chk_habc), self.dimension)

        Lx, Lz = domain_dim[:2]
        Lx_habc, Lz_habc = domain_lay[:2]

        # Geometric properties of the rectangular layer
        if self.dimension == 2:  # 2D
            self.area = Lx_habc * Lz_habc
            self.a_rat = self.area / (Lx * Lz)
            self.f_Ah = 4
            print("Area Ratio: {:5.3f}".format(self.a_rat))

        if self.dimension == 3:  # 3D
            Ly = domain_dim[2]
            self.vol = Lx_habc * Lz_habc * Ly_habc
            self.v_rat = self.vol / (Lx * Lz * Ly)
            self.f_Vh = 8
            print("Volume Ratio: {:5.3f}".format(self.v_rat))
