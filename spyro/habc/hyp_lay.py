import firedrake as fire
import numpy as np
import ipdb
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HyperLayer():
    '''
    Define a hyperlliptical layer in 2D or hyperellipsoidal in 3D.
    '''

    def __init__(self, n_hyp=2, dimension=2):
        '''
        Initializes the HyperLayer class.
        '''

        # Model dimension
        self.dimension = dimension

        # Hypershape degree
        self.n_hyp = n_hyp

    def define_hyperaxes(self, domain_dim):

        # Domain dimensions = [Lx, Lz, Ly]
        verif_domd = len(domain_dim)
        if self.dimension == verif_domd:
            self.domain_dim = domain_dim

        else:
            aux0 = "Number of domain dimensions"
            aux1 = "different from the model dimension"
            aux = aux0 + " ({:0}) " + aux1 + " ({:1}D)."
            UserWarning(aux.format(verif_domd, self.dimension))

        # Hypershape semi-axes = [a_hyp, b_hyp, c_hyp]
        a_hyp = 0.5 * self.length_x + self.pad_len
        b_hyp = 0.5 * self.length_z + self.pad_len
        self.hyper_axes = [a_hyp, b_hyp]
        if self.dimension == 3:  # 3D
            c_hyp = 0.5 * self.length_y + self.pad_len
            self.hyper_axes.append(c_hyp)

    def calc_degree_nhyp(self, x_rel, lim, n_min=2, n_max=20, monitor=False):
        '''
        Define the limits for the hypershape degree. See Salas et al (2022).
        r < 1 ensures that the point is inside the layer
        '''

        # Hypershape semi-axes
        a = self.hyper_axes[0]
        b = self.hyper_axes[1]

        # Superness s= 0.5^(-1/n): Extreme points of Hyperellipse
        xs = x_rel[0]
        ys = x_rel[1]

        # Harmonic mean
        h = max(int(np.ceil(np.log(0.5) / np.log(
                (1 / a + 1 / b) / (1 / xs + 1 / ys)))), 2)
        rh = abs(xs / a)**h + abs(ys / b)**h

        # Geometric mean
        g = max(int(np.ceil(np.log(0.25) / np.log(xs * ys / (a * b)))), 2)
        rg = abs(xs / a)**g + abs(ys / b)**g

        # Arithmetic mean
        z = max(int(np.ceil(np.log(0.5) / np.log((xs + ys) / (a + b)))), 2)
        rz = abs(xs / a)**z + abs(ys / b)**z

        r = n = 1
        while r >= 1 and n < n_max:
            n += 1
            r = abs(xs / a)**n + abs(ys / b)**n
            if monitor:
                print('ParHypEll - r: {:>5.3f} - n: {:>}'.format(r, n))

        if monitor:
            print("'Harm' Superness. r: {:>5.3f} - n: {:>}".format(rh, h))
            print("'Geom' Superness. r: {:>5.3f} - n: {:>}".format(rg, g))
            print("'Arit' Superness. r: {:>5.3f} - n: {:>}".format(rz, z))

        if lim == 'MIN':
            n = np.clip(max(n, h, g, z), n_min, n_max)
        elif lim == 'MAX':
            n = np.clip(min(n, h, g, z), n_min + 1, n_max)

        lim_str = "min" if lim == 'MIN' else "max"
        pr0_str = "Hypershape Parameters. r_" + lim_str
        pr1_str = ": {:>5.3f} - n_" + lim_str + ": {:>}"
        print(pr0_str + pr1_str.format(r, n))
        print("Superness Coordinates (km): ({:5.3f}, {:5.3f})".format(xs, ys))

        return n

    def define_hyperlayer(self, pad_len, lmin):
        '''
        Define the hyperlayer limits.
        n_max: Exponent for ensuring pad_len in the domain diagonal
        '''

        # Hypershape semi-axes and domain dimensions
        a_hyp = self.hyper_axes[0]
        b_hyp = self.hyper_axes[1]
        Lx = self.domain_dim[0]
        Lz = self.domain_dim[1]

        if self.dimension == 3:  # 3D
            self.c_hyp = self.hyper_axes[2]
            Ly = self.domain_dim[2]

        # Hyperellipse degree
        n_hyp = self.n_hyp

        # Minimum allowed exponent
        x_min = [0.5 * Lx + lmin, 0.5 * Lz + lmin]
        r_min = abs(x_min[0] / a_hyp)**n_hyp + abs(x_min[1] / b_hyp)**n_hyp

        # Maximum allowed exponent
        theta = np.arctan2(Lz, Lx)
        x_max = [0.5 * Lx + pad_len * np.cos(theta),
                 0.5 * Lz + pad_len * np.sin(theta)]
        r_max = abs(x_max[0] / a_hyp)**n_hyp + abs(x_max[1] / b_hyp)**n_hyp

        # Verification of hyperellipse exponent
        print("Checking Degree Hypershape n_hyp: {}".format(n_hyp))
        print("Semi-axes (km): a_hyp:{:5.3f} - b_hyp:{:5.3f}".format(a_hyp, b_hyp))

        print("Minimum Degree for Hypershape Layer")
        n_min = self.calc_degree_nhyp(x_min, 'MIN')

        print("Maximum Degree for Hypershape Layer")
        n_max = self.calc_degree_nhyp(x_max, 'MAX', n_min=n_min)

        ipdb.set_trace()

        # condA = n_hyp < p
        # condB = n_hyp >= p and n_hyp > pf

        # if condA or condB:
        #     print('Current Exponent: {}'.format(n_hyp))
        #     print('Minimum Exponent for Hyperellipse: {}'.format(p))
        #     print('Maximum Exponent for Hyperellipse: {}'.format(pf))
        #     if condA:
        #         sys.exit('Low Exponent for Hyperellipse')
        #     elif condB:
        #         sys.exit('High Exponent for Hyperellipse')
        # else:
        #     print('Minimum Exponent for Hyperellipse: {}'.format(p))
        #     print('Current Exponent: {}'.format(n_hyp))
        #     print('Maximum Exponent for Hyperellipse: {}'.format(pf))

        # print('************************************')
        # if pH['verHypExp']:
        #     sys.exit('Verification of Exponent Limits for Hyperellipse')
