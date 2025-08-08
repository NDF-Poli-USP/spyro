import firedrake as fire

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class Dir_point_bc(fire.DirichletBC):
    '''
    Class for Eikonal boundary conditions at a point.

    Attributes
    ----------
    nodes : `array`
        Points where the boundary condition is to be applied
    '''

    def __init__(self, V, value, nodes):
        '''
        Initialize the Dir_point_bc class.

        Parameters
        ----------
        V : `firedrake function space`
            Function space where the boundary condition is applied
        value : `firedrake constant`
            Value of the boundary condition
        nodes : `array`
            Points where the boundary condition is to be applied

        Returns
        -------
        None
        '''

        # Calling superclass init and providing a dummy subdomain id
        super(Dir_point_bc, self).__init__(V, value, 0)

        # Overriding the "nodes" property
        self.nodes = nodes
