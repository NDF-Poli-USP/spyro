import firedrake as fire
# from fenics import SubMesh, File, BoundaryMesh
# from fenics import Function, pi, CompiledSubDomain
# import numpy as np
import ipdb

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class NRBCHabc():
    '''
    class for NRBCs applied to outer boundary absorbing layer in HABC scheme.

    Attributes
    ----------


    Methods
    -------
    cos_ang_HigdonBC()
    '''

    def cos_ang_HigdonBC(self):
        '''
        Field for Higdon BC where cosine of the incidence angle is mapped
        '''

        # Initialize Higdon BC first order
        print("\nCreating Field for Higdon BC")
        self.cosHig = fire.Function(self.function_space, name='cosHig')

        # Boundary nodes
        bnd_nod = fire.DirichletBC(self.function_space, 0, "on_boundary").nodes

        # Node coordinates
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        bnd_z = z_f.dat.data_with_halos[bnd_nod]
        bnd_x = x_f.dat.data_with_halos[bnd_nod]

        if self.dimension == 3:  # 3D
            y_f = fire.Function(
                self.function_space).interpolate(bnd_coord[:, 2])
            bnd_y = y_f.dat.data_with_halos[bnd_nod]

        # Identify source locations
        possou = self.eik_bnd[0][-1]
        psoux = possou[0]
        psouy = possou[1]

        # Compute cosine of the incidence angle with dot product
        
        ipdb.set_trace()


        # theta = np.arctan2(Mes.mesh_coord[dofCosBC, 1] - psouy, Mes.mesh_coord[dofCosBC, 0] - psoux)

        # if pH['TipLay'] == 'REC':  # Rectangular layer
        #     thetaR1 = np.arctan2(pmly + Ly - psouy, pmlx + Lx - psoux)
        #     thetaR2 = np.arctan2(pmly + Ly - psouy, pmlx - psoux)
        #     thetaR3 = np.arctan2(pmly - psouy, pmlx + Lx - psoux)
        #     thetaR4 = np.arctan2(pmly - psouy, pmlx - psoux)
        # else:  # Elliptical layer
        #     if pH['CamComp']:
        #         thetaR1 = pi/4
        #         thetaR2 = 3*pi/4
        #     else:
        #         thetaR1 = np.arctan2(pmly + Ly - psouy, pmlx + Lx - psoux)
        #         thetaR2 = np.arctan2(pmly + Ly - psouy, pmlx - psoux)

        #     thetaR3 = -pi/4
        #     thetaR4 = -3*pi/4
        # # print(thetaR1*180/pi, thetaR2*180/pi, thetaR3*180/pi, thetaR4*180/pi)
        # condA = (theta > thetaR1) & (theta < thetaR2)
        # condB = (theta < thetaR3) & (theta > thetaR4)
        # condC = (theta >= thetaR2)
        # condD = (theta <= thetaR4)
        # theta[condA] = -pi/2 + theta[condA]
        # theta[condB] = pi/2 + theta[condB]
        # theta[condC] = -pi + theta[condC]
        # theta[condD] = pi + theta[condD]
        # # Field with cosine of the incidence angle on the boundary
        # cosHig.vector()[dofCosBC] = np.cos(theta)
        # cosHig.vector().apply('insert')
        # del theta
        # if pH['saveFile']:
        #     cosHig_file = File(pH['FolderCase'] + '/out/cosHig.pvd')
        #     cosHig_file << cosHig

        # return cosHig
