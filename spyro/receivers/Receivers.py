from firedrake import *  # noqa: F403
from spyro.receivers.dirac_delta_projector import Delta_projector

import numpy as np


class Receivers(Delta_projector):
    """Project data defined on a triangular mesh to a
    set of 2D/3D coordinates for variable spatial order
    using Lagrange interpolation.

    Can interpolate receiveir values that do not coincide with
    mesh or DOF points

    ...

    Attributes
    ----------
    mesh : Firedrake.mesh
        mesh where receivers are located
    V: Firedrake.FunctionSpace object
        The space of the finite elements
    my_ensemble: Firedrake.ensemble_communicator
        An ensemble communicator
    dimension: int
        The dimension of the space
    degree: int
        Degree of the function space
    receiver_locations: list
        List of tuples containing all receiver locations
    num_receivers: int
        Number of receivers
    quadrilateral: boolean
        Boolean that specifies if cells are quadrilateral
    is_local: list of booleans
        List that checks if receivers are present in cores
        spatial paralelism

    Methods
    -------
    build_maps()
        Calculates and stores tabulations for interpolation
    interpolate(field)
        Interpolates field value at receiver locations
    apply_receivers_as_source(rhs_forcing, residual, IT)
        Applies receivers as source with values from residual
        in timestep IT, for usage with adjoint propagation
    """

    def __init__(self, wave_object):
        """Initializes class and gets all receiver parameters from
        input file.
        Parameters
        ----------
        model: `dictionary`
            Contains simulation parameters and options.
        mesh: a Firedrake.mesh
            2D/3D simplicial mesh read in by Firedrake.Mesh
        V: Firedrake.FunctionSpace object
            The space of the finite elements
        my_ensemble: Firedrake.ensemble_communicator
            An ensemble communicator
        Returns
        -------
        Receivers: :class: 'Receiver' object
        """
        super().__init__(wave_object)
        self.point_locations = wave_object.receiver_locations

        if self.dimension == 3 and wave_object.automatic_adjoint:
            # self.column_x = model["acquisition"]["num_rec_x_columns"]
            # self.column_y = model["acquisition"]["num_rec_y_columns"]
            # self.column_z = model["acquisition"]["num_rec_z_columns"]
            # self.number_of_points = self.column_x*self.column_y
            raise ValueError("Implement this later")
        else:
            self.number_of_points = wave_object.number_of_receivers

        self.is_local = [0] * self.number_of_points
        if not self.automatic_adjoint:
            self.build_maps()

    def apply_receivers_as_source(self, rhs_forcing, residual, IT):
        """The adjoint operation of interpolation (injection)

        Injects residual, and timestep IT, at receiver locations
        as source and stores their value in the right hand side
        operator rhs_forcing.

        Parameters
        ----------
        rhs_forcing: object
            Firedrake assembled right hand side operator to store values
        residual: list
            List of residual values at different receiver locations
            and timesteps
        IT: int
            Desired time step number to get residual value from

        Returns
        -------
        rhs_forcing: object
            Firedrake assembled right hand side operator with injected values
        """
        for rid in range(self.number_of_points):
            value = residual[IT][rid]
            if self.is_local[rid]:
                idx = np.int_(self.cellNodeMaps[rid])
                phis = self.cell_tabulations[rid]

                tmp = np.dot(phis, value)
                rhs_forcing.dat.data_with_halos[idx] += tmp
            else:
                tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing

    def set_point_cloud(self, comm):
        # Receivers always parallel to z-axis

        rec_pos = self.point_locations

        # 2D --
        if self.dimension == 2:
            num_rec = self.number_of_points
            δz = np.linspace(rec_pos[0, 0], rec_pos[num_rec - 1, 0], 1)
            δx = np.linspace(rec_pos[0, 1], rec_pos[num_rec - 1, 1], num_rec)

            Z, X = np.meshgrid(δz, δx)
            xs = np.vstack((Z.flatten(), X.flatten())).T

        # 3D
        elif self.dimension == 3:
            δz = np.linspace(rec_pos[0][0], rec_pos[1][0], self.column_z)
            δx = np.linspace(rec_pos[0][1], rec_pos[1][1], self.column_x)
            δy = np.linspace(rec_pos[0][2], rec_pos[1][2], self.column_y)

            Z, X, Y = np.meshgrid(δz, δx, δy)
            xs = np.vstack((Z.flatten(), X.flatten(), Y.flatten())).T
        else:
            print("This dimension is not accepted.")
            quit()

        point_cloud = VertexOnlyMesh(self.mesh, xs)  # noqa: F405

        return point_cloud
