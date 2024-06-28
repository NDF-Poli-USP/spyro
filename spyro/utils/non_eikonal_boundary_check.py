import numpy as np
import firedrake as fire
import spyro


class Boundary_checker:
    """
    This class is used to check the boundaries of a wave object. It finds the critical boundary point and time.

    Attributes:
        is_checked (bool): A flag to check if the boundary is checked or not.
        tol (float): The tolerance value for the boundary check.
        mesh_z (object): The z-coordinate firedrake object of the mesh.
        mesh_x (object): The x-coordinate firedrake object of the mesh.
        function_space (float): The function space of the wave object.
        length_z (float): The length of the mesh in the z-direction.
        length_x (float): The length of the mesh in the x-direction.
    """
    def __init__(self, is_checked=True, Wave_obj=None, u_n=None):
        """
        Initializes the Boundary_checker class with the given parameters.

        Parameters:
            is_checked (bool): A flag to check if the boundary is checked or not. Default is True.
            Wave_obj (object): The wave object to check the boundary for.
            u_n (float): The current wave speed. Default is None.
        """
        if is_checked is False:
            self.is_checked = is_checked
            pass
        else:
            self.tol = 1e-6
            # calculate tolerance
            # self.calculate_tolerance

            # Finding boundary locations
            self.mesh_z = Wave_obj.mesh_z
            self.mesh_x = Wave_obj.mesh_x
            self.function_space = Wave_obj.function_space
            self.length_z = Wave_obj.length_z
            self.length_x = Wave_obj.length_x
            self.current_pressure = u_n
            self.interpolating_mesh_data()
            self.finding_boundary()
            self.initializing_variables()

    def interpolating_mesh_data(self):
        self.function_z = fire.Function(self.function_space)
        self.function_z.interpolate(self.mesh_z)
        self.function_x = fire.Function(self.function_space)
        self.function_x.interpolate(self.mesh_x)

    def finding_boundary(self):
        tol = self.tol
        function_z = self.function_z
        function_x = self.function_x
        length_z = self.length_z
        length_x = self.length_x

        left_boundary = np.where(function_x.dat.data[:] <= tol)
        right_boundary = np.where(function_x.dat.data[:] >= length_x-tol)
        bottom_boundary = np.where(function_z.dat.data[:] <= tol-length_z)
        self.boundaries = [left_boundary, right_boundary, bottom_boundary]
        self.boundary_names = ["left", "right", "bottom"]

    def initializing_variables(self):
        self.check_left = True
        self.check_right = True
        self.check_bottom = True

        self.t_left = np.inf
        self.t_right = np.inf
        self.t_bottom = np.inf

        self.bottom_point_dof = np.nan
        self.left_point_dof = np.nan
        self.right_point_dof = np.nan

        self.check_individual_boundary = [self.check_left, self.check_right, self.check_bottom]
        self.time_found_in_boundaries = [self.t_left, self.t_right, self.t_bottom]
        self.dof_in_boundaries = [self.bottom_point_dof, self.left_point_dof, self.right_point_dof]

    def check(self, pressure_across_domain, time):
        if self.is_checked is False:
            pass

        self.pressure = pressure_across_domain
        for i in len(self.boundaries):
            if self.check_individual_boundary[i]:
                self.check_boudary(i, time)

    def check_individual_boundary(self, boundary_index, time):
        u_n = self.pressure
        threshold = self.tol
        local_boundary = self.boundaries[boundary_index]
        pressure_on_boundary = u_n.dat.data_ro_with_halos[local_boundary]

        if np.any(np.abs(pressure_on_boundary) > threshold):
            print("Pressure on left boundary is not zero")
            print(f"Time hit left boundary = {time}")
            self.time_found_in_boundaries[boundary_index] = time
            self.check_individual_boundary[boundary_index] = False
            vector_indices = np.where(np.abs(pressure_on_boundary) > threshold)
            vector_indices = vector_indices[0]
            global_indices = local_boundary[0][vector_indices]
            z_values = function_z.dat.data[global_indices]
            z_avg = np.average(z_values)
            indice = global_indices[np.argmin(np.abs(z_values-z_avg))]
            left_point_dof = indice
