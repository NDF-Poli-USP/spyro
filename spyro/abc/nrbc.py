"""Non-reflecting boundary condition helpers for ABCs."""

from firedrake import Function, VTKFile
from numpy import abs, asarray, cos, maximum, pi, sign, sqrt, sum
from numpy.linalg import norm
from .abc import AbsorbingBC
from ..io.basicio import parallel_print as pprint
from ..tools.abc_set_path_cases import path_to_save_abc_case
from ..utils.error_management import validate_enum, validate_numeric, validate_parameter
from ..utils.typing import AbsorbingBCsType, BoundaryConditionsType, NRBCBoundaryType

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender
# TODO: Add citation


class NRBC(AbsorbingBC):
    """Class for Non-Reflective BCs applied to the outer boundary of an absorbing layer.

    Attributes
    ----------
    abc_boundary_type : `typing.NRBCBoundaryType`, optional
        Boundary typr where NRBCs are applied . Options: `NRBCBoundaryType.STRAIGHT`
        or `NRBCBoundaryType.HYPERSHAPE`. Default is `NRBCBoundaryType.STRAIGHT`.
    angle_max : `float`
        Maximum incidence angle considered. Default is `numpy.pi/4`.
    case_nrbc : `str`
        Label for the output files that includes the NRBC type ("HIG" or "SOM")
        and the boundary type where the NRBCs are applied ("STB" or "HYP").
        Examples: "HIG_STB", "HIG_HYP", "SOM_STB" or "SOM_HYB".
    cos_Hig : `firedrake function`
        Profile of the cosine of incidence angle for 1^st-order Higdon BC.
        Free surfaces and interior nodes are set to 0.
    cos_min : `float`
        Minimum value of the cosine of the incidence angle.
    non_reflect_bc : `typing.BoundaryConditionsType`, optional
            Type of boundary condition to apply on the domain boundaries (for only NRBCs)
            or the outer absorbing layer boundaries (HABCs: Absorbing Layer aand NRBCs).
            Options:'BoundaryConditionsType.HIGDON' or 'BoundaryConditionsType.SOMMERFELD'.
            Dafault is 'BoundaryConditionsType.HIGDON'.
    path_case_nrbc : `string`
        Path to save data for the current case study of NRBCs.
    path_save : `string`
        Path to save data.

    Methods
    -------
    cos_ang_HigdonBC()
        Compute the cosine of the incidence angle for first-order Higdon BC.
    hypershape_normal_vector()
        Compute the normal vector to a hypershape at a boundary point.
    source_to_bnd_reference_vector()
        Compute a unitary direction vector from the source to a boundary point.
    """

    def __init__(self, domain_dim, non_reflect_bc=BoundaryConditionsType.HIGDON,
                 angle_max=pi/4., abc_boundary_type=NRBCBoundaryType.STRAIGHT,
                 dimension=2, nrbc_in_habc=False, output_folder=None, comm=None):
        """Initialize the NRBC class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D.
        non_reflect_bc : `typing.BoundaryConditionsType`, optional
            Type of boundary condition to apply on the domain boundaries (for only NRBCs)
            or the outer absorbing layer boundaries (HABCs: Absorbing Layer aand NRBCs).
            Options: `BoundaryConditionsType.HIGDON` or `BoundaryConditionsType.SOMMERFELD`.
            Dafault is `BoundaryConditionsType.HIGDON`.
        angle_max : `float`, optional
            Maximum incidence angle considered in the NRBC. Default is `numpy.pi/4` (45°).
        abc_boundary_type : `typing.NRBCBoundaryType`, optional
            Boundary type where NRBCs are applied . Options: `NRBCBoundaryType.STRAIGHT`
            or `NRBCBoundaryType.HYPERSHAPE`. Default is `NRBCBoundaryType.STRAIGHT`.
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        nrbc_in_habc : `bool`, optional
            If `True`, the NRBCs are applied on the outer absorbing layer boundaries
            (HABCs: Absorbing Layer and NRBCs). If `False`, the NRBCs are applied on
            the original domain boundaries. Default is `False`.
        output_folder : `str`, optional
            The folder where output data will be saved. Default is `None`.
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Initializing the AbsorbingBC class if NRBCs are not in HABC scheme
        if not nrbc_in_habc:
            AbsorbingBC.__init__(self, domain_dim, dimension=dimension, comm=comm)

            # Non-reflective BC type
        self.non_reflect_bc = validate_parameter('non_reflect_bc', non_reflect_bc,
                                                 [BoundaryConditionsType.HIGDON,
                                                  BoundaryConditionsType.SOMMERFELD])

        # Boundary type where NRBCs are applied
        self.abc_boundary_type = validate_enum("abc_boundary_type",
                                               abc_boundary_type,
                                               NRBCBoundaryType)

        # Maximum incidence angle considered
        self.angle_max = validate_numeric("angle_max", angle_max, float_num=True,
                                          integer_num=False, lower_bound=0.,
                                          include_lower_bound=True)

        # Maximum value of the cosine of the incidence angle
        self.cos_min = cos(angle_max)

        """"
        Create the path to save data
        The required abc_type argument from path_to_save_abc_layer_case() method is set
        to `AbsorbingBCsType.NRBC` since it is an instance of `typing.AbsorbingBCsType`.
        """
        self.path_save, self.case_nrbc, self.path_case_nrbc = \
            path_to_save_abc_case(AbsorbingBCsType.NRBC,
                                  non_reflect_bc=self.non_reflect_bc,
                                  angle_max=self.angle_max,
                                  abc_boundary_type=self.abc_boundary_type,
                                  output_folder=output_folder)

        # Initializing the MeasureError class if NRBCs are not in HABC scheme
        if not nrbc_in_habc:
            self.initialize_paths_for_error(output_folder=self.path_save,
                                            output_case=self.path_case_nrbc)

    def source_to_bnd_reference_vector(self, source_coord, bnd_nodes_nfs):
        """Compute a unitary direction vector from the source to a boundary point.

        Parameters
        ----------
        source_coord : `tuple`
            Source coordinates.
        bnd_nodes_nfs : `tuple`
            Mesh node coordinates on non-free surfaces.
            - (z_data[nfs_idx], x_data[nfs_idx]) for 2D.
            - (z_data[nfs_idx], x_data[nfs_idx], y_data[nfs_idx]) for 3D.

        Returns
        -------
        unit_ref_vct : `array`
            Unit direction vector from the source to a boundary point.
        """

        # Boundary node data
        bnd_z, bnd_x = bnd_nodes_nfs[:2]

        # Source coordinates
        psouz = source_coord[0]
        psoux = source_coord[1]

        # Components of the vector pointing to the boundary point
        ref_x = bnd_x - psoux
        ref_z = bnd_z - psouz
        ref_vct = [ref_x, ref_z]

        if self.dimension == 3:  # 3D

            # Third component of the vector pointing to the boundary point
            bnd_y = bnd_nodes_nfs[2]
            psouy = source_coord[2]
            ref_y = bnd_y - psouy
            ref_vct.append(ref_y)

        # Unitary vector pointing to the boundary point
        unit_ref_vct = asarray(ref_vct) / norm(ref_vct, axis=0)

        return unit_ref_vct

    def hypershape_normal_vector(self, bnd_pnts, hyper_axes, n):
        """Compute the normal vector to a boundary point of a hypershape.

        Compute the normal vector to a hyperellipse (|x/a|^n + |y/b|^n = 1) or
        a hyperellipsoid (|x/a|^n + |y/b|^n + |z/c|^n = 1) at a boundary point.
        The hypershape must have the center at the origin.

        Parameters
        ----------
        bnd_pnts : `list`
            Boundary hypershape points where the normal vector is computed.
            Structure: [x, y] for 2D and [x, y, z] for 3D.
        hyper_axes : `list`
            Semi-axes of the hyperellipse [a, b] or hyperellipsoid [a, b, c].
        n : `float`
            Degree of the hyperellipse.

        Returns
        -------
        unit_nrm_vct : `array`
            Unitary normal vector to the hypershape at the boundary point.

        Notes
        -----
        Let f(x, y) = |x/a|^n - |y/b|^n -1 = 0 a level curve (level set for
        two variables) for f(x, y, z) at z = 0. The gradient of the function
        f given by ∇f(x,y) = [∂f/∂x, ∂f/∂y] is a normal vector to the curve.
        The normal vector is given by the partial derivatives of the function.
        """

        # Point coordinates
        x, y = bnd_pnts[:2]

        # Hypershape semi-axes
        a, b = hyper_axes[:2]

        # Compute partial derivatives
        df_dx = (n / (a**n)) * sign(x) * abs(x)**(n - 1)
        df_dy = (n / (b**n)) * sign(y) * abs(y)**(n - 1)

        nrm_vct = [df_dx, df_dy]

        if self.dimension == 3:  # 3D

            # Third coordinate
            z = bnd_pnts[2]

            # Third hypershape semi-axis
            c = hyper_axes[2]

            # Partial derivative with respect to third coordinate
            df_dz = (n / (c**n)) * sign(z) * abs(z)**(n - 1)

            nrm_vct.append(df_dz)

        # Unitary hypershape normal vector
        unit_nrm_vct = asarray(nrm_vct) / norm(nrm_vct, axis=0)

        return unit_nrm_vct

    def cos_ang_HigdonBC(self, V, source_coord, bnd_nod_ids_nfs,
                         bnd_nodes_nfs, hyp_par=None, save_file=True):
        """Compute the cosine of the incidence angle for first-order Higdon BC.

        Parameters
        ----------
        V : `firedrake function space`
            Function space where the Non-Reflective BCs are defined.
        source_coord : `tuple`
            Source coordinates.
        bnd_nod_ids_nfs : 'array'
            Mesh node indices on non-free surfaces.
        bnd_nodes_nfs : `tuple`
            Mesh node coordinates on non-free surfaces.
            - (z_data[nfs_idx], x_data[nfs_idx]) for 2D.
            - (z_data[nfs_idx], x_data[nfs_idx], y_data[nfs_idx]) for 3D.
        hyp_par : `tuple`, optional
            Hyperellipse parameters. Structure:
            (n_hyp, a_hyp, b_hyp) for 2D or (n_hyp, a_hyp, b_hyp, b_hyp) for 3D.
            - n_hyp : `float`
                Degree of the hyperellipse.
            - a_hyp : `float`
                Hyperellipse semi-axis in direction x.
            - b_hyp : `float`
                Hyperellipse semi-axis in direction z.
            - c_hyp : `float`
                Hyperellipse semi-axis in direction y (3D only).
        save_file : `bool`, optional
            If `True`, save the velocity model with absorbing layer in a .pvd file.
            Default is `True`.

        Returns
        -------
        None
        """

        pprint(f"Creating Field for NRBC: {self.non_reflect_bc.value}", comm=self.comm)

        # Initialize field for the cosine of the incidence angle
        self.cosHig = Function(V, name='cosHig')

        if self.non_reflect_bc == BoundaryConditionsType.SOMMERFELD:  # Sommerfeld BC
            cos_Hig = 1.

        else:  # Higdon BC

            # Unitary reference vector pointing to the boundary point
            unit_ref_vct = self.source_to_bnd_reference_vector(source_coord,
                                                               bnd_nodes_nfs)

            # Normal vector to the boundary
            if self.abc_boundary_type == NRBCBoundaryType.STRAIGHT:
                # Normal vector to the boundary is a orthonormal vector, then
                # cosine on incidence angle can be estimated from a projection
                # of the reference vector to boundary onto the orthonormal
                # vectors ([1, 0, 0] (2D), [0, 1, 0] (2D), [0, 0, 1] (3D))
                cos_Hig = maximum.reduce(abs(unit_ref_vct))

            if self.abc_boundary_type == NRBCBoundaryType.HYPERSHAPE:

                # Original domain dimensions
                length_z, length_x = self.domain_dim[:2]

                # Hypershape degree and semi-axes
                n_hyp, hyp_axes = hyp_par[0], hyp_par[1:]

                # Boundary points of the hypershape centered at the origin
                bnd_z, bnd_x = bnd_nodes_nfs[:2]  # Boundary node data
                bnd_pnts = [bnd_x - length_x / 2, bnd_z + length_z / 2]
                if self.dimension == 3:  # 3D
                    length_y = self.domain_dim[2]
                    bnd_y = bnd_nodes_nfs[2]
                    bnd_pnts.append(bnd_y - length_y / 2)

                # Normal vector to the boundary
                unit_nrm_vct = self.hypershape_normal_vector(bnd_pnts, hyp_axes, n_hyp)

                # Cosine of the incidence angle
                cos_Hig = sum(unit_ref_vct * unit_nrm_vct, axis=0)

            # Adjust values to minimum cosine of incidence angle
            cos_Hig[cos_Hig < self.cos_min] = sqrt(1. - cos_Hig[cos_Hig < self.cos_min]**2)

        self.cosHig.dat.data_with_halos[bnd_nod_ids_nfs] = cos_Hig

        # Save boundary profile of cosine of incidence angle
        if save_file:
            outfile = VTKFile(self.path_save_nrbc + "cosHig.pvd")
            outfile.write(self.cosHig)
