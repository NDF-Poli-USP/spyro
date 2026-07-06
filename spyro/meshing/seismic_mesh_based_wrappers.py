import meshio
from .meshing_utils import check_seismicmesh
from firedrake import Mesh as FireMeshReader
from firedrake import DistributedMeshOverlapType


try:
    import SeismicMesh
except ImportError:
    SeismicMesh = None


@check_seismicmesh
def create_seismicmesh_2D_mesh_with_velocity_model(mesh_parameters):
    """
    Creates a 2D mesh with velocity-based refinement using SeismicMesh.

    Returns
    -------
    mesh : Firedrake Mesh
        The generated 2D mesh with velocity-based sizing.

    Notes
    -----
    This method uses the velocity model to determine the mesh element sizes,
    ensuring appropriate resolution for wave propagation. The sizing function
    is derived from the velocity model SEGY file. Only the ensemble rank 0
    performs the mesh generation, and the mesh is then distributed across
    all processes.
    """
    if mesh_parameters.comm.ensemble_comm.rank == 0:
        v_min = mesh_parameters.minimum_velocity
        frequency = mesh_parameters.source_frequency

        C = mesh_parameters.cpw

        length_z = mesh_parameters.length_z
        length_x = mesh_parameters.length_x
        domain_pad = mesh_parameters.abc_pad
        lbda_min = v_min / frequency

        bbox = (-length_z, 0.0, 0.0, length_x)
        domain = SeismicMesh.Rectangle(bbox)

        hmin = lbda_min / C
        mesh_parameters.comm.comm.barrier()

        ef = SeismicMesh.get_sizing_function_from_segy(
            mesh_parameters.velocity_model,
            bbox,
            hmin=hmin,
            wl=C,
            freq=frequency,
            grade=0.15,
            domain_pad=domain_pad,
            pad_style="edge",
            units='km/s',
            comm=mesh_parameters.comm.comm,
        )
        mesh_parameters.comm.comm.barrier()

        # Creating rectangular mesh
        points, cells = SeismicMesh.generate_mesh(
            domain=domain,
            edge_length=ef,
            verbose=0,
            mesh_improvement=False,
            comm=mesh_parameters.comm.comm,
        )
        mesh_parameters.comm.comm.barrier()

        print('entering spatial rank 0 after mesh generation')
        if mesh_parameters.comm.comm.rank == 0:
            meshio.write_points_cells(
                "automatic_mesh.msh",
                points,
                [("triangle", cells)],
                file_format="gmsh22",
                binary=False
            )

            meshio.write_points_cells(
                "automatic_mesh.vtk",
                points,
                [("triangle", cells)],
                file_format="vtk"
            )

    mesh_parameters.comm.comm.barrier()
    mesh = FireMeshReader(
        'automatic_mesh.msh',
        distribution_parameters={
            "overlap_type": (DistributedMeshOverlapType.NONE, 0)
        },
        comm=mesh_parameters.comm.comm,
    )

    return mesh


def create_seismicmesh_2D_mesh_homogeneous(mesh_parameters):
    """
    Creates a 2D mesh based on SeismicMesh meshing utilities, with homogeneous velocity model.

    Returns
    -------
    mesh : `Firedrake.Mesh`
        The generated 2D mesh with uniform element sizes.

    Notes
    -----
    This method creates a rectangular mesh with uniform element sizing.
    The edge length is either user-specified or calculated based on the
    minimum velocity, source frequency, and cells per wavelength.
    Boundary entities with low quality are removed to improve mesh quality.
    """
    length_z = mesh_parameters.length_z
    length_x = mesh_parameters.length_x
    pad = mesh_parameters.abc_pad

    if pad is not None:
        real_lz = length_z + pad
        real_lx = length_x + 2 * pad
    else:
        real_lz = length_z
        real_lx = length_x
        pad = 0.0

    edge_length = mesh_parameters.edge_length
    if edge_length is None:
        edge_length = mesh_parameters.minimum_velocity / (mesh_parameters.source_frequency * mesh_parameters.cpw)

    bbox = (-real_lz, 0.0, -pad, real_lx - pad)
    rectangle = SeismicMesh.Rectangle(bbox)

    points, cells = SeismicMesh.generate_mesh(
        domain=rectangle,
        edge_length=edge_length,
        verbose=0,
    )

    points, cells = SeismicMesh.geometry.delete_boundary_entities(
        points, cells, min_qual=0.6
    )

    meshio.write_points_cells(
        mesh_parameters.output_file_name,
        points,
        [("triangle", cells)],
        file_format="gmsh22",
        binary=False,
    )
    meshio.write_points_cells(
        mesh_parameters.output_file_name + ".vtk",
        points,
        [("triangle", cells)],
        file_format="vtk",
    )

    return FireMeshReader(mesh_parameters.output_file_name)
