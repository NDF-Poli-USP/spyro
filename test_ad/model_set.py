import spyro
import firedrake as fire
import SeismicMesh
import meshio


def model_settings(vel_model):
    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadrature": "KMV",  # Equi or KMV
        "degree": 1,  # p order
        "dimension": 2,  # dimension
        "regularization": False,  # regularization is on?
        "gamma": 1e-5,  # regularization parameter
    }

    model["parallelism"] = {
        "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the ABL.
    if vel_model == "horizont_layers":
        model["mesh"] = {
            "Lz": 1.0,  # depth in km - always positive
            "Lx": 1.0,  # width in km - always positive
            "Ly": 0.0,  # thickness in km - always positive
            "meshfile": "not_used.msh",
            "initmodel": "not_used.hdf5",
            "truemodel": "not_used.hdf5",
        }
    if vel_model == "marmousi":
        model["mesh"] = {
            "Lz": 3.5,  # depth in km - always positive
            "Lx": 10.,  # width in km - always positive
            "Ly": 0.0,  # thickness in km - always positive
            "meshfile": "meshes/mm.msh",
            "initmodel": "velocity_models/mm_guess.hdf5",
            "truemodel": "velocity_models/mm.hdf5",
        }
    if vel_model == "br_model":
        model["mesh"] = {
            "Lz": 7.5,  # depth in km - always positive
            "Lx": 17.312,  # width in km - always positive
            "Ly": 0.0,  # thickness in km - always positive
            "meshfile": "meshes/gm.msh",
                "initmodel": initmodel + ".hdf5",
            "truemodel": "velocity_models/gm_2020.hdf5",
        }
          
    # Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
    model["BCs"] = {
        "status": False,  # True or False, used to turn on any type of BC
        "method": "Damping", # either PML or Damping, used to turn on any type of BC
        "outer_bc": "none", #  none or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 2,  # damping layer has a exponent variation
        "cmax": 1.5,  # maximum acoustic wave velocity in PML - km/s
        "R": 1e-6,  # theoretical reflection coefficient
        "lz": 1.0,  # thickness of the PML in the z-direction (km) - always positive
        "lx": 1.0,  # thickness of the PML in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
    }
    if vel_model == "horizont_layers":
        model["acquisition"] = {
            "source_type": "Ricker",
            "num_sources": 1,
            "source_pos": [(1.0, 0.5)],
            "frequency": 10.0,
            "delay": 1.0,
            "num_receivers": 10,
            "receiver_locations": spyro.create_transect(
                (0.5, 0.2), (0.5, 0.8), 10
            ),
        }
    if vel_model == "marmousi" or vel_model == "br_model":
        model["acquisition"] = {
            "source_type": "Ricker",
            "frequency": 7.0,
            "delay": 1.0,
            # "num_sources": 1,
            "num_sources": 1,
            "source_pos": [(-0.125, 5.0)],
            "amplitude": 1.0,
            "num_receivers": 400,
            "receiver_locations": spyro.create_transect((-0.225, 0.2), (-0.225, 9.8), 400),
            }
    
    model["aut_dif"] = {
        "status": True, 
    }

    model["timeaxis"] = {
        "t0": 0.0,  # Initial time for event
        "tf": 0.5,  # Final time for event (for test 7)
        "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
        "fspool": 1,  # how frequently to save solution to RAM
    }
    return model


def meshing(model):
    
    mesher = "SM"  # either SM (SeismicMesh), FM (Firedrake mesher), or RM (read an existing mesh)

    if mesher == "FM":
        mesh = fire.RectangleMesh(100, 100, 1.5, 1.5) # to test FWI, mesh aligned with interface

    elif mesher == "SM":

        bbox = (0.0, 
                (model["mesh"]["Lz"] + model["BCs"]["lz"]), 
                0.0 - model["BCs"]["lx"], 
                model["mesh"]["Lz"] + model["BCs"]["lx"])

    rect = SeismicMesh.Rectangle(bbox)
    points, cells = SeismicMesh.generate_mesh(domain=rect, edge_length=0.03)
    mshname = "meshes/test_mu=0.msh"
    meshio.write_points_cells(
        mshname,
        points[:],  # do not swap here
        [("triangle", cells)],
        file_format="gmsh22",
        binary=False
    )
    mesh = fire.Mesh(mshname)

    return mesh


def _make_vp_pml(V, mesh, v0=1.5, v1=3.5):
    """Create a half space"""
    z, x = fire.SpatialCoordinate(mesh)
    velocity = fire.conditional(z < 0.5, v0, v1)
    vp = fire.Function(V, name="vp").interpolate(velocity)
    
    return vp