import firedrake as fire
import math
import pytest
import spyro

try:
    from SeismicMesh import write_velocity_model

    seismic_mesh = True
except ImportError:
    seismic_mesh = False


@pytest.mark.skipif(not seismic_mesh, reason="No SeismicMesh")
def test_read_and_write_segy():
    vp_name = "velocity_models/test"
    segy_file = vp_name + ".segy"
    hdf5_file = vp_name + ".hdf5"
    mesh = fire.UnitSquareMesh(10, 10)
    mesh.coordinates.dat.data[:, 0] *= -1

    V = fire.FunctionSpace(mesh, "CG", 3)
    x, y = fire.SpatialCoordinate(mesh)
    r = 0.2
    xc = -0.5
    yc = 0.5

    vp = fire.Function(V)

    c = fire.conditional((x - xc) ** 2 + (y - yc) ** 2 < r**2, 3.0, 1.5)

    vp.interpolate(c)

    xi, yi, zi = spyro.io.write_function_to_grid(vp, V, 10.0 / 1000.0)
    spyro.io.create_segy(zi, segy_file)
    write_velocity_model(segy_file, vp_name)

    model = {}

    model["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": 'equispaced',  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 3,  # p order
        "dimension": 2,  # dimension
    }
    model["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "user_mesh": mesh,
        "mesh_file": None,  # specify the mesh file
    }
    model["BCs"] = {
        "status": False,
    }
    model["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.0,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }
    model["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.0, 1.0)],#, (-0.605, 1.7), (-0.61, 1.7), (-0.615, 1.7)],#, (-0.1, 1.5), (-0.1, 2.0), (-0.1, 2.5), (-0.1, 3.0)],
        "frequency": 5.0,
        "delay": 1.5,
        "receiver_locations": [(-0.0, 0.5)],
    }

    Wave_obj = spyro.AcousticWave(dictionary=model)

    vp_read = spyro.io.interpolate(Wave_obj, hdf5_file, Wave_obj.function_space)

    fire.File("velocity_models/test.pvd").write(vp_read)

    value_at_center = vp_read.at(xc, yc)
    test1 = math.isclose(value_at_center, 3.0)
    value_outside_circle = vp_read.at(xc + r + 0.1, yc)
    test2 = math.isclose(value_outside_circle, 1.5)
    assert all([test1, test2])


def test_saving_shot_record():
    from .inputfiles.model import dictionary
    dictionary["time_axis"]["final_time"] = 0.5
    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(dx=0.02)
    Wave_obj.set_initial_velocity_model(constant=1.5)
    Wave_obj.forward_solve()
    spyro.io.save_shots(Wave_obj, file_name="test_shot_record")


def test_loading_shot_record():
    from .inputfiles.model import dictionary
    dictionary["time_axis"]["final_time"] = 0.5
    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(dx=0.02)
    spyro.io.load_shots(Wave_obj, file_name="test_shot_record")


if __name__ == "__main__":
    test_read_and_write_segy()
    test_saving_shot_record()
    test_loading_shot_record()
