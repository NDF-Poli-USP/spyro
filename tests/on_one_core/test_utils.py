import spyro
import scipy as sp
import numpy as np
import math


def test_butter_lowpast_filter():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
    "status": False,
    "damping_type": None,
    "pad_length": 0.,
}
    Wave_obj = spyro.examples.Rectangle_acoustic(dictionary=dictionary)
    layer_values = [1.5, 2.0, 2.5, 3.0]
    z_switches = [-0.25, -0.5, -0.75]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)
    Wave_obj.forward_solve()

    spyro.io.save_shots(Wave_obj, file_name="test_butter_prefilter")
    shot_record = Wave_obj.forward_solution_receivers
    rec10 = shot_record[:, 10]

    fs = (1.0 / Wave_obj.dt) + 1

    # Checks if frequency with greater power density is close to 5
    (f, S) = sp.signal.periodogram(rec10, fs)
    peak_frequency = f[np.argmax(S)]
    test1 = math.isclose(peak_frequency, 5.0, rel_tol=1e-2)

    # Checks if the new frequency is lower than the cutoff
    cutoff_frequency = 3.0
    filtered_shot = spyro.utils.utils.butter_lowpass_filter(
        shot_record, cutoff_frequency, fs
    )
    filtered_rec10 = filtered_shot[:, 10]

    (filt_f, filt_S) = sp.signal.periodogram(filtered_rec10, fs)
    filtered_peak_frequency = filt_f[np.argmax(filt_S)]
    test2 = filtered_peak_frequency < cutoff_frequency

    print(f"Peak frequency is close to what it is supposed to be: {test1}")
    print(f"Filtered peak frequency is lower than cutoff frequency: {test2}")

    assert all([test1, test2])


def test_geometry_creation():
    # Checking 3D grid
    points1_3D = spyro.create_3d_grid((0, 0, 0), (1, 1, 1), 5)
    test0 = len(points1_3D) == 5**3
    test1 = points1_3D[0] == (0.0, 0.0, 0.0)
    test2 = points1_3D[3] == (0.0, 0.0, 0.75)
    test3 = points1_3D[6] == (0.25, 0.0, 0.25)
    test4 = points1_3D[12] == (0.5, 0.0, 0.5)
    test5 = points1_3D[18] == (0.75, 0.0, 0.75)
    test6 = points1_3D[124] == (1.0, 1.0, 1.0)

    print("Geometry creation test 0: ", test0)
    print("Geometry creation test 1: ", test1)
    print("Geometry creation test 2: ", test2)
    print("Geometry creation test 3: ", test3)
    print("Geometry creation test 4: ", test4)
    print("Geometry creation test 5: ", test5)
    print("Geometry creation test 6: ", test6)

    assert all([test0, test1, test2, test3, test4, test5, test6])


if __name__ == "__main__":
    test_butter_lowpast_filter()
    test_geometry_creation()
