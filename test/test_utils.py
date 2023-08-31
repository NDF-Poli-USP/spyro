import spyro
import scipy as sp
import numpy as np
import math


def test_butter_lowpast_filter():
    Wave_obj = spyro.examples.Rectangle_acoustic()
    layer_values = [1.5, 2.0, 2.5, 3.0]
    z_switches = [-0.25, -0.5, -0.75]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)
    Wave_obj.forward_solve()

    spyro.io.save_shots(Wave_obj, file_name="test_butter_prefilter")
    shot_record = Wave_obj.forward_solution_receivers
    rec10 = shot_record[:, 10]

    fs = 1.0/Wave_obj.dt

    # Checks if frequency with greater power density is close to 5
    (f, S) = sp.signal.periodogram(rec10, fs)
    peak_frequency = f[np.argmax(S)]
    test1 = math.isclose(peak_frequency, 5.0, rel_tol=1e-2)

    # Checks if the new frequency is lower than the cutoff
    cutoff_frequency = 3.0
    filtered_shot = spyro.utils.utils.butter_lowpass_filter(shot_record, cutoff_frequency, fs)
    filtered_rec10 = filtered_shot[:, 10]

    (filt_f, filt_S) = sp.signal.periodogram(filtered_rec10, fs)
    filtered_peak_frequency = filt_f[np.argmax(filt_S)]
    test2 = (filtered_peak_frequency < cutoff_frequency)

    print(f"Peak frequency is close to what it is supposed to be: {test1}")
    print(f"Filtered peak frequency is lower than cutoff frequency: {test2}")

    assert all([test1, test2])


if __name__ == "__main__":
    test_butter_lowpast_filter()
