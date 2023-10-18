import pickle
import numpy as np


dt = 0.001

def error_calc(receivers, analytical, dt):
    rec_len, num_rec = np.shape(receivers)

    # Interpolate analytical solution into numerical dts
    final_time = dt * (rec_len - 1)
    time_vector_rec = np.linspace(0.0, final_time, rec_len)
    time_vector_ana = np.linspace(0.0, final_time, len(analytical[:, 0]))
    ana = np.zeros(np.shape(receivers))
    for i in range(num_rec):
        ana[:, i] = np.interp(
            time_vector_rec, time_vector_ana, analytical[:, i]
        )

    total_numerator = 0.0
    total_denumenator = 0.0
    for i in range(num_rec):
        diff = receivers[:, i] - ana[:, i]
        diff_squared = np.power(diff, 2)
        numerator = np.trapz(diff_squared, dx=dt)
        ref_squared = np.power(ana[:, i], 2)
        denominator = np.trapz(ref_squared, dx=dt)
        total_numerator += numerator
        total_denumenator += denominator

    squared_error = total_numerator / total_denumenator

    error = np.sqrt(squared_error)
    return error


with open("cpw3p0rec.pck", "rb") as f:
    rec3p0 = pickle.load(f)
with open("cpw5p0rec.pck", "rb") as f:
    rec5p0 = pickle.load(f)

print("END")
