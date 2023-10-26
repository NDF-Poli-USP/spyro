import numpy as np
import matplotlib.pyplot as plt
import spyro


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    diff = p_analytical - p_numerical
    error_time = np.linalg.norm(diff, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def analytical_solution(dt, final_time, offset):
    amplitude = 1/(4*np.pi*offset)
    delay = offset/1.5 + 1.5 * np.sqrt(6.0) / (np.pi * 5.0)
    p_analytic = spyro.full_ricker_wavelet(
        dt, final_time,
        5.0,
        delay=delay,
        delay_type='time',
        amplitude=amplitude,
    )
    return p_analytic


offset = 0.5
final_time = 1.0
dt = 0.0001
rec_full = np.load("interior_3D_MLT4_dt"+str(dt)+".npy")
rec = rec_full.flatten()
# ana = analytical_solution(dt, final_time, offset)
# np.save("ana_3D_MLT4_dt"+str(dt)+".npy", ana)
ana = np.load("ana_3D_MLT4_dt"+str(dt)+".npy")

cutoff = 0.7
cutoff_index = int(cutoff/dt)

rec_cut = rec[:cutoff_index]
ana_cut = ana[:cutoff_index]

nt = len(ana_cut)

error = error_calc(rec_cut, ana_cut, nt)

print(f"Error is: {error}")
