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


dts = [
    0.0001,
    0.00005,
]

offset = 0.5
final_time = 1.0
errors = []

for dt in dts:
    rec_full = np.load("interior_3D_MLT4_dt"+str(dt)+".npy")
    rec = rec_full.flatten()
    ana = analytical_solution(dt, final_time, offset)

    cutoff = 0.6
    cutoff_index = int(cutoff/dt)

    rec_cut = rec[:cutoff_index]
    ana_cut = ana[:cutoff_index]

    nt = len(ana_cut)

    errors.append(error_calc(rec_cut, ana_cut, nt))

plt.loglog(dts, errors)

theory = [t**2 for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '--')

theory = [t for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '-.')
plt.show()
