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
]

offset = 0.5
final_time = 1.0
errors = []

for dt in dts:
    rec_full = np.load("interior_3D_ML3Tet_dt"+str(dt)+".npy")
    rec = rec_full.flatten()
    ana = analytical_solution(dt, final_time, offset)

    start = 0.35
    cutoff = 0.76
    start_index = int(start/dt)+1
    cutoff_index = int(cutoff/dt)+1

    rec_cut = rec[start_index:cutoff_index]
    ana_cut = ana[start_index:cutoff_index]

    nt = len(ana_cut)
    error = error_calc(rec_cut, ana_cut, nt)
    print(error)

    errors.append(error)

timevector = np.linspace(0.0, final_time, len(rec_full))
timevector_cut = np.linspace(start, cutoff, len(rec_cut))
# plt.plot(timevector, rec)
# plt.show()
plt.plot(timevector_cut, rec_cut, label="Numerical")
plt.plot(timevector_cut, ana_cut, "--", label="Analytical")
plt.title("Pressure at r = 0.5 km")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (Pa)")
plt.legend()
plt.savefig("mls_3d_ricker_propagation_comparison.png")
plt.show()
# plt.loglog(dts, errors)

# theory = [t**2 for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '--')

# theory = [t for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '-.')
# plt.show()
print("END")
