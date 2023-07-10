import numpy as np
import matplotlib.pyplot as plt
import math

def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time

def ricker_wavelet(t, freq, amp=1.0, delay=1.5, distance_delay = 0.0):
    """Creates a Ricker source function with a
    delay in term of multiples of the distance
    between the minimums.
    """
    # t = t - delay * math.sqrt(6.0) / (math.pi * freq)
    t = t - delay / freq - distance_delay/1.5
    return (
        amp
        * (1.0 - (2.0) * (math.pi * freq) * (math.pi * freq) * t * t)
        * math.exp(
            (-1.0 ) * (math.pi * freq) * (math.pi * freq) * t * t
        )
    )


def full_ricker_wavelet(dt, final_time, frequency, amplitude=1.0, cutoff=None, delay = 1.5, distance_delay = 0.0):
    """Compute the Ricker wavelet optionally applying low-pass filtering
    using cutoff frequency in Hertz.
    """
    nt = int(final_time / dt) + 1 # number of timesteps
    time = 0.0
    full_wavelet = np.zeros((nt,))
    for t in range(nt):
        full_wavelet[t] = ricker_wavelet(time, frequency, amplitude, delay = delay, distance_delay = distance_delay)
        time += dt
    if cutoff is not None:
        fs = 1.0 / dt
        order = 2
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        full_wavelet = filtfilt(b, a, full_wavelet)
    return full_wavelet


# analytical_files =[
#     "analytical_solution_dt_0.0005.npy",
#     "analytical_solution_dt_0.0006.npy",
#     "analytical_solution_dt_0.00075.npy",
#     "analytical_solution_dt_0.0008.npy",
#     "analytical_solution_dt_0.001.npy",
# ]
# numerical_file = [
#     "rec_out0.0005.npy",
#     "rec_out0.0006.npy",
#     "rec_out0.00075.npy",
#     "rec_out0.0008.npy",
#     "rec_out0.001.npy",
# ]
# dts = [
#     0.0005,
#     0.0006,
#     0.00075,
#     0.0008,
#     0.001,
# ]

numerical_files = [
    # "dofs_3D_0p3_quads_rec_out0.0005.npy",
    # "dofs_3D_0p4_quads_rec_out0.0005.npy",
    "dofs_3D_0p5_quads_rec_out0.0005.npy",
    # "dofs_3D_0p6_quads_rec_out0.0005.npy",
    # "dofs_3D_0p7_quads_rec_out0.0005.npy",
]
rs = [
    # 0.3,
    # 0.4,
    0.5,
    # 0.6,
    # 0.7,
]
dt = 0.0005




errors = []

# exp = 0.159155 # r=0.5
# r = 0.5
# exp = r/np.pi
# exp = 0.26526 # r=0.3
# exp = 0.11368
for i in range(len(numerical_files)):
    exp = 1/(4*np.pi*rs[i])
    p_numerical = np.load(numerical_files[i])
    p_numerical = p_numerical.flatten()
    p_analytical = full_ricker_wavelet(0.0005, 1.0, 5.0, distance_delay = rs[i], amplitude=exp)
    time = np.linspace(0.0, 1.0, int(1.0/dt)+1)
    nstart = 800
    nend = 1700
    p_analytical = p_analytical[nstart:nend]
    p_numerical = p_numerical[nstart:nend]
    time = time[nstart:nend]
    plt.plot(time, p_numerical, label="Numerical", color="red")
    plt.plot(time, p_analytical, label="Analytical", color="black", linestyle="--")
    plt.title(f"Pressure at r = {rs[i]} km")
    plt.legend()
    # plt.plot(time, -(p_analytical - p_numerical))
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.show()
    nt = len(time)
    error_time = error_calc(p_numerical, p_analytical, nt)
    errors.append(error_time)
    print(f"r = {rs[i]}")
    print(f"Error = {error_time}")

# plt.loglog(dts, errors)

# theory = [t**2 for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '--')

# theory = [t for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '-.')

# plt.show()

# p_analytical = np.load("analytical_solution_dt_0.0005.npy")

# p_4_full = np.load("rec_out.npy")
# p_4 = p_4_full.flatten()
# time = np.linspace(0.0, 1.0, int(1.0/0.0005))
# nt = len(time)

# plt.plot(time, p_analytical, '--', label="Analytical")
# plt.plot(time, p_4, label="4th order")

# error_time = np.linalg.norm(p_analytical - p_4, 2) / np.sqrt(nt)
# print(error_time)

# # plt.plot(time, 100 *(p_analytical- p_4), '-b', label='difference x100')

# plt.show()

print("End of script")