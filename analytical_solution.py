import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt
import sys

dt = float(sys.argv[1])

sx, sz = (1.0, 1.0)
rx, rz = (1.0, 1.5)
c0 = 1.5

def ricker(t_initial, t_final, dt, f):
    t = np.linspace(-t_initial, t_final-t_initial, int((t_final)/dt))
    tt = (np.pi**2) * (f**2) * (t**2)
    y = (1.0 - 2.0 * tt) * np.exp(- tt)
    # y = zero_ricker(t_initial, t_final, dt, y)
    return y

# def zero_ricker(t_initial, t_final, dt, ricker):
#     t = np.linspace(t_initial, t_final, int((t_final)/dt))
#     zero_time = int(t_initial/(2*dt))
#     for i in range(zero_time):
#         ricker[i] = 0
#     return ricker


def analytical(dt, t_final, f0):
    # Fourier constants
    nt = int((t_final)/dt)
    nf = int(nt/2 + 1)
    fnyq = 1. / (2 * dt)
    df = 1.0 / t_final
    faxis = df * np.arange(nf)

    # wavelet = ricker(f0, t_final, dt, 1.5/f0)
    # wavelet = ricker(t_initial, t_final, dt, f0)
    wavelet = ricker(1.5/f0, t_final, dt, f0)

    # Take the Fourier transform of the source time-function
    R = np.fft.fft(wavelet)
    R = R[0:nf]
    nf = len(R)

    # Compute the Hankel function and multiply by the source spectrum
    U_a = np.zeros((nf), dtype=complex)
    for a in range(1, nf-1):
        k = 2 * np.pi * faxis[a] / c0
        tmp = k * np.sqrt(((rx - sx))**2 + ((rz - sz))**2)
        U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * R[a]

    # Do inverse fft on 0:dt:T and you have analytical solution
    U_t = 1.0/(2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], nt))

    # The analytic solution needs be scaled by dx^2 to convert to pressure
    return np.real(U_t)/c0**2

# t_initial = 0.5
t_final = 20.0
f0 = 5

# r = ricker(1.5/f0, t_final, dt, f0)

# plt.plot(x, r)
# plt.show()

p = analytical(dt, t_final, f0)
print(np.shape(p))

t_final = t_final/20.0
nt = int((t_final)/dt)
x = np.linspace(0.0, t_final, int((t_final)/dt))
p = p[0:nt]

# plt.plot(x, p)
# plt.show()

np.save("analytical_solution_dt_"+str(dt)+".npy", p)
# U_t = analytical(t_initial, dt, t_final, f0)
# U_t = U_t[0:1501]

print('END')