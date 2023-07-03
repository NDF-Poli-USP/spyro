import numpy as np
from scipy.special import hankel2
import matplotlib.pyplot as plt
import sys

dt = float(sys.argv[1])
# dt = 0.0005
# p_numerical = np.load("test_quads_rec_out5e-05.npy")
# dt = 0.1
t_final = 1.0
nnt = (np.divide(t_final, dt) + 1).astype(int)

sx, sz = (1.5, 1.5)
rx, rz = (1.5, 2.0)
c0 = 1.5

def ricker(t_initial, t_final, dt, f):
    t = np.linspace(-t_initial, t_final-t_initial, int((t_final)/dt)+1)
    t = t + dt
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


def analytical(nt, dt, t_final, f0):
    # Fourier constants
    nf = int(nt/2 + 1)
    fnyq = 1. / (2 * dt)
    df = 1.0 / t_final
    faxis = df * np.arange(nf)

    # wavelet = ricker(f0, t_final, dt, 1.5/f0)
    # wavelet = ricker(t_initial, t_final, dt, f0)
    print(1.5/f0)
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
    return np.real(U_t)

# t_initial = 0.5
f0 = 5

time1 = np.linspace(0.0, 20*t_final, 20*(nnt-1)+1 )
p = analytical(20*(nnt-1)+1, time1[1]-time1[0], 20*t_final, f0)
print(np.shape(p))

t_final = t_final
nt = int((t_final)/dt)+1
p = p[0:nt]
x = np.linspace(0.0, t_final, int((t_final)/dt)+1)

r = ricker(1.5/f0, t_final, dt, f0)

# plt.plot(x, r)
# plt.show()
# plt.plot(x, p)
# plt.show()

np.save("analytical_solution_dt_"+str(dt)+".npy", p)
# U_t = analytical(t_initial, dt, t_final, f0)
# U_t = U_t[0:1501]

print('END')