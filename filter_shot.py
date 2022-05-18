import imp
from unicodedata import name
import spyro
import numpy as np
from numpy.fft import fft, ifft, rfft
import matplotlib.pyplot as plt
from firedrake import File
import copy
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,remez)
from   scipy.signal  import butter, filtfilt, sosfilt
from scipy.signal import sosfilt
from scipy.signal import zpk2sos
from scipy.fftpack import fftshift, ifftshift
from copy import deepcopy



def butter_lowpass_filter_source(wavelet, cutoff, fs, order=2):
    """Low-pass filter the shot record with sampling-rate fs Hz
    and cutoff freq. Hz
    """
    filtered_shot = deepcopy(wavelet)

    fe = 0.5 * fs  # Nyquist Frequency
    f = cutoff/fe

    z, p, k, = iirfilter(order, f, btype='lowpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)

    firstpass = sosfilt(sos,wavelet )
    filtered_shot = sosfilt(sos, firstpass[::-1])[::-1]*7.4
        
    return filtered_shot

def plot_receivers(p_exact_1,p_filter, final_time, dt):
    nt = int(final_time/ dt)  # number of timesteps
    times = np.linspace(0.0, final_time, nt)

    plt.plot(times, p_exact_1, label='not filtered 10Hz')
    plt.plot(times, p_filter, label = 'filtered')
    plt.legend()

    plt.xlabel("time (s)", fontsize=18)
    plt.ylabel("amplitude", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(start_index, end_index)
    # plt.ylim(tf, 0)

    plt.show()
    plt.close()
    return None

def plot_source(p_exact_1,p_exact_2,p_filter, final_time, dt):
    nt = int(final_time/ dt)  # number of timesteps
    times = np.linspace(0.0, final_time, nt)

    plt.plot(times, p_exact_1, label='not filtered 10Hz')
    plt.plot(times, p_exact_2, label='not filtered 5Hz')
    plt.plot(times, p_filter, label = 'filtered')
    plt.legend()

    plt.xlabel("time (s)", fontsize=18)
    plt.ylabel("amplitude", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.xlim(start_index, end_index)
    # plt.ylim(tf, 0)

    plt.show()
    plt.close()
    return None

def plot_shot_record(
    model,
    comm,
    p1,
    p2,
    show=True,
    vmin=-1e-2,
    vmax=1e-2,
    start_index=0,
    end_index=0,
    name = None,
):
    """Plot a shot record and save the image to disk. Note that
    this automatically will rename shots when ensmeble paralleism is
    activated.
    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    comm:A Firedrake commmunicator
        The communicator you get from calling spyro.utils.mpi_init()
    arr: array-like
        An array in which rows are intervals in time and columns are receivers
    show: `boolean`, optional
        Should the images appear on screen?
    file_name: string, optional
        The name of the saved image
    vmin: float, optional
        The minimum value to plot on the colorscale
    vmax: float, optional
        The maximum value to plot on the colorscale
    file_format: string, optional
        File format, pdf or png
    start_index: integer, optional
        The index of the first receiver to plot
    end_index: integer, optional
        The index of the last receiver to plot
    Returns
    -------
    None
    """

    num_recvs = len(model["acquisition"]["receiver_locations"])

    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]

    nt = int(tf / dt)  # number of timesteps

    if end_index == 0:
        end_index = num_recvs

    x_rec = np.linspace(start_index, end_index, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    cmap = plt.get_cmap("gray")
    plt.subplot(1,2,1)
    plt.contourf(X, Y, p1, levels =100, vmin=vmin, vmax=vmax)
    # savemat("test.mat", {"mydata": arr})
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)

    plt.subplot(1,2,2)
    plt.contourf(X, Y, p2, levels =100, vmin=vmin, vmax=vmax)
    # savemat("test.mat", {"mydata": arr})
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    # plt.axis("image")
    if name != None:
        plt.title(name)
    plt.show()
    plt.close()
    return None

def weiner_filter_shot(shot, filter_frequency, dt, final_time):
    num_times, num_receivers = np.shape(shot)
    filtered_shot = np.zeros((num_times, num_receivers))
    target = spyro.full_ricker_wavelet(dt,final_time,filter_frequency)

    fe = 0.5 * (1/dt)  # Nyquist Frequency
    f = filter_frequency/fe
    nt = int(final_time/ dt)
    time_domain = np.linspace(0.0, final_time,nt)
    frequency_domain = np.linspace(-np.pi,np.pi, len(time_domain))

    target_f = fftshift(fft(target) ) / len(time_domain)

    for receiver in range(num_receivers):
        source = copy.deepcopy(shot[:,receiver])
        
        e = 0.0001
        source_f = fftshift( fft(source)  ) / len(time_domain)
        f = target_f * np.conjugate(source_f) / ( np.abs(source_f)**2  +e**2 )
        new_source_f = f*source_f
        new_source = ifftshift(  ifft(new_source_f) )
        filtered_shot[:,receiver] = new_source

    return filtered_shot


def butter_lowpass_filter(shot, cutoff, fs, order=4):
    """Low-pass filter the shot record with sampling-rate fs Hz
    and cutoff freq. Hz
    """
    num_times, num_receivers = np.shape(shot)
    filtered_shot = np.zeros((num_times, num_receivers))

    fe = 0.5 * fs  # Nyquist Frequency
    f = cutoff/fe

    z, p, k, = iirfilter(order, f, btype='lowpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)

    for receiver, ts in enumerate(shot.T):
        firstpass = sosfilt(sos,shot[:,receiver] )
        filtered_shot[:,receiver] = sosfilt(sos, firstpass[::-1])[::-1]
        
    return filtered_shot

def butter_filter(shot, cutoff, fs, order=1):

    """ Low-pass filter the shot record with sampling-rate fs Hz
        and cutoff freq. Hz
    """
    
    nyq = 0.5*fs  # Nyquist Frequency
    normal_cutoff = (cutoff) / nyq
    filtered_shot = np.copy(shot)
  
    # Get the filter coefficients  
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    
    nc, nr = np.shape(shot)

    for rec in range(nr):
        
        filtered_shot[:,rec] = filtfilt(b, a, shot[:,rec])
    
    return filtered_shot




model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "automatic",
}
model["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/forward_10.0Hz.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "velocity_models/vp_marmousi-ii.hdf5",
}
model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": spyro.create_transect((-0.01, 4.0), (-0.01, 15.0), 1),
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 200,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 200),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 5.00,  # Final time for event
    "dt": 0.00025,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

## Creates a 10Hz and 5Hz source wavelet
wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    final_time=model["timeaxis"]["tf"],
    frequency=10.0,
)

wavelet5 = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    final_time=model["timeaxis"]["tf"],
    frequency=5.0,
)

## Loads 10 Hz and 5Hz shot records
p = spyro.io.load_shots(model, comm)
p5 = spyro.io.load_shots(model, comm, file_name='shots/5Hz.dat')


## Applies a filter
#p_filter= weiner_filter_shot(p, 5.0, 0.00025, 5.0)
p_filter = butter_filter(p, 5.0, 1/0.00025)
#p_filter = butter_lowpass_filter(p, 5.0, 1./0.00025, order=2)

## Plot_receivers can be used to plot a recording from one receiver or a wavelet
receiver_id = 100
plot_receivers(p[:,receiver_id], p_filter[:,receiver_id], 5.0, 0.00025)

plot_source(wavelet, wavelet5,butter_lowpass_filter_source(wavelet, 5.0, 1/0.00025, order=2), 5.0, 0.00025)


## Plot shot records 
plot_shot_record(model, comm, p, p5, vmin=-1e-2, vmax=1e-2, name="Comparison of unfiltered 10Hz and 5Hz shots")
#plot_shot_record(model, comm, p, p_filter, vmin=-1e-2, vmax=1e-2)
plot_shot_record(model, comm, p5, p_filter, vmin=-1e-2, vmax=1e-2, name = "Comparison of unfiltered and filtered 5Hz")
#plot_shot_record(model, comm, p_filter, vmin=-1e-2, vmax=1e-2)

