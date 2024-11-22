import spyro
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt
from scipy.signal import sosfilt
import sys

filter_type = 'butter'

filter_frequency = 7.0


def filter_shot(shot, cutoff, fs, filter_type='butter'):
    if filter_type == 'butter':
        return butter_filter(shot, cutoff, fs)


def butter_filter(shot, cutoff, fs, order=1):
    """ Low-pass filter the shot record with sampling-rate fs Hz
        and cutoff freq. Hz
    """

    nyq = 0.5*fs  # Nyquist Frequency
    normal_cutoff = (cutoff) / nyq

    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    nc, nr = np.shape(shot)

    for rec in range(nr):
        shot[:, rec] = filtfilt(b, a, shot[:, rec])

    return shot


frequency = 7.0
dt = 0.0001
degree = 4

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": degree,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 2.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 2.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_type": "firedrake_mesh"
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 12),
    "frequency": frequency,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.45, 0.7), (-1.45, 1.3), 200),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 0.9,  # Final time for event
    "dt": 0.0001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}
dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}
dictionary["inversion"] = {
    "initial_guess_model_file": None,
    "perform_fwi": True,
    "real_shot_record_files": "shots/shot_record_",
}


fs = 1/dt
fwi = spyro.FullWaveformInversion(dictionary=dictionary)
spyro.io.load_shots(fwi, file_name="shots/shot_record_")
shots = fwi.forward_solution_receivers
shots *= 5.455538535049624
print(f'Applying {filter_type} filter for {filter_frequency}Hz', flush=True)
p_filtered = filter_shot(shots, filter_frequency, fs, filter_type=filter_type)
shot_filename = f"shots/shot_record_{filter_frequency}_"
spyro.io.save_shots(fwi, file_name=shot_filename)
