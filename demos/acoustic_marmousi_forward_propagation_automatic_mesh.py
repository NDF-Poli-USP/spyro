"""Demo script for running a Marmousi2 acoustic forward case with automatic meshing.

This is ``acoustic_marmousi_forward_propagation.py`` without the mesh file:
the triangle mesh is generated with SeismicMesh and adapted to the P-wave
velocity. Each cell is sized for :data:`cells_per_wavelength` cells per
local P-wavelength at the source frequency, with a limited size gradation
between slow and fast regions, so the mesh is coarser in the fast rock than
in the water. Find the cells-per-wavelength your discretization needs with
``demos/acoustic_marmousi2_cpw.py``.

The P-wave velocity SEG-Y of Martin, Wiley and Marfurt (2006),
doi: 10.1190/1.2172306, is not distributed with spyro. Save it as
``velocity_models/vp_marmousi-ii.segy``, or give another file to
:func:`run`. SeismicMesh sizes the mesh from the SEG-Y directly, which is
then interpolated onto the mesh nodes.

This demo has automatic parallelism set up and 1 shot, so any number of
cores works in ``mpiexec -n N_CORES``. From the repository root::

    mpiexec -n N_CORES python3 demos/acoustic_marmousi_forward_propagation_automatic_mesh.py

The script just runs a single acoustic Marmousi2 forward propagation,
saving the generated mesh to ``meshes/marmousi_automatic.msh``, the
velocity to ``vp.pvd``, the pressure to ``results/forward_output.pvd`` and
the shot record to ``shots/``.
"""

from firedrake import VTKFile
from time import perf_counter

import spyro

#: Slowest P-wave velocity of Marmousi2, the water, in km/s.
minimum_velocity = 1.5
#: Cells per local P-wavelength of the mesh, for degree 4 mass lumped triangles.
cells_per_wavelength = 2.67


def build_dictionary(
    degree: int,
    frequency: float,
    final_time: float,
    cpw: float,
    velocity_file: str,
    mesh_file: str,
) -> dict:
    """Build the model dictionary of the acoustic Marmousi2 forward case.

    Parameters
    ----------
    degree : int
        Polynomial degree of the mass lumped triangles.
    frequency : float
        Ricker source peak frequency, in Hz.
    final_time : float
        Final time of the propagation, in s.
    cpw : float
        Cells per local P-wavelength of the mesh.
    velocity_file : str
        P-wave velocity SEG-Y, in km/s, that sizes the mesh.
    mesh_file : str
        File the generated mesh is written to.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.AcousticWave`.
    """
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
        "length_z": 3.5,  # depth in km - always positive
        "length_x": 17.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "mesh_type": "SeismicMesh",
        "velocity_model": velocity_file,
        "cells_per_wavelength": cpw,
        "minimum_velocity": minimum_velocity,
        "source_frequency": frequency,
        "abc_pad_length": 0.0,  # no absorbing layer to pad the domain with
        "output_filename": mesh_file,
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.01, 8.0)],
        "frequency": frequency,
        "delay": 1.0/frequency,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-0.1, 4.0), (-0.1, 12.0), 100),
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.001,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
    }
    return dictionary


def run(
    degree: int = 4,
    frequency: float = 5.0,
    final_time: float = 4.0,
    cpw: float = cells_per_wavelength,
    velocity_file: str = "velocity_models/vp_marmousi-ii.segy",
    mesh_file: str = "meshes/marmousi_automatic.msh",
) -> spyro.AcousticWave:
    """Generate a velocity adapted mesh and run the acoustic Marmousi2 forward propagation.

    Parameters
    ----------
    degree : int, optional
        Polynomial degree of the mass lumped triangles. Default is 4.
    frequency : float, optional
        Ricker source peak frequency, in Hz. Default is 5.0.
    final_time : float, optional
        Final time of the propagation, in s. Default is 4.0.
    cpw : float, optional
        Cells per local P-wavelength of the mesh. Default is
        :data:`cells_per_wavelength`.
    velocity_file : str, optional
        P-wave velocity SEG-Y, in km/s. Default is
        ``velocity_models/vp_marmousi-ii.segy``.
    mesh_file : str, optional
        File the generated mesh is written to. Default is
        ``meshes/marmousi_automatic.msh``.

    Returns
    -------
    spyro.AcousticWave
        The wave, after the forward solve.
    """
    dictionary = build_dictionary(
        degree, frequency, final_time, cpw, velocity_file, mesh_file,
    )

    wave = spyro.AcousticWave(dictionary=dictionary)
    # Fast interpolation reads the velocity at the mesh nodes directly,
    # without building a grid mesh to project from
    wave.set_initial_velocity_model(new_file=velocity_file, fast_interpolate=True)
    start = perf_counter()
    wave.forward_solve()
    end = perf_counter()
    print(f"Time of {end-start}", flush=True)

    VTKFile("vp.pvd").write(wave.c)
    spyro.io.save_shots(wave)
    return wave


if __name__ == "__main__":
    run()
