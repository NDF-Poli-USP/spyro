"""Demo script for running a Marmousi2 acoustic forward case with a PML.

This is ``acoustic_marmousi_forward_propagation_automatic_mesh.py`` with a
perfectly matched layer (PML) absorbing the waves that leave the domain.
The layer is :data:`pad_length` thick, by default one water wavelength, and
surrounds the domain except at the sea surface. The triangle mesh is generated with SeismicMesh and adapted to the P-wave
velocity, and covers the layer too. Each cell is sized for :data:`cells_per_wavelength` cells per
local P-wavelength at the source frequency, with a limited size gradation
between slow and fast regions, so the mesh is coarser in the fast rock than
in the water. Find the cells-per-wavelength your discretization needs with
``demos/acoustic_marmousi2_cpw.py``.

The P-wave velocity SEG-Y of Martin, Wiley and Marfurt (2006),
doi: 10.1190/1.2172306, is not distributed with spyro. Save it as
``velocity_models/vp_marmousi-ii.segy``, or give another file to
:func:`run`. SeismicMesh sizes the mesh from the SEG-Y directly, which is
then interpolated onto the mesh nodes, the layer taking the velocity of the
nearest domain boundary.

This demo has automatic parallelism set up and 1 shot, so any number of
cores works in ``mpiexec -n N_CORES``. From the repository root::

    mpiexec -n N_CORES python3 demos/acoustic_marmousi_forward_propagation_pml.py

The script just runs a single acoustic Marmousi2 forward propagation,
saving the generated mesh to ``meshes/marmousi_pml.msh``, the velocity to
``vp.pvd``, the pressure to ``results/forward_output.pvd`` and the shot
record to ``shots/``.
"""

from firedrake import VTKFile
from time import perf_counter

import spyro

#: Slowest P-wave velocity of Marmousi2, the water, in km/s.
minimum_velocity = 1.5
#: Cells per local P-wavelength of the mesh, for degree 4 mass lumped triangles.
cells_per_wavelength = 2.67
#: Fastest P-wave velocity of Marmousi2, in km/s, which sets the PML damping.
maximum_velocity = 4.7


def build_dictionary(
    degree: int,
    frequency: float,
    final_time: float,
    cpw: float,
    pad_length: float,
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
    pad_length : float
        Thickness of the PML, in km.
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
        "abc_pad_length": pad_length,
        "output_filename": mesh_file,
    }
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "abc_type": "PML",
        "exponent": 2,  # of the polynomial damping profile
        "cmax": maximum_velocity,
        "R": 1e-6,  # theoretical reflection coefficient
        "pad_length": pad_length,
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
    pad_length: float | None = None,
    velocity_file: str = "velocity_models/vp_marmousi-ii.segy",
    mesh_file: str = "meshes/marmousi_pml.msh",
) -> spyro.AcousticWave:
    """Run the acoustic Marmousi2 forward propagation with a PML.

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
    pad_length : float or None, optional
        Thickness of the PML, in km. Default is None, for one water
        wavelength at the source frequency.
    velocity_file : str, optional
        P-wave velocity SEG-Y, in km/s. Default is
        ``velocity_models/vp_marmousi-ii.segy``.
    mesh_file : str, optional
        File the generated mesh is written to. Default is
        ``meshes/marmousi_pml.msh``.

    Returns
    -------
    spyro.AcousticWave
        The wave, after the forward solve.
    """
    if pad_length is None:
        pad_length = minimum_velocity/frequency
    dictionary = build_dictionary(
        degree, frequency, final_time, cpw, pad_length, velocity_file, mesh_file,
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
