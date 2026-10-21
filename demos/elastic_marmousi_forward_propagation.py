"""Demo script for running an isotropic elastic Marmousi2 forward case.

The P-wave velocity, S-wave velocity and density are read from the
Marmousi2 SEG-Ys of Martin, Wiley and Marfurt (2006),
doi: 10.1190/1.2172306, which are not distributed with spyro. Save them as
``velocity_models/vp_marmousi-ii.segy``,
``velocity_models/vs_marmousi-ii.segy`` and
``velocity_models/density_marmousi-ii.segy``, or give other files to
:func:`run`. Each is converted to an HDF5 file next to it, and
interpolated onto the mesh. The top 0.45 km is water, an elastic material
with zero S-wave velocity.

The mesh is generated, uniform, and sized by the slowest S-wave in the
rock. Find the cells-per-wavelength your discretization needs with
``demos/elastic_marmousi2_cpw.py``. The time step is the largest stable
one estimated for the mesh and material. Every boundary has a first-order
Stacey absorbing condition.

This demo has automatic parallelism set up and 1 shot, so any number of
cores works in ``mpiexec -n N_CORES``. The elastic mesh is much finer than
the acoustic one of ``acoustic_marmousi_forward_propagation.py``, so the
run is long in serial. From the repository root::

    mpiexec -n N_CORES python3 demos/elastic_marmousi_forward_propagation.py

The script just runs a single isotropic elastic Marmousi2 forward
propagation, saving the P-wave velocity, S-wave velocity and density to
``vp.pvd``, ``vs.pvd`` and ``rho.pvd``, the displacement to
``results/elastic_marmousi_forward_output.pvd`` and the shot record to
``shots/``.
"""

import numpy as np
from firedrake import VTKFile

import spyro

#: Slowest nonzero S-wave velocity of Marmousi2, in km/s.
minimum_s_wave_velocity = 0.27
#: Default cells per slowest S-wavelength of the mesh.
cells_per_wavelength = 2.5


def build_dictionary(
    degree: int,
    frequency: float,
    edge_length: float,
    final_time: float,
    material_files: dict[str, str],
) -> dict:
    """Build the model dictionary of the elastic Marmousi2 forward case.

    Parameters
    ----------
    degree : int
        Polynomial degree of the spectral quadrilaterals.
    frequency : float
        Ricker source peak frequency, in Hz.
    edge_length : float
        Edge length of the generated mesh, in km.
    final_time : float
        Final time of the propagation, in s.
    material_files : dict of str to str
        SEG-Y or HDF5 file of the P-wave velocity, S-wave velocity and
        density, keyed by their ``"synthetic_data"`` names.

    Returns
    -------
    dict
        Spyro model dictionary for :class:`spyro.IsotropicWave`.
    """
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
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
        "mesh_type": "firedrake_mesh",
        "edge_length": edge_length,
    }
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "abc_type": "nrbc",
        "nrbc": {"type": "Stacey", "dt_scheme": "backward"},
    }
    dictionary["synthetic_data"] = {"type": "file", **material_files}
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.01, 8.0)],
        "frequency": frequency,
        "delay": 1.0/frequency,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-0.1, 4.0), (-0.1, 12.0), 100),
        # Horizontal (x-direction) point force
        "amplitude": np.array([0.0, 1.0]),
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.001,  # replaced by the estimated maximum stable timestep
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/elastic_marmousi_forward_output.pvd",
    }
    return dictionary


def run(
    degree: int = 4,
    frequency: float = 5.0,
    edge_length: float | None = None,
    final_time: float = 4.0,
    p_wave_velocity_file: str = "velocity_models/vp_marmousi-ii.segy",
    s_wave_velocity_file: str = "velocity_models/vs_marmousi-ii.segy",
    density_file: str = "velocity_models/density_marmousi-ii.segy",
) -> spyro.IsotropicWave:
    """Run the elastic Marmousi2 forward propagation.

    Parameters
    ----------
    degree : int, optional
        Polynomial degree of the spectral quadrilaterals. Default is 4.
    frequency : float, optional
        Ricker source peak frequency, in Hz. Default is 5.0.
    edge_length : float or None, optional
        Edge length of the generated mesh, in km. Default is None, for
        :data:`cells_per_wavelength` cells per slowest S-wavelength.
    final_time : float, optional
        Final time of the propagation, in s. Default is 4.0.
    p_wave_velocity_file : str, optional
        P-wave velocity SEG-Y or HDF5, in km/s or m/s. Default is
        ``velocity_models/vp_marmousi-ii.segy``.
    s_wave_velocity_file : str, optional
        S-wave velocity SEG-Y or HDF5, in km/s. Default is
        ``velocity_models/vs_marmousi-ii.segy``.
    density_file : str, optional
        Density SEG-Y or HDF5, in g/cm^3. Default is
        ``velocity_models/density_marmousi-ii.segy``.

    Returns
    -------
    spyro.IsotropicWave
        The wave, after the forward solve.
    """
    if edge_length is None:
        edge_length = minimum_s_wave_velocity/frequency/cells_per_wavelength
    material_files = {
        "p_wave_velocity": p_wave_velocity_file,
        "s_wave_velocity": s_wave_velocity_file,
        "density": density_file,
    }
    dictionary = build_dictionary(
        degree, frequency, edge_length, final_time, material_files,
    )

    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    # Read here, before the forward solve, as the timestep estimate needs it
    wave.initialize_model_parameters_from_file(dictionary["synthetic_data"])
    wave.get_and_set_maximum_dt(fraction=0.7)

    wave.forward_solve()

    VTKFile("vp.pvd").write(wave.c)
    VTKFile("vs.pvd").write(wave.c_s)
    VTKFile("rho.pvd").write(wave.rho)
    spyro.io.save_shots(wave)
    return wave


if __name__ == "__main__":
    run()
