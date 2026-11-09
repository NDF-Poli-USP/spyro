"""Performance tests of the maximum stable time step from each modal solver.

Every method computes the largest eigenvalue of the stiffness and mass forms
supplied by the wave physics, and `comp_cost` records its runtime and memory.
The exact eigensolvers must agree on the time step, and `"ANALYTICAL"` must
reproduce its own estimate from the matrix diagonals.
"""

from pathlib import Path

import numpy as np
from firedrake import COMM_WORLD as comm, conditional
from pytest import mark, param

import spyro
from spyro.io.basicio import parallel_print as pprint
from spyro.utils.cost import comp_cost

# Long enough that rounding the time step to a divisor of the final time
# changes it by less than 1e-3 relative
FINAL_TIME = 10.0
# The exact eigensolvers stop at a relative tolerance of 1e-4
RTOL = 1e-3


def wave_instance(
    physics: str, dimension: int, cell_type: str, edge_length: float
) -> spyro.AcousticWave | spyro.IsotropicWave:
    """Build a wave with degree-4 lumped elements, up to its modal forms.

    The 2D domain is 1.0 x 1.5 km and the 3D one is a 0.7 km cube. The
    acoustic velocity has two layers, 1.5 km/s on top of 3.0 km/s, and the
    elastic medium is homogeneous with vp = 3.0 km/s, vs = 1.5 km/s and a
    unit density.

    Parameters
    ----------
    physics : str
        Wave physics, `"acoustic"` or `"elastic"`.
    dimension : int
        Dimension of the problem, 2 or 3.
    cell_type : str
        `"T"` for triangles or `"Q"` for quadrilaterals/hexahedra.
    edge_length : float
        Edge length of the mesh in km.

    Returns
    -------
    spyro.AcousticWave or spyro.IsotropicWave
        Wave with its mesh and material parameters set.
    """
    if dimension == 2:
        length_z, length_x, length_y = 1.0, 1.5, 0.0
    else:
        length_z, length_x, length_y = 0.7, 0.7, 0.7
    center = [-length_z / 2, length_x / 2, length_y / 2][:dimension]

    dictionary = {
        "options": {
            "cell_type": cell_type,
            "variant": "lumped",
            "degree": 4,
            "dimension": dimension,
        },
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": length_z,
            "length_x": length_x,
            "length_y": length_y,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [tuple(center)],
            "frequency": 5.0,
            "receiver_locations": [tuple(center)],
        },
        "time_axis": {
            "final_time": FINAL_TIME,
            "dt": 0.001,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
    }

    if physics == "elastic":
        dictionary["acquisition"]["amplitude"] = np.eye(dimension)[1]
        dictionary["synthetic_data"] = {
            "type": "object",
            "density": 1.0,
            "p_wave_velocity": 3.0,
            "s_wave_velocity": 1.5,
            "real_velocity_file": None,
        }
        wave = spyro.IsotropicWave(dictionary)
        wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
        wave._initialize_model_parameters()
    else:
        wave = spyro.AcousticWave(dictionary=dictionary)
        wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
        wave.set_initial_velocity_model(
            conditional=conditional(wave.mesh_z < -length_z / 2, 3.0, 1.5)
        )
    return wave


def run_timestep_methods(
    wave: spyro.AcousticWave | spyro.IsotropicWave,
    expected_dt: dict[str, float],
    case_name: str,
) -> None:
    """Compute the time step with each method and compare it with its value.

    Parameters
    ----------
    wave : spyro.AcousticWave or spyro.IsotropicWave
        Wave whose maximum stable time step is computed.
    expected_dt : dict[str, float]
        Expected time step, in s, for each method of `Modal_Solver` to run.
    case_name : str
        Name of the case, used for the folder of the cost files.
    """
    cost_folder = Path("output/modal_timestep_test") / case_name
    cost_folder.mkdir(parents=True, exist_ok=True)

    for method, exp_value in expected_dt.items():
        pprint(f"\nModal Solver: {method}", comm=comm)

        # Reference to resource usage
        tRef = comp_cost("tini")

        dt = wave.get_and_set_maximum_dt(fraction=1.0, method=method)

        # Estimating computational resource usage
        comp_cost("tfin", tRef=tRef, user_name=f"{cost_folder}/{method}_")

        met_str = f"Maximum time step {case_name}. Method {method}"
        cmp_str = f"Expected {exp_value:.6e}, got = {dt:.6e}"
        assert np.isclose(dt, exp_value, rtol=RTOL), f"✗ {met_str}  → {cmp_str}"
        pprint(f"✓ {met_str} Verified: {cmp_str}", comm=comm)


def _expected(
    analytical: float,
    exact: float,
    exact_methods: tuple[str, ...] = ("ARNOLDI", "LOBPCG", "KRYLOVSCH_CH"),
) -> dict[str, float]:
    """Map each method to its expected time step.

    Parameters
    ----------
    analytical : float
        Expected time step of `"ANALYTICAL"`, in s.
    exact : float
        Time step shared by the exact eigensolvers, in s.
    exact_methods : tuple[str, ...], optional
        Exact eigensolvers to run. Default is `("ARNOLDI", "LOBPCG",
        "KRYLOVSCH_CH")`.

    Returns
    -------
    dict[str, float]
        Expected time step of each method to run.
    """
    return {"ANALYTICAL": analytical} | dict.fromkeys(exact_methods, exact)


@mark.parametrize(
    "physics, dimension, cell_type, edge_length, expected_dt",
    [
        ("acoustic", 2, "Q", 0.1, _expected(3.984064e-3, 3.483107e-3)),
        ("acoustic", 2, "T", 0.1, _expected(2.768549e-3, 1.841621e-3)),
        ("elastic", 2, "Q", 0.1, _expected(5.040323e-3, 3.880481e-3)),
        ("elastic", 2, "T", 0.1, _expected(3.501401e-3, 1.971609e-3)),
        param(
            "acoustic",
            2,
            "Q",
            0.05,
            _expected(1.992032e-3, 1.740947e-3),
            marks=mark.slow,
        ),
        param(
            "elastic",
            2,
            "T",
            0.05,
            _expected(1.750700e-3, 9.858044e-4),
            marks=mark.slow,
        ),
        param(
            "acoustic",
            3,
            "Q",
            0.1,
            _expected(3.253090e-3, 2.844950e-3),
            marks=mark.slow,
        ),
        # ARNOLDI took over 10 min and LOBPCG over 3 min on this case
        param(
            "elastic",
            3,
            "Q",
            0.1,
            _expected(4.599816e-3, 3.280840e-3, ("KRYLOVSCH_CH",)),
            marks=mark.slow,
        ),
    ],
)
def test_maximum_timestep(
    physics: str,
    dimension: int,
    cell_type: str,
    edge_length: float,
    expected_dt: dict[str, float],
):
    """Compare the maximum stable time step of each modal solver.

    Parameters
    ----------
    physics : str
        Wave physics, `"acoustic"` or `"elastic"`.
    dimension : int
        Dimension of the problem, 2 or 3.
    cell_type : str
        `"T"` for triangles or `"Q"` for quadrilaterals/hexahedra.
    edge_length : float
        Edge length of the mesh in km.
    expected_dt : dict[str, float]
        Expected time step, in s, for each method to run.
    """
    wave = wave_instance(physics, dimension, cell_type, edge_length)
    case_name = f"{physics}_{dimension}d_{cell_type}"
    run_timestep_methods(wave, expected_dt, case_name)
