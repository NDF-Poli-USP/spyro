"""Tests for Non-Reflective BCs applied on the outer boundary of an absorbing layer.

The absorbing layer owns an `NRBC` object (`layer_ops.nrbc`) instead of inheriting
from `NRBC`, so the NRBC keeps its own attributes and does not change the layer's.
"""
from pathlib import Path

import pytest
from numpy import isclose, isfinite

import spyro
from spyro.abc.nrbc import NRBC
from spyro.utils.typing import BoundaryConditionsType


def pml_wave_dictionary() -> dict:
    """Build a small 2D PML model that runs a few time steps.

    Returns
    -------
    dictionary : `dict`
        Model dictionary for `spyro.AcousticWave`.
    """
    return {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 2, "dimension": 2},
        "parallelism": {"type": "automatic"},
        "mesh": {"length_z": 1.0, "length_x": 1.0, "length_y": 0.0,
                 "mesh_file": None, "mesh_type": "firedrake_mesh"},
        "acquisition": {"source_type": "ricker", "source_locations": [(-0.1, 0.5)],
                        "frequency": 5.0, "delay": 0.3, "delay_type": "time",
                        "receiver_locations": [(-0.15, 0.5)]},
        "time_axis": {"initial_time": 0.0, "final_time": 0.01, "dt": 0.001,
                      "amplitude": 1, "output_frequency": 100,
                      "gradient_sampling_frequency": 100},
        "absorving_boundary_conditions": {"status": True, "abc_type": "PML",
                                          "exponent": 2, "cmax": 4.5, "R": 1e-6,
                                          "pad_length": 0.25},
        "visualization": {"forward_output": False, "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None,
                          "gradient_output": False, "gradient_filename": None},
    }


@pytest.mark.parametrize("non_reflect_bc", [
    BoundaryConditionsType.SOMMERFELD,
    pytest.param(BoundaryConditionsType.HIGDON, marks=pytest.mark.xfail(
        strict=True, raises=ValueError,
        reason="NRBC.source_to_bnd_reference_vector indexes the boundary coordinates "
               "by row; fixed with the NRBC refactoring in ruben/test_nrbc_v2")),
])
def test_nrbc_on_pml_layer(non_reflect_bc: BoundaryConditionsType,
                           tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a PML forward solve with NRBCs on the outer boundary of the layer.

    Parameters
    ----------
    non_reflect_bc : `typing.BoundaryConditionsType`
        Non-reflective BC applied on the outer boundary of the PML.
    tmp_path : `pathlib.Path`
        Temporary folder used as working directory for the output files.
    monkeypatch : `pytest.MonkeyPatch`
        Fixture used to change the working directory.

    Returns
    -------
    None
    """
    monkeypatch.chdir(tmp_path)

    wave = spyro.AcousticWave(dictionary=pml_wave_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.set_initial_velocity_model(constant=1.5)

    layer = wave.layer_ops
    layer.bc_boundary_pml = non_reflect_bc
    if non_reflect_bc == BoundaryConditionsType.HIGDON:
        # Set by the Eikonal analysis, which is skipped when the pad length is given
        layer.crit_source = wave.source_locations[0]

    path_save, path_case_absl = layer.path_save, layer.path_case_absl
    assert layer.nrbc is None

    wave.forward_solve()

    # The layer owns the NRBC and keeps its own output paths
    assert isinstance(layer.nrbc, NRBC)
    assert not isinstance(layer, NRBC)
    assert layer.path_save == path_save
    assert layer.path_case_absl == path_case_absl

    # Cosine of the incidence angle is only defined on the outer layer boundary
    cos_hig = layer.nrbc.cosHig.dat.data_ro
    cos_bnd = cos_hig[cos_hig != 0.]
    assert cos_bnd.size > 0
    if non_reflect_bc == BoundaryConditionsType.SOMMERFELD:
        assert isclose(cos_bnd, 1.).all()
    else:
        assert (cos_bnd >= layer.nrbc.cos_min - 1e-12).all()
        assert (cos_bnd <= 1. + 1e-12).all()
    assert Path(path_case_absl, "cosHig.pvd").is_file()

    assert isfinite(wave.forward_solution_receivers).all()
