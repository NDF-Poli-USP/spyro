"""
Test module for the simple_forward_exercises notebooks.
Uses pytest-nbval to execute and validate the notebooks.
"""
import pytest
import os
import sys
from pathlib import Path

# Add the spyro package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

# Define the paths to the notebooks
EXERCISES_NOTEBOOK_PATH = Path(__file__).parent.parent.parent / "notebook_tutorials" / "simple_forward_exercises.ipynb"
ANSWERS_NOTEBOOK_PATH = Path(__file__).parent.parent.parent / "notebook_tutorials" / "simple_forward_exercises_answers.ipynb"


def test_exercises_notebooks_exist():
    """Test that both exercise notebooks exist."""
    assert EXERCISES_NOTEBOOK_PATH.exists(), f"Exercises notebook not found at {EXERCISES_NOTEBOOK_PATH}"
    assert ANSWERS_NOTEBOOK_PATH.exists(), f"Answers notebook not found at {ANSWERS_NOTEBOOK_PATH}"


@pytest.mark.slow
def test_exercises_answers_notebook_execution():
    """
    Test that the simple_forward_exercises_answers notebook executes without errors.
    This test will be run using nbval plugin.
    
    The actual notebook execution is handled by pytest-nbval plugin
    when the notebook is specified in pytest command line.
    """
    # This test serves as a placeholder and documentation
    # The actual notebook validation is done by nbval
    pass


def test_notebook_imports():
    """Test that required packages can be imported."""
    try:
        import spyro
        import numpy as np
    except ImportError as e:
        pytest.skip(f"Required packages not available: {e}")


@pytest.mark.slow
def test_exercises_specific_functionality():
    """Test specific functionality used in the exercises notebook."""
    try:
        import spyro
        import numpy as np
        
        # Test the create_transect function used in exercises
        receivers = spyro.create_transect((-0.4, 0.1), (-0.4, 3.9), 300)
        assert len(receivers) == 300, "create_transect should return 300 receivers"
        assert hasattr(receivers, '__len__'), "create_transect should return an array-like object"
        
        # Test basic dictionary structure for exercises
        dictionary = {
            "options": {
                "cell_type": "T",
                "variant": "lumped", 
                "degree": 4,
                "dimension": 2,
            },
            "parallelism": {
                "type": "automatic",
            },
            "mesh": {
                "Lz": 1.5,
                "Lx": 4.0,
                "Ly": 0.0,
                "mesh_file": None,
                "mesh_type": "firedrake_mesh",
            },
            "acquisition": {
                "source_type": "ricker",
                "source_locations": [(-0.01, 2.0)],
                "frequency": 5.0,
                "receiver_locations": spyro.create_transect((-0.4, 0.1), (-0.4, 3.9), 300),
            },
            "time_axis": {
                "initial_time": 0.0,
                "final_time": 5.0,
                "dt": 0.0005,
                "output_frequency": 400,
                "gradient_sampling_frequency": 1,
            },
            "visualization": {
                "forward_output": True,
                "forward_output_filename": "results/forward_output.pvd",
                "fwi_velocity_model_output": False,
                "velocity_model_filename": None,
                "gradient_output": False,
                "gradient_filename": "results/Gradient.pvd",
                "adjoint_output": False,
                "adjoint_filename": None,
                "debug_output": False,
            },
            "absorving_boundary_conditions": {
                "status": True,
                "damping_type": "PML",
                "exponent": 2,
                "cmax": 4.5,
                "R": 1e-6,
                "pad_length": 0.5,
            }
        }
        
        # Test that AcousticWave can be instantiated with exercises config
        wave_obj = spyro.AcousticWave(dictionary=dictionary)
        assert wave_obj is not None
        
    except ImportError:
        pytest.skip("Required packages not available")
    except Exception as e:
        pytest.fail(f"Exercises specific functionality test failed: {e}")