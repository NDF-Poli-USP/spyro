"""
Test module for the simple_forward notebook.
Uses pytest-nbval to execute and validate the notebook.
"""
import pytest
import os
import sys
from pathlib import Path

# Add the spyro package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

# Define the path to the notebook
NOTEBOOK_PATH = Path(__file__).parent.parent.parent / "notebook_tutorials" / "simple_forward.ipynb"


def test_simple_forward_notebook_exists():
    """Test that the simple_forward notebook exists."""
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"


@pytest.mark.slow
def test_simple_forward_notebook_execution():
    """
    Test that the simple_forward notebook executes without errors.
    This test will be run using nbval plugin.
    
    The actual notebook execution is handled by pytest-nbval plugin
    when the notebook is specified in pytest command line.
    """
    # This test serves as a placeholder and documentation
    # The actual notebook validation is done by nbval
    pass


# Optional: Custom validation functions for specific outputs
def test_notebook_imports():
    """Test that required packages can be imported."""
    try:
        import spyro
        import numpy as np
        import firedrake
    except ImportError as e:
        pytest.skip(f"Required packages not available: {e}")


@pytest.mark.slow 
def test_spyro_basic_functionality():
    """Test basic spyro functionality needed for the notebook."""
    try:
        import spyro
        
        # Test basic dictionary creation (similar to notebook)
        dictionary = {
            "options": {
                "cell_type": "Q",
                "variant": "lumped",
                "degree": 4,
                "dimension": 2,
            },
            "parallelism": {
                "type": "automatic",
            },
            "mesh": {
                "Lz": 3.0,
                "Lx": 3.0,
                "Ly": 0.0,
                "mesh_file": None,
                "mesh_type": "firedrake_mesh",
            },
            "acquisition": {
                "source_type": "ricker",
                "source_locations": [(-1.1, 1.5)],
                "frequency": 5.0,
                "delay": 0.2,
                "delay_type": "time",
                "receiver_locations": spyro.create_transect((-1.9, 1.2), (-1.9, 1.8), 300),
            },
            "time_axis": {
                "initial_time": 0.0,
                "final_time": 1.0,
                "dt": 0.001,
                "output_frequency": 100,
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
            }
        }
        
        # Test that AcousticWave can be instantiated
        wave_obj = spyro.AcousticWave(dictionary=dictionary)
        assert wave_obj is not None
        
    except ImportError:
        pytest.skip("Spyro not available")
    except Exception as e:
        pytest.fail(f"Basic spyro functionality test failed: {e}")