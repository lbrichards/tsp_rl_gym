import numpy as np
import pytest
from pathlib import Path

# This import will fail until we create the data_loader module
from tsp_rl_gym.utils.data_loader import load_tsplib_instance

def test_load_tsplib_instance():
    """
    Tests loading of a standard TSPLIB .tsp file included with the package.
    """
    # Use pathlib to find the file relative to the package
    file_path = Path(__file__).parent.parent / "tsp_rl_gym" / "data" / "tsplib" / "ulysses16.tsp"
    
    instance = load_tsplib_instance(file_path)

    # Check that the loaded data is correct
    assert isinstance(instance, dict)
    assert "name" in instance
    assert "coords" in instance
    assert instance["name"] == "ulysses16"
    
    coords = instance["coords"]
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (16, 2)
    
    # Verify the coordinates of the first and last nodes
    assert np.allclose(coords[0], [38.24, 20.42])
    assert np.allclose(coords[15], [35.53, -13.12])