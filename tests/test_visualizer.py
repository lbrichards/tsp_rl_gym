import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# These imports will fail until we create the implementation file
from tsp_rl_gym.utils.visualizer import plot_tour, plot_convergence

@pytest.fixture
def sample_coords():
    """Provides a simple 4-city square for plotting."""
    return np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
    ], dtype=np.float32)

@pytest.fixture
def sample_tour(sample_coords):
    """A sample tour for the coordinates."""
    return np.arange(len(sample_coords))


def test_plot_tour_runs_without_error(sample_coords, sample_tour):
    """
    A smoke test to ensure the plot_tour function can be called
    without crashing and returns a matplotlib Axes object.
    """
    fig, ax = plt.subplots()
    result_ax = plot_tour(ax, sample_coords, sample_tour)
    
    assert isinstance(result_ax, Axes)
    assert result_ax.get_title() is not None # Check if a title was set
    plt.close(fig) # Prevent figures from displaying during tests


def test_plot_convergence_runs_without_error():
    """
    A smoke test to ensure the plot_convergence function can be called
    without crashing and returns a matplotlib Axes object.
    """
    fig, ax = plt.subplots()
    # A sample history of improving fitness values
    history = [100.0, 95.0, 92.5, 90.0, 90.0]
    
    result_ax = plot_convergence(ax, history)
    
    assert isinstance(result_ax, Axes)
    assert ax.get_xlabel() is not None # Check if axis labels were set
    assert ax.get_ylabel() is not None
    plt.close(fig) # Prevent figures from displaying during tests