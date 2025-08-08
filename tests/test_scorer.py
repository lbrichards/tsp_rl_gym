import numpy as np
import pytest

from tsp_rl_gym.utils.tsp_ops import calculate_distance_matrix, calculate_tour_length

# This import will fail until we create the scorer class
from tsp_rl_gym.envs.core_scorer import CoreScorer

@pytest.fixture
def sample_coords():
    """A set of 4 coordinates forming a simple square of side length 1."""
    return np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [1.0, 1.0],  # Node 2
        [0.0, 1.0],  # Node 3
    ], dtype=np.float32)

def test_scorer_initialization(sample_coords):
    """
    Tests that the CoreScorer initializes correctly, calculating the
    distance matrix and a baseline tour length (L0).
    """
    scorer = CoreScorer(coords=sample_coords)
    assert scorer.D.shape == (4, 4)
    assert scorer.L0 > 0
    # For a square, a random tour will likely be 3*sqrt(1) + sqrt(2) or 2*sqrt(1) + 2*sqrt(2)
    # A simple check is that it's greater than the optimal length of 4.
    assert scorer.L0 >= 4.0

def test_scorer_length(sample_coords):
    """
    Tests the scorer's length calculation method.
    """
    scorer = CoreScorer(coords=sample_coords)
    
    # Optimal tour for a square is 0-1-2-3-0
    tour = np.array([0, 1, 2, 3])
    length = scorer.length(tour)
    assert np.isclose(length, 4.0)

    # Sub-optimal tour 0-2-1-3-0
    tour_2 = np.array([0, 2, 1, 3])
    length_2 = scorer.length(tour_2)
    expected_length_2 = np.sqrt(2.0) + np.sqrt(2.0) + 1.0 + 1.0
    assert np.isclose(length_2, expected_length_2)

def test_scorer_delta_reward(sample_coords):
    """
    Tests the delta method, which calculates the normalized improvement (reward).
    """
    scorer = CoreScorer(coords=sample_coords)
    
    tour_prev = np.array([0, 2, 1, 3]) # The longer tour
    tour_new = np.array([0, 1, 2, 3])  # The shorter, optimal tour

    L_prev = scorer.length(tour_prev)
    L_new = scorer.length(tour_new)

    # The reward should be the normalized improvement
    expected_reward = (L_prev - L_new) / scorer.L0
    actual_reward = scorer.delta(tour_prev, tour_new)
    assert np.isclose(actual_reward, expected_reward)

    # If the tour gets worse, the reward should be negative
    negative_reward = scorer.delta(tour_new, tour_prev)
    assert negative_reward < 0
    assert np.isclose(negative_reward, -expected_reward)