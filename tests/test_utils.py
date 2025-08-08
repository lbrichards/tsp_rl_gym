import numpy as np
import pytest

# We will create these functions in the next step.
from tsp_rl_gym.utils.tsp_ops import (
    calculate_distance_matrix,
    calculate_tour_length,
    apply_2_opt_swap,
    generate_nn_tour,
    build_action_pairs,
    canonicalize_tour,
)

@pytest.fixture
def sample_coords():
    """A set of 4 coordinates forming a simple square."""
    return np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [1.0, 1.0],  # Node 2
        [0.0, 1.0],  # Node 3
    ])

@pytest.fixture
def sample_dist_matrix(sample_coords):
    """Pre-calculated distance matrix for the square coordinates."""
    return calculate_distance_matrix(sample_coords)


def test_calculate_distance_matrix(sample_coords):
    """
    Tests that the distance matrix is calculated correctly.
    The diagonal should be 0, and off-diagonal elements should be Euclidean distances.
    """
    dist_matrix = calculate_distance_matrix(sample_coords)
    assert dist_matrix.shape == (4, 4)
    assert np.all(np.diag(dist_matrix) == 0)
    # Distance from (0,0) to (1,1) should be sqrt(2)
    assert np.isclose(dist_matrix[0, 2], np.sqrt(2))
    # Distance from (0,0) to (1,0) should be 1.0
    assert np.isclose(dist_matrix[0, 1], 1.0)


def test_calculate_tour_length(sample_dist_matrix):
    """
    Tests that the total length of a tour is calculated correctly.
    """
    # A simple tour 0 -> 1 -> 2 -> 3 -> 0
    tour = np.array([0, 1, 2, 3])
    # Length should be 1.0 + 1.0 + 1.0 + 1.0 = 4.0
    expected_length = 4.0
    actual_length = calculate_tour_length(tour, sample_dist_matrix)
    assert np.isclose(actual_length, expected_length)

    # A different tour 0 -> 2 -> 1 -> 3 -> 0
    tour_2 = np.array([0, 2, 1, 3])
    # Length should be sqrt(2) + 1.0 + sqrt(2) + 1.0
    expected_length_2 = 2 * np.sqrt(2) + 2.0
    actual_length_2 = calculate_tour_length(tour_2, sample_dist_matrix)
    assert np.isclose(actual_length_2, expected_length_2)


def test_apply_2_opt_swap():
    """
    Tests the 2-opt swap, which reverses a segment of the tour.
    """
    original_tour = np.array([0, 1, 2, 3, 4, 5])

    # Reverse the segment from index 1 to 3 (inclusive)
    # The segment is [1, 2, 3]. Reversed is [3, 2, 1]
    swapped_tour = apply_2_opt_swap(original_tour.copy(), 1, 3)
    expected_tour = np.array([0, 3, 2, 1, 4, 5])
    np.testing.assert_array_equal(swapped_tour, expected_tour)

    # Reverse segment from index 2 to 5 (inclusive)
    # The segment is [2, 3, 4, 5]. Reversed is [5, 4, 3, 2]
    swapped_tour_2 = apply_2_opt_swap(original_tour.copy(), 2, 5)
    expected_tour_2 = np.array([0, 1, 5, 4, 3, 2])
    np.testing.assert_array_equal(swapped_tour_2, expected_tour_2)


def test_generate_nn_tour(sample_dist_matrix):
    """
    Tests the greedy nearest-neighbor heuristic for generating an initial tour.
    """
    # With a deterministic algorithm, starting at node 0 should produce a
    # consistent tour. For the square, a greedy choice from 0 could be 1 or 3.
    # np.argmin will pick the first in case of a tie, so we expect 0 -> 1.
    # From 1, nearest is 2. From 2, nearest is 3.
    expected_tour = np.array([0, 1, 2, 3])
    
    nn_tour = generate_nn_tour(sample_dist_matrix, start_node=0)

    # Check that the tour is a full permutation
    assert len(nn_tour) == 4
    assert sorted(nn_tour) == [0, 1, 2, 3]

    np.testing.assert_array_equal(nn_tour, expected_tour)


def test_build_action_pairs():
    """Tests the generation of valid (i, j) pairs for 2-opt swaps."""
    # This import will fail until we create it
    from tsp_rl_gym.utils.tsp_ops import build_action_pairs

    # For N=5 cities, and max_span=3
    # Valid pairs (i < j):
    # (0, 1), (0, 2), (0, 3)
    # (1, 2), (1, 3), (1, 4)
    # (2, 3), (2, 4)
    # (3, 4)
    # Total = 3 + 3 + 2 + 1 = 9
    pairs = build_action_pairs(num_cities=5, max_span=3)
    assert len(pairs) == 9
    assert (0, 3) in pairs
    assert (0, 4) not in pairs # Exceeds max_span
    assert (2, 4) in pairs

def test_canonicalize_tour():
    """Tests that a tour is correctly rotated to start with city 0."""
    from tsp_rl_gym.utils.tsp_ops import canonicalize_tour

    # A tour that doesn't start with 0
    tour = np.array([3, 4, 0, 1, 2])
    canonical_tour = canonicalize_tour(tour)
    expected_tour = np.array([0, 1, 2, 3, 4])
    np.testing.assert_array_equal(canonical_tour, expected_tour)

    # A tour that already starts with 0
    tour_2 = np.array([0, 2, 4, 1, 3])
    canonical_tour_2 = canonicalize_tour(tour_2)
    np.testing.assert_array_equal(canonical_tour_2, tour_2)