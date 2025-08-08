"""
This module contains the CoreScorer, a pure evaluation engine for TSP tours.
"""
import numpy as np

from tsp_rl_gym.utils.tsp_ops import (
    calculate_distance_matrix,
    calculate_tour_length,
    generate_nn_tour,
)

class CoreScorer:
    """
    A pure, state-less evaluation engine for TSP tours.

    This class handles the core logic of scoring TSP solutions without any
    environment-specific or agent-specific implementation details. It is
    responsible for calculating distances, tour lengths, and rewards based
    on a normalized improvement metric.
    """

    def __init__(self, coords: np.ndarray, baseline_tour_type: str = "nn"):
        """
        Initializes the scorer with the problem's coordinates.

        Args:
            coords: A NumPy array of shape (N, 2) with city coordinates.
            baseline_tour_type: The heuristic to use for the baseline tour.
                                Currently only "nn" (nearest-neighbor) is supported.
        """
        self.coords = np.array(coords, dtype=np.float32)
        self.num_cities = self.coords.shape[0]

        # Pre-calculate the full distance matrix
        self.D = calculate_distance_matrix(self.coords)

        # Generate the baseline tour (L0) using the specified heuristic
        if baseline_tour_type == "nn":
            # Start NN tour from node 0 for reproducibility
            self.tour0 = generate_nn_tour(self.D, start_node=0)
        else:
            # Fallback to a random tour if heuristic is unknown
            rng = np.random.default_rng(0) # Fixed seed for reproducibility
            self.tour0 = rng.permutation(self.num_cities)

        self.L0 = self.length(self.tour0)

    def length(self, tour: np.ndarray) -> float:
        """Calculates the total length of a given tour."""
        return calculate_tour_length(tour, self.D)

    def delta(self, prev_tour: np.ndarray, new_tour: np.ndarray) -> float:
        """
        Calculates the normalized improvement (reward) between two tours.

        The reward is the difference in length, normalized by the baseline
        tour length (L0). A positive value indicates an improvement.

        Args:
            prev_tour: The tour before the action was taken.
            new_tour: The tour after the action was taken.

        Returns:
            The reward value as a float.
        """
        L_prev = self.length(prev_tour)
        L_new = self.length(new_tour)
        return (L_prev - L_new) / self.L0