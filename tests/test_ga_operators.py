import numpy as np
import pytest
from pathlib import Path

from tsp_rl_gym.utils.data_loader import load_tsplib_instance
from tsp_rl_gym.envs.core_scorer import CoreScorer

# These imports will fail until we create the implementation file
from tsp_rl_gym.solvers.ga_operators import (
    selection_tournament,
    crossover_ox1,
    mutation_swap,
)

@pytest.fixture(scope="module")
def scorer():
    """A single scorer instance for the entire test module, loaded from a real file."""
    file_path = Path(__file__).parent.parent / "tsp_rl_gym" / "data" / "tsplib" / "ulysses16.tsp"
    instance = load_tsplib_instance(str(file_path))
    return CoreScorer(coords=instance["coords"])

@pytest.fixture
def sample_population(scorer):
    """A sample population of 4 tours for the ulysses16 problem."""
    rng = np.random.default_rng(seed=42)
    population = np.array([rng.permutation(scorer.num_cities) for _ in range(4)])
    return population

@pytest.fixture
def sample_fitnesses(sample_population, scorer):
    """Calculates the fitness (length) for each individual in the population."""
    return np.array([scorer.length(tour) for tour in sample_population])


def test_selection_tournament(sample_fitnesses):
    """
    Tests tournament selection. It should always select the individual with
    the better (lower) fitness score from a given set of competitors.
    """
    # Tournament between individuals at index 0 and 1
    competitor_indices = np.array([0, 1])
    # Find the expected winner by comparing their fitnesses directly
    expected_winner_idx = competitor_indices[np.argmin(sample_fitnesses[competitor_indices])]
    
    winner_idx = selection_tournament(sample_fitnesses, competitor_indices)
    assert winner_idx == expected_winner_idx


def test_crossover_ox1():
    """
    Tests the Ordered Crossover (OX1) operator.
    """
    parent1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    parent2 = np.array([5, 3, 1, 0, 6, 7, 4, 2])
    
    # Crossover segment from index 2 to 4
    child = crossover_ox1(parent1, parent2, 2, 4)
    
    # The segment [2, 3, 4] is copied from parent1.
    # The rest is filled from parent2, skipping duplicates from the segment.
    # P2 order (wrapping): [4, 2, 5, 3, 1, 0, 6, 7]
    # Filtered P2 (removing 2,3,4): [5, 1, 0, 6, 7]
    # Child should be: [5, 1, |2, 3, 4|, 0, 6, 7]
    expected_child = np.array([5, 1, 2, 3, 4, 0, 6, 7])
    np.testing.assert_array_equal(child, expected_child)
    # The child must be a valid permutation
    assert len(np.unique(child)) == len(parent1)


def test_mutation_swap():
    """
    Tests the simple swap mutation operator.
    """
    tour = np.array([0, 1, 2, 3, 4, 5])
    
    # Swap elements at index 1 and 4
    mutated_tour = mutation_swap(tour, 1, 4)
    
    expected_tour = np.array([0, 4, 2, 3, 1, 5])
    np.testing.assert_array_equal(mutated_tour, expected_tour)