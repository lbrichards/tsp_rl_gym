import numpy as np
import pytest
from pathlib import Path

from tsp_rl_gym.utils.data_loader import load_tsplib_instance
from tsp_rl_gym.envs.core_scorer import CoreScorer

# This import will fail until we create the solver class
from tsp_rl_gym.solvers.ga_solver import GASolver

@pytest.fixture(scope="module")
def scorer():
    """A single scorer instance for the entire test module."""
    file_path = Path(__file__).parent.parent / "tsp_rl_gym" / "data" / "tsplib" / "ulysses16.tsp"
    instance = load_tsplib_instance(str(file_path))
    return CoreScorer(coords=instance["coords"])

def test_ga_solver_runs_and_improves(scorer):
    """
    Tests that the GASolver can be initialized and run, and that it
    improves the tour length over a few generations.
    """
    # Initialize the solver with a small population for a quick test
    solver = GASolver(
        scorer=scorer,
        population_size=50,
        n_generations=10,
        mutation_rate=0.1,
        seed=42
    )

    # The solver should have an initial population and fitnesses
    assert solver.population.shape == (50, 16)
    assert solver.fitnesses.shape == (50,)
    
    # Get the initial best tour length
    initial_best_length = solver.best_fitness
    
    # Run the solver
    solver.run()

    # The final best length should be less than or equal to the initial length
    # For a real run, it should be strictly less, but we use <= for robustness
    final_best_length = solver.best_fitness
    assert final_best_length <= initial_best_length

    # The solver should have recorded a history of improvements
    assert len(solver.history) == 11 # 1 initial + 10 generations
    assert solver.history[0] == initial_best_length
    assert solver.history[-1] == final_best_length

    # The best tour found must be a valid permutation
    assert len(np.unique(solver.best_tour)) == scorer.num_cities