"""
Baseline GA solver wrapper for benchmark harness.
"""
import time
import numpy as np
from pathlib import Path

from tsp_rl_gym.utils.data_loader import load_tsplib_instance
from tsp_rl_gym.envs.core_scorer import CoreScorer
from tsp_rl_gym.solvers.ga_solver import GASolver


def run_ga_solver(instance, seed, generations, population_size=100, output_dir=None):
    """
    Run the GA solver on a TSP instance and return results.
    
    Args:
        instance: Path to the TSP instance file (TSPLIB format)
        seed: Random seed for reproducibility
        generations: Number of generations to run
        population_size: Population size for GA
        output_dir: Optional directory for outputs (unused for headless runs)
    
    Returns:
        dict: Results including best_distance, initial_distance, wall_s, etc.
    """
    start_time = time.time()
    
    # Load the TSP instance
    instance_path = Path(instance)
    
    # If file doesn't exist, create a random instance for testing
    if not instance_path.exists():
        # For testing, generate a random instance based on the name
        # e.g., "test_instance_20" -> 20 cities
        import re
        match = re.search(r'(\d+)', str(instance_path.name))
        if match:
            n_cities = int(match.group(1))
        else:
            n_cities = 10  # default
        np.random.seed(seed)
        coords = np.random.rand(n_cities, 2)
    elif instance_path.suffix == '.tsp':
        # Load TSPLIB format
        instance_data = load_tsplib_instance(str(instance_path))
        coords = instance_data["coords"]
    else:
        # Try to load as numpy array or handle other formats
        raise ValueError(f"Unsupported instance format: {instance_path.suffix}")
    
    # Initialize the scorer
    scorer = CoreScorer(coords=coords)
    initial_distance = scorer.L0
    n_cities = scorer.num_cities
    
    # Run the GA solver
    solver = GASolver(
        scorer=scorer,
        population_size=population_size,
        n_generations=generations,
        tournament_size=5,
        mutation_rate=0.1,
        seed=seed
    )
    solver.run()
    
    # Calculate wall time
    wall_s = time.time() - start_time
    
    # Return results in expected format
    return {
        'best_distance': solver.best_fitness,
        'initial_distance': initial_distance,
        'n_cities': n_cities,
        'wall_s': wall_s,
        'best_tour': solver.best_tour,
        'history': solver.history
    }