#!/usr/bin/env python
"""
GA Verification Script for Sprint 1
Runs the Genetic Algorithm on a 20-city problem to verify core functionality.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tsp_rl_gym.envs.core_scorer import CoreScorer
from tsp_rl_gym.solvers.ga_solver import GASolver
from tsp_rl_gym.utils.visualizer import plot_tour, plot_convergence
from tsp_rl_gym.utils.logging import create_experiment_directory

def generate_random_tsp_instance(n_cities: int, seed: int) -> np.ndarray:
    """Generate a random TSP instance with n_cities."""
    np.random.seed(seed)
    return np.random.rand(n_cities, 2)

def main():
    """Main function to run GA verification."""
    # Configuration matching the verification requirements
    n_cities = 20
    seed = 42
    generations = 500
    population_size = 100
    tournament_size = 5
    mutation_rate = 0.1
    instance_name = f"random{n_cities}"
    
    # Use new standardized directory structure
    output_dir = Path(create_experiment_directory("GA", instance_name, seed))
    
    # Save config for reference
    config = {
        "n_cities": n_cities,
        "seed": seed,
        "generations": generations,
        "population_size": population_size,
        "tournament_size": tournament_size,
        "mutation_rate": mutation_rate,
        "instance_type": "random"
    }
    
    import json
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Generate a 20-city problem (since we don't have pr76.tsp)
    print(f"Generating random {n_cities}-city TSP instance with seed {seed}...")
    coords = generate_random_tsp_instance(n_cities, seed)
    
    # Initialize the Scorer
    scorer = CoreScorer(coords=coords)
    print(f"Initial tour length (random tour): {scorer.L0:.4f}")
    
    # Initialize and run the GA Solver
    print(f"\nRunning GA with:")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Seed: {seed}")
    
    solver = GASolver(
        scorer=scorer,
        population_size=population_size,
        n_generations=generations,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        seed=seed
    )
    solver.run()
    
    # Print results
    print(f"\n=== GA Results ===")
    print(f"Initial tour length: {scorer.L0:.4f}")
    print(f"Final tour length: {solver.best_fitness:.4f}")
    improvement = (scorer.L0 - solver.best_fitness) / scorer.L0 * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot the best tour
    plot_tour(ax1, coords, solver.best_tour, 
              title=f"Best Tour ({n_cities} cities) - Length: {solver.best_fitness:.4f}")
    
    # Plot convergence
    plot_convergence(ax2, solver.history, title="GA Convergence")
    
    fig.suptitle(f"GA Verification - {n_cities} Cities, Seed {seed}", fontsize=16)
    fig.tight_layout()
    
    # Save the plot
    output_file = output_dir / "solution_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Solution plot saved to: {output_file}")
    
    # Save text summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"GA Verification Results\n")
        f.write(f"=======================\n")
        f.write(f"Problem: {n_cities} cities (random instance)\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Population Size: {population_size}\n")
        f.write(f"Generations: {generations}\n")
        f.write(f"Initial tour length: {scorer.L0:.4f}\n")
        f.write(f"Final tour length: {solver.best_fitness:.4f}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")
        f.write(f"\nBest tour found:\n")
        f.write(f"{solver.best_tour.tolist()}\n")
    print(f"Summary saved to: {summary_file}")
    
    print("\n✅ GA verification completed successfully!")
    return solver.best_fitness

if __name__ == "__main__":
    final_length = main()