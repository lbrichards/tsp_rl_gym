import matplotlib.pyplot as plt
from pathlib import Path

from tsp_rl_gym.utils.data_loader import load_tsplib_instance
from tsp_rl_gym.envs.core_scorer import CoreScorer
from tsp_rl_gym.solvers.ga_solver import GASolver
from tsp_rl_gym.utils.visualizer import plot_tour, plot_convergence
from tsp_rl_gym.utils.logging import create_experiment_directory

def main():
    """
    Main function to run the GA solver and produce structured, logged outputs.
    """
    # --- Configuration ---
    instance_name = "ulysses16"
    population_size = 100
    n_generations = 150
    tournament_size = 5
    mutation_rate = 0.1
    seed = 42

    # 1. Create a unique, timestamped directory for this experiment run
    log_dir = Path(create_experiment_directory("GA", instance_name, seed))
    
    # Save config for reference
    import json
    config = {
        "instance_name": instance_name,
        "population_size": population_size,
        "n_generations": n_generations,
        "tournament_size": tournament_size,
        "mutation_rate": mutation_rate,
        "seed": seed
    }
    with open(log_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)

    # 2. Load the TSP instance
    print(f"Loading TSP instance: {instance_name}...")
    file_path = Path(__file__).parent.parent / "tsp_rl_gym" / "data" / "tsplib" / f"{instance_name}.tsp"
    instance = load_tsplib_instance(str(file_path))
    coords = instance["coords"]
    
    # 3. Initialize the Scorer
    scorer = CoreScorer(coords=coords)

    # 4. Initialize and run the GA Solver
    solver = GASolver(
        scorer=scorer,
        population_size=population_size,
        n_generations=n_generations,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        seed=seed
    )
    solver.run()

    # 5. Create visualizations
    print("Generating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_tour(ax1, coords, solver.best_tour, title=f"Best Tour ({instance_name}) - Length: {solver.best_fitness:.2f}")
    plot_convergence(ax2, solver.history, title="GA Convergence")
    fig.tight_layout()
    
    # 6. Save the output plot to the unique log directory
    output_filename = log_dir / "ga_solution.png"
    plt.savefig(output_filename)
    print(f"Solution plot saved to {output_filename}")
    # plt.show() # Comment out to prevent blocking in automated runs

if __name__ == "__main__":
    main()