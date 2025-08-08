import matplotlib.pyplot as plt
from pathlib import Path

from tsp_rl_gym.utils.data_loader import load_tsplib_instance
from tsp_rl_gym.envs.core_scorer import CoreScorer
from tsp_rl_gym.solvers.ga_solver import GASolver
from tsp_rl_gym.utils.visualizer import plot_tour, plot_convergence

def main():
    """
    Main function to run the GA solver and produce visualizations.
    """
    # 1. Load the TSP instance
    print("Loading TSP instance...")
    file_path = Path(__file__).parent.parent / "tsp_rl_gym" / "data" / "tsplib" / "ulysses16.tsp"
    instance = load_tsplib_instance(str(file_path))
    coords = instance["coords"]
    instance_name = instance.get("name", "tsp_instance")
    
    # 2. Initialize the Scorer
    print("Initializing scorer...")
    scorer = CoreScorer(coords=coords)

    # 3. Initialize and run the GA Solver
    solver = GASolver(
        scorer=scorer,
        population_size=100,
        n_generations=150,
        tournament_size=5,
        mutation_rate=0.1,
        seed=42
    )
    solver.run()

    # 4. Create visualizations
    print("Generating visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot the initial vs. best tour
    initial_tour_len = scorer.length(scorer.tour0)
    plot_tour(ax1, coords, solver.best_tour, title=f"Best Tour ({instance_name}) - Length: {solver.best_fitness:.2f}")
    
    # Plot the convergence history
    plot_convergence(ax2, solver.history, title="GA Convergence")

    fig.tight_layout()
    
    # 5. Save the output
    output_filename = f"outputs/{instance_name}_ga_solution.png"
    plt.savefig(output_filename)
    print(f"Solution plot saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()