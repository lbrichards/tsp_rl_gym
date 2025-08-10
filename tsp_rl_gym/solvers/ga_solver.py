"""
This module contains the main Genetic Algorithm solver class.
"""
import numpy as np
from tqdm import tqdm

from tsp_rl_gym.envs.core_scorer import CoreScorer
from .ga_operators import selection_tournament, crossover_ox1, mutation_swap

class GASolver:
    """
    A canonical Genetic Algorithm for solving the Traveling Salesperson Problem.
    """
    def __init__(self, scorer: CoreScorer, population_size: int, n_generations: int,
                 tournament_size: int = 5, mutation_rate: float = 0.1, seed: int = 0):
        self.scorer = scorer
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)
        
        self.num_cities = self.scorer.num_cities
        self.population = self._initialize_population()
        self.fitnesses = self._calculate_fitnesses()
        
        # Track initial population statistics
        self.initial_avg_fitness = np.mean(self.fitnesses)
        self.initial_best_fitness = np.min(self.fitnesses)
        
        self.best_tour = None
        self.best_fitness = float('inf')
        self.history = []
        
        self._update_best()

    def _initialize_population(self) -> np.ndarray:
        """Creates the initial population of random tours."""
        population = np.zeros((self.population_size, self.num_cities), dtype=int)
        for i in range(self.population_size):
            population[i] = self.rng.permutation(self.num_cities)
        return population

    def _calculate_fitnesses(self) -> np.ndarray:
        """Calculates the fitness (tour length) for the entire population."""
        return np.array([self.scorer.length(tour) for tour in self.population])

    def _update_best(self):
        """Finds the best individual in the current population and updates history."""
        best_idx = np.argmin(self.fitnesses)
        current_best_fitness = self.fitnesses[best_idx]
        
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_tour = self.population[best_idx]
        
        self.history.append(self.best_fitness)

    def run(self):
        """
        Executes the main loop of the Genetic Algorithm.
        """
        print("Running Genetic Algorithm...")
        for _ in tqdm(range(self.n_generations), desc="Generations"):
            next_population = np.zeros_like(self.population)
            
            # Elitism: Keep the best individual from the current population
            best_idx = np.argmin(self.fitnesses)
            next_population[0] = self.population[best_idx]
            
            # Generate the rest of the new population
            for i in range(1, self.population_size):
                # Selection
                competitor_indices = self.rng.choice(self.population_size, self.tournament_size, replace=False)
                parent1_idx = selection_tournament(self.fitnesses, competitor_indices)
                
                competitor_indices = self.rng.choice(self.population_size, self.tournament_size, replace=False)
                parent2_idx = selection_tournament(self.fitnesses, competitor_indices)
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Crossover
                crossover_points = sorted(self.rng.choice(self.num_cities, 2, replace=False))
                child = crossover_ox1(parent1, parent2, crossover_points[0], crossover_points[1])
                
                # Mutation
                if self.rng.random() < self.mutation_rate:
                    mutation_points = self.rng.choice(self.num_cities, 2, replace=False)
                    child = mutation_swap(child, mutation_points[0], mutation_points[1])
                
                next_population[i] = child
            
            self.population = next_population
            self.fitnesses = self._calculate_fitnesses()
            self._update_best()
        
        print(f"GA finished. Best tour length: {self.best_fitness:.2f}")