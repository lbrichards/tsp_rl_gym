"""
This module contains standard operators for a Genetic Algorithm to solve the TSP.
"""
import numpy as np

def selection_tournament(fitnesses: np.ndarray, competitor_indices: np.ndarray) -> int:
    """
    Performs a tournament selection.

    From a given array of competitor indices, this function finds the one
    with the best (lowest) fitness value.

    Args:
        fitnesses: A 1D array of fitness values for the entire population.
        competitor_indices: A 1D array of indices representing the competitors
                           in this tournament.

    Returns:
        The index of the winning individual.
    """
    # Get the fitness values for the competitors
    competitor_fitnesses = fitnesses[competitor_indices]
    # Find the index of the minimum fitness *within the competitor array*
    winner_local_idx = np.argmin(competitor_fitnesses)
    # Return the global index from the original competitor list
    return competitor_indices[winner_local_idx]


def crossover_ox1(parent1: np.ndarray, parent2: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Performs Ordered Crossover (OX1).

    A segment from parent1 is copied to the child. The remaining values are
    filled from parent2 in the order they appear, skipping any values already
    present in the child from parent1's segment.

    Args:
        parent1: The first parent tour.
        parent2: The second parent tour.
        start: The starting index of the crossover segment (inclusive).
        end: The ending index of the crossover segment (inclusive).

    Returns:
        A new tour representing the child.
    """
    num_cities = len(parent1)
    child = np.full(num_cities, -1, dtype=int)

    # Step 1: Copy the segment from parent1 to the child
    child[start : end + 1] = parent1[start : end + 1]
    
    # Step 2: Get the values from parent2 that are not in the child segment
    p2_vals_to_add = [item for item in parent2 if item not in child]
    
    # Step 3: Fill the remaining slots in the child
    p2_idx = 0
    for i in range(num_cities):
        if child[i] == -1:
            child[i] = p2_vals_to_add[p2_idx]
            p2_idx += 1
            
    return child


def mutation_swap(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Performs a simple swap mutation on a tour.

    Args:
        tour: The tour to be mutated.
        i: The index of the first city to swap.
        j: The index of the second city to swap.

    Returns:
        The mutated tour.
    """
    mutated_tour = tour.copy()
    mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
    return mutated_tour