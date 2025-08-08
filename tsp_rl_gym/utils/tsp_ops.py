"""
Core operations for the Traveling Salesperson Problem (TSP).

This module provides fundamental, highly-optimized functions for handling TSP data,
such as calculating distance matrices and performing tour manipulations.
"""

import numpy as np
from typing import Union

def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix for a given set of coordinates.

    Args:
        coords: A NumPy array of shape (N, 2) or (N, 3) representing the
                coordinates of N cities.

    Returns:
        A NumPy array of shape (N, N) where element (i, j) is the
        Euclidean distance between city i and city j.
    """
    return np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)


def calculate_tour_length(tour: np.ndarray, dist_matrix: np.ndarray) -> float:
    """
    Calculates the total length of a given tour.

    The tour is a sequence of city indices. The total length is the sum of
    distances between consecutive cities in the sequence, plus the distance
    from the last city back to the first.

    Args:
        tour: A 1D NumPy array of integer indices representing the tour.
        dist_matrix: The N x N matrix of distances between cities.

    Returns:
        The total length of the tour as a float.
    """
    # Get the coordinates in the order of the tour
    tour_indices = tour.astype(int)
    # Get the "next" city in the tour for each city, wrapping around at the end
    next_tour_indices = np.roll(tour_indices, -1)
    # Sum the distances between each city and the next one in the tour
    return dist_matrix[tour_indices, next_tour_indices].sum()


def apply_2_opt_swap(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Performs a 2-opt swap on a tour, reversing the segment between two indices.

    Args:
        tour: A 1D NumPy array of integer indices representing the tour.
        i: The starting index of the segment to reverse (inclusive).
        j: The ending index of the segment to reverse (inclusive).

    Returns:
        A new NumPy array representing the tour with the specified segment reversed.
    """
    # Ensure i is less than or equal to j for slicing
    start, end = min(i, j), max(i, j)

    # Create a copy to avoid modifying the original array in place
    new_tour = tour.copy()

    # Reverse the segment from start to end
    segment = new_tour[start : end + 1]
    new_tour[start : end + 1] = segment[::-1]

    return new_tour


def generate_nn_tour(dist_matrix: np.ndarray, start_node: int = 0) -> np.ndarray:
    """
    Generates a tour using the nearest-neighbor heuristic.

    Starting from a given node, the algorithm greedily selects the nearest
    unvisited node at each step until all nodes have been visited.

    Args:
        dist_matrix: The N x N matrix of distances between cities.
        start_node: The index of the city to start the tour from.

    Returns:
        A 1D NumPy array of integer indices representing the NN tour.
    """
    num_cities = dist_matrix.shape[0]
    tour = np.zeros(num_cities, dtype=int)
    visited = np.zeros(num_cities, dtype=bool)

    current_node = start_node
    tour[0] = current_node
    visited[current_node] = True

    for i in range(1, num_cities):
        # Set distances to already visited nodes to infinity
        distances = dist_matrix[current_node, :].copy()
        distances[visited] = np.inf
        
        # Find the nearest unvisited neighbor
        next_node = np.argmin(distances)
        
        tour[i] = next_node
        visited[next_node] = True
        current_node = next_node
        
    return tour


def build_action_pairs(num_cities: int, max_span: int) -> list[tuple[int, int]]:
    """
    Generates a list of valid (i, j) pairs for 2-opt swaps.

    The pairs are constrained by i < j and (j - i) <= max_span.

    Args:
        num_cities: The total number of cities (N).
        max_span: The maximum allowed span for a segment reversal.

    Returns:
        A list of tuples, where each tuple is a valid (i, j) action pair.
    """
    pairs = []
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            if j - i <= max_span:
                pairs.append((i, j))
    return pairs


def canonicalize_tour(tour: np.ndarray) -> np.ndarray:
    """
    Rotates a tour so that it begins with city 0.

    This is useful for creating a canonical representation of a tour to
    reduce symmetries in the observation space.

    Args:
        tour: A 1D NumPy array representing the tour.

    Returns:
        A new 1D NumPy array representing the rotated tour.
    """
    # Find the position of city 0 in the tour
    start_index = np.where(tour == 0)[0][0]
    
    # Roll the array to move city 0 to the front
    return np.roll(tour, -start_index)