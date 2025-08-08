"""
This module provides visualization utilities for TSP solutions and solver performance.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List

def plot_tour(ax: Axes, coords: np.ndarray, tour: np.ndarray, title: str = "TSP Tour") -> Axes:
    """
    Plots a TSP tour on a given matplotlib Axes object.

    Args:
        ax: The matplotlib Axes to plot on.
        coords: A NumPy array of shape (N, 2) with city coordinates.
        tour: A 1D NumPy array representing the permutation of city indices.
        title: The title for the plot.

    Returns:
        The matplotlib Axes object with the plot.
    """
    # Create a closed loop tour by appending the start node to the end
    tour_coords = coords[tour]
    closed_tour_coords = np.vstack([tour_coords, tour_coords[0]])

    # Plot the tour path
    ax.plot(closed_tour_coords[:, 0], closed_tour_coords[:, 1], 'b-')

    # Plot the cities as points
    ax.scatter(coords[:, 0], coords[:, 1], c='r', s=50, zorder=3)

    # Annotate the cities with their indices
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=12, ha='right')

    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal', adjustable='box')
    return ax

def plot_convergence(ax: Axes, history: List[float], title: str = "GA Convergence") -> Axes:
    """
    Plots the convergence history of a solver.

    Args:
        ax: The matplotlib Axes to plot on.
        history: A list of the best fitness values over generations/iterations.
        title: The title for the plot.

    Returns:
        The matplotlib Axes object with the plot.
    """
    ax.plot(history)
    ax.set_title(title)
    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Best Tour Length")
    ax.grid(True)
    return ax