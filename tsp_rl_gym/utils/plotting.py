"""
Reusable plotting utilities for TSP solutions.
This module provides a simple interface for saving solution plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, List
import os
import glob

from tsp_rl_gym.utils.visualizer import plot_tour, plot_convergence

# For parsing tensorboard event files
try:
    from tbparse import SummaryReader
    TBPARSE_AVAILABLE = True
except ImportError:
    print("Warning: tbparse not available. Learning curves will use placeholders.")
    TBPARSE_AVAILABLE = False


def plot_solution(
    coords: np.ndarray, 
    tour_indices: np.ndarray, 
    output_path: Union[str, Path],
    title: Optional[str] = None,
    tour_length: Optional[float] = None,
    convergence_history: Optional[List[float]] = None
) -> None:
    """
    Create and save a plot of a TSP solution.
    
    Args:
        coords: Coordinates of cities, shape (N, 2)
        tour_indices: Order of cities in the tour, shape (N,)
        output_path: Path where the plot will be saved
        title: Optional title for the plot
        tour_length: Optional tour length to display in title
        convergence_history: Optional convergence history to plot
    """
    # Convert string path to Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine layout based on whether we have convergence history
    if convergence_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create title with tour length if provided
    if title is None:
        if tour_length is not None:
            title = f"TSP Solution - Length: {tour_length:.2f}"
        else:
            title = "TSP Solution"
    elif tour_length is not None:
        title = f"{title} - Length: {tour_length:.2f}"
    
    # Plot the tour
    plot_tour(ax1, coords, tour_indices, title=title)
    
    # Plot convergence history if provided
    if convergence_history is not None:
        plot_convergence(ax2, convergence_history, title="Training Convergence")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to: {output_path}")


def plot_comparison(
    coords: np.ndarray,
    tours: dict,
    output_path: Union[str, Path],
    title: str = "TSP Solutions Comparison"
) -> None:
    """
    Create and save a comparison plot of multiple TSP solutions.
    
    Args:
        coords: Coordinates of cities, shape (N, 2)
        tours: Dictionary mapping solver names to (tour, length) tuples
        output_path: Path where the plot will be saved
        title: Title for the overall plot
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_tours = len(tours)
    cols = min(3, n_tours)
    rows = (n_tours + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_tours == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (solver_name, (tour, length)) in enumerate(tours.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        plot_title = f"{solver_name}\nLength: {length:.2f}"
        plot_tour(ax, coords, tour, title=plot_title)
    
    # Hide unused subplots
    for idx in range(n_tours, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison plot saved to: {output_path}")


def plot_learning_curve(
    log_dir: Union[str, Path],
    output_path: Union[str, Path],
    title: str = "PPO Training Progress",
    monitor_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Parses binary TensorBoard event files and plots key training metrics.
    If monitor_path is provided, also plots episode rewards from monitor CSV files.
    
    Args:
        log_dir: Path to the tensorboard log directory
        output_path: Path where the plot will be saved
        title: Title for the plot
        monitor_path: Optional path to directory containing monitor CSV files
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not TBPARSE_AVAILABLE:
        print("Warning: tbparse not installed. Cannot parse TensorBoard logs.")
        _create_placeholder_plot(log_dir, output_path, title)
        return
    
    print(f"Generating learning curve from TensorBoard logs at: {log_dir}")
    
    try:
        # Use SummaryReader to parse the TensorBoard event files
        reader = SummaryReader(str(log_dir))
        df = reader.scalars
        
        if df is None or df.empty:
            print("Warning: No scalar data found in TensorBoard logs.")
            _create_placeholder_plot(log_dir, output_path, title)
            return
            
    except Exception as e:
        print(f"Error loading TensorBoard data: {e}")
        _create_placeholder_plot(log_dir, output_path, title)
        return
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Check if we have monitor CSV files for episode rewards
    monitor_data = None
    if monitor_path:
        monitor_path = Path(monitor_path) if isinstance(monitor_path, str) else monitor_path
        monitor_files = list(monitor_path.glob("*.monitor.csv"))
        if monitor_files:
            # Load and combine all monitor CSV files
            monitor_dfs = []
            for mfile in monitor_files:
                try:
                    # Monitor CSV has 2 header rows - skip first, use second
                    mdf = pd.read_csv(mfile, skiprows=1)
                    monitor_dfs.append(mdf)
                except Exception as e:
                    print(f"Could not read monitor file {mfile}: {e}")
            
            if monitor_dfs:
                monitor_data = pd.concat(monitor_dfs, ignore_index=True)
                # Calculate cumulative timesteps for x-axis
                if 'l' in monitor_data.columns:  # 'l' is episode length
                    monitor_data['timesteps'] = monitor_data['l'].cumsum()
                print(f"Loaded episode rewards from {len(monitor_files)} monitor file(s)")
    
    # Define metrics to plot with their TensorBoard tags
    metrics = [
        ('rollout/ep_rew_mean', 'Mean Episode Reward', axes[0, 0], 'green'),
        ('train/value_loss', 'Value Loss', axes[0, 1], 'red'),
        ('train/entropy_loss', 'Entropy Loss', axes[1, 0], 'purple'),
        ('train/policy_gradient_loss', 'Policy Gradient Loss', axes[1, 1], 'orange')
    ]
    
    plots_created = 0
    
    for metric_tag, metric_name, ax, color in metrics:
        data_plotted = False
        
        # For episode reward, prefer monitor data if available
        if metric_tag == 'rollout/ep_rew_mean' and monitor_data is not None and 'r' in monitor_data.columns:
            # Plot episode rewards from monitor
            if 'timesteps' in monitor_data.columns:
                ax.plot(monitor_data['timesteps'], monitor_data['r'], 
                       color=color, linewidth=2, alpha=0.8, label='Episode Rewards')
            else:
                ax.plot(range(len(monitor_data)), monitor_data['r'], 
                       color=color, linewidth=2, alpha=0.8, label='Episode Rewards')
            
            # Add rolling mean for smoothing
            if len(monitor_data) > 10:
                rolling_mean = monitor_data['r'].rolling(window=10, min_periods=1).mean()
                if 'timesteps' in monitor_data.columns:
                    ax.plot(monitor_data['timesteps'], rolling_mean, 
                           color='darkgreen', linewidth=2, alpha=0.9, label='Rolling Mean (10 eps)')
                else:
                    ax.plot(range(len(monitor_data)), rolling_mean, 
                           color='darkgreen', linewidth=2, alpha=0.9, label='Rolling Mean (10 eps)')
            
            ax.set_title('Mean Episode Reward')
            ax.set_xlabel('Training Timesteps' if 'timesteps' in monitor_data.columns else 'Episodes')
            ax.set_ylabel('Episode Reward')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
            data_plotted = True
            plots_created += 1
        
        # For other metrics or if no monitor data, use TensorBoard data
        if not data_plotted:
            # Filter data for this metric
            metric_df = df[df['tag'] == metric_tag].copy()
            
            if not metric_df.empty:
                # Sort by step to ensure proper plotting
                metric_df = metric_df.sort_values('step')
                
                # Plot the metric
                ax.plot(metric_df['step'], metric_df['value'], color=color, linewidth=2, alpha=0.8)
                ax.set_title(metric_name)
                ax.set_xlabel('Training Timesteps')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
                
                # Add log scale for loss metrics if values span multiple orders of magnitude
                if 'loss' in metric_name.lower() and len(metric_df) > 0:
                    values = metric_df['value'].dropna()
                    if len(values) > 0 and values.max() / values.min() > 100:
                        ax.set_yscale('log')
                
                plots_created += 1
            else:
                # No data for this metric
                ax.text(0.5, 0.5, f'{metric_name}\n(No data available)', 
                       ha='center', va='center', fontsize=12, alpha=0.5)
                ax.set_title(metric_name)
                ax.set_xlabel('Training Timesteps')
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
    
    if plots_created == 0:
        print("Warning: No recognized metrics found in TensorBoard logs.")
        _create_placeholder_plot(log_dir, output_path, title)
        return
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Real learning curve plot saved to: {output_path}")
    print(f"Successfully plotted {plots_created} metrics from TensorBoard logs")


def _create_placeholder_plot(log_dir: Path, output_path: Path, title: str) -> None:
    """Create a placeholder plot when real data is unavailable."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    
    metrics = [
        ('Mean Episode Reward', axes[0, 0]),
        ('Value Loss', axes[0, 1]),
        ('Entropy Loss', axes[1, 0]),
        ('Policy Gradient Loss', axes[1, 1])
    ]
    
    for metric_name, ax in metrics:
        ax.text(0.5, 0.5, 
               f'{metric_name}\n\nTraining metrics saved in binary format.\n\n'
               f'View with:\ntensorboard --logdir {log_dir}', 
               ha='center', va='center', fontsize=10)
        ax.set_title(metric_name)
        ax.set_xlabel('Training Timesteps')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Placeholder learning curve saved to: {output_path}")