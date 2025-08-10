"""
Basic Gymnasium environment for TSP with 2-opt operations and reversal guard.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from tsp_rl_gym.envs.core_scorer import CoreScorer
from tsp_rl_gym.utils.tsp_ops import apply_2_opt_swap, build_action_pairs, canonicalize_tour


class OperatorEnv(gym.Env):
    """
    A Gymnasium environment for TSP that allows 2-opt operations.
    
    Features:
    - Guard against immediate action reversal to prevent oscillation
    - Potential-Based Reward Shaping (PBRS) for stable RL training
    - Step penalty to encourage efficient solutions
    
    Reward formula: r = F(s') - F(s) - step_penalty
    where F(s) = -tour_length (potential function)
    """
    metadata = {"render_modes": []}

    def __init__(self, scorer: CoreScorer, max_span: int, max_steps: int, patience: int, step_penalty: float = 1e-4):
        super().__init__()
        self.scorer = scorer
        self.coords = scorer.coords
        self.num_cities = scorer.num_cities

        # Environment parameters
        self.max_span = min(max_span, self.num_cities - 1)
        self.max_steps = max_steps
        self.patience_limit = patience
        self.step_penalty = step_penalty  # Penalty per step to encourage efficiency

        # Pre-compute valid actions (2-opt swaps)
        self.action_pairs = build_action_pairs(self.num_cities, self.max_span)
        self.action_space = spaces.Discrete(len(self.action_pairs))

        # Define observation space with action mask
        self.observation_space = spaces.Dict({
            "coords": spaces.Box(low=0, high=1, shape=(self.num_cities, 2), dtype=np.float32),
            "tour": spaces.Box(low=0, high=self.num_cities - 1, shape=(self.num_cities,), dtype=np.int64),
            "action_mask": spaces.Box(low=0, high=1, shape=(len(self.action_pairs),), dtype=np.uint8)
        })

        # State variables
        self.tour = None
        self.L_current = None
        self.L_best = None
        self.step_count = 0
        self.patience = 0
        self.last_action_index = -1  # Track last action for reversal guard

    def _get_action_mask(self):
        """
        Generate action mask, disabling the last action taken to prevent immediate reversal.
        """
        # Start with all actions enabled
        mask = np.ones(len(self.action_pairs), dtype=np.uint8)
        
        # Disable the last action if one was taken
        if self.last_action_index >= 0:
            mask[self.last_action_index] = 0
            
        return mask

    def _get_obs(self):
        """Constructs the observation dictionary from the current state."""
        return {
            "coords": self.coords.astype(np.float32),
            "tour": self.tour.astype(np.int64),
            "action_mask": self._get_action_mask()
        }
    
    def _apply_operator(self, tour, i, j):
        """Apply 2-opt swap operation."""
        return apply_2_opt_swap(tour, i, j)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize tour
        self.tour = canonicalize_tour(self.scorer.tour0)
        self.L_current = self.scorer.length(self.tour)
        self.L_best = self.L_current
        
        # Reset counters
        self.step_count = 0
        self.patience = self.patience_limit
        self.last_action_index = -1  # Clear last action on reset
        
        info = {"L": self.L_current, "best": self.L_best}
        return self._get_obs(), info

    def step(self, action: int):
        # Validate action against mask
        mask = self._get_action_mask()
        if not mask[action]:
            raise ValueError(f"Action {action} is not valid (masked)")
        
        # Store initial tour length for reward calculation
        initial_length = self.L_current
        
        # Apply the 2-opt swap
        i, j = self.action_pairs[action]
        new_tour = self._apply_operator(self.tour, i, j)
        new_length = self.scorer.length(new_tour)
        
        # Calculate potential-based reward: F(s') - F(s) - step_penalty
        # where F(s) = -tour_length (negative because we minimize)
        phi_initial = -initial_length
        phi_new = -new_length
        reward = phi_new - phi_initial - self.step_penalty
        
        # Update state
        self.tour = new_tour
        self.L_current = new_length
        self.last_action_index = action  # Store the action for reversal guard
        
        # Update best and patience
        if self.L_current < self.L_best:
            self.L_best = self.L_current
            self.patience = self.patience_limit
        else:
            self.patience -= 1
        
        # Check termination
        self.step_count += 1
        terminated = (self.step_count >= self.max_steps) or (self.patience <= 0)
        truncated = False
        
        info = {"L": self.L_current, "best": self.L_best}
        
        return self._get_obs(), reward, terminated, truncated, info