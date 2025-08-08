import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tsp_rl_gym.envs.core_scorer import CoreScorer
# This import will fail until we create the environment class
from tsp_rl_gym.envs.operator_env import OperatorEnv

@pytest.fixture
def scorer():
    """Provides a consistent CoreScorer instance for a 10-city problem."""
    coords = np.random.rand(10, 2)
    return CoreScorer(coords=coords)

@pytest.fixture
def env(scorer):
    """Initializes a standard OperatorEnv for tests."""
    return OperatorEnv(scorer=scorer, max_span=5, max_steps=20, patience=5)


def test_env_initialization(env):
    """Tests that the environment's spaces are configured correctly."""
    assert env.action_space.n > 0
    assert "coords" in env.observation_space.spaces
    assert "tour" in env.observation_space.spaces
    assert env.observation_space["coords"].shape == (10, 2)
    assert env.observation_space["tour"].shape == (10,)

def test_env_reset(env):
    """Tests that reset() returns a valid observation and info dict."""
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert "L" in info
    assert info["L"] > 0
    # Test that the internal tour starts at city 0 for canonical representation
    assert env.tour[0] == 0

def test_step_logic_and_reward(env):
    """
    Tests that a step correctly applies an operator and returns a valid transition.
    """
    obs, info = env.reset()
    L_initial = info["L"]

    # Force an action that we know is a 2-opt swap (e.g., action 0)
    # The exact change depends on the initial tour, but the mechanics should be consistent.
    new_obs, reward, terminated, truncated, new_info = env.step(0)

    L_new = new_info["L"]

    # Reward should equal the normalized change in length from the CoreScorer
    expected_reward = (L_initial - L_new) / env.scorer.L0
    assert np.isclose(reward, expected_reward)
    
    # New observation must be valid
    assert env.observation_space.contains(new_obs)
    assert not terminated  # Unlikely to terminate on the first step
    assert not truncated

def test_patience_termination(scorer):
    """Tests that the episode terminates when patience runs out."""
    # Use a tiny patience to force termination
    env = OperatorEnv(scorer=scorer, max_steps=50, patience=2)
    env.reset()

    # Find an action that makes the tour worse (negative reward) to decrement patience
    action = 0
    for i in range(100): # Limit search to avoid infinite loop
        # Apply a 2-opt swap
        new_tour = env._apply_operator(env.tour, *env.action_pairs[i])
        if env.scorer.length(new_tour) > env.scorer.length(env.tour):
            action = i
            break
    
    # Two bad steps should trigger termination
    _, _, terminated, _, _ = env.step(action)
    assert not terminated
    _, _, terminated, _, _ = env.step(action)
    assert terminated

def test_gymnasium_compatibility(env):
    """
    Uses the official Gymnasium checker to validate the environment's API.
    """
    check_env(env, skip_render_check=True)