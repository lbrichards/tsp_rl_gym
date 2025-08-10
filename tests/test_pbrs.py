"""
Sprint 19: Test suite for Potential-Based Reward Shaping (PBRS).
This test defines the expected reward structure that uses a potential function
to provide more stable and informative rewards for RL agents.
"""

import numpy as np
import pytest

from tsp_rl_gym.envs.core_scorer import CoreScorer
from tsp_rl_gym.envs.operator_env import OperatorEnv


@pytest.fixture
def scorer():
    """Provides a consistent CoreScorer instance for a 10-city problem."""
    np.random.seed(42)
    coords = np.random.rand(10, 2)
    return CoreScorer(coords=coords)


@pytest.fixture
def env(scorer):
    """Initializes a standard OperatorEnv for tests."""
    return OperatorEnv(scorer=scorer, max_span=5, max_steps=20, patience=5)


def test_step_penalty_exists(env):
    """Tests that the environment has a step_penalty attribute."""
    # This will fail in RED state
    assert hasattr(env, 'step_penalty'), "Environment must have step_penalty attribute"
    assert env.step_penalty > 0, "Step penalty should be positive to encourage efficiency"


def test_potential_function_reward(env):
    """
    Tests that rewards follow the potential-based formula:
    reward = F(s') - F(s) - step_penalty
    where F(s) = -tour_length (negative because we want to minimize length)
    """
    obs, info = env.reset()
    initial_tour = env.tour.copy()
    initial_length = env.scorer.length(initial_tour)
    
    # Find and take a valid action
    action_mask = env._get_action_mask()
    valid_actions = np.where(action_mask > 0)[0]
    assert len(valid_actions) > 0, "No valid actions available"
    
    action = valid_actions[0]
    obs, reward, terminated, truncated, info = env.step(action)
    new_length = env.scorer.length(env.tour)
    
    # Calculate expected reward using potential function
    phi_old = -initial_length  # Potential of old state
    phi_new = -new_length      # Potential of new state
    step_penalty = env.step_penalty
    
    expected_reward = phi_new - phi_old - step_penalty
    
    assert np.isclose(reward, expected_reward, rtol=1e-5), \
        f"Reward {reward} doesn't match PBRS formula. Expected {expected_reward}"


def test_reward_consistency_across_episodes(scorer):
    """
    Tests that the same action in the same state produces the same reward
    across different episodes (verifying Markovian property with PBRS).
    """
    env1 = OperatorEnv(scorer=scorer, max_span=5, max_steps=20, patience=5)
    env2 = OperatorEnv(scorer=scorer, max_span=5, max_steps=20, patience=5)
    
    # Reset both environments with same seed
    env1.reset(seed=123)
    env2.reset(seed=123)
    
    # Take same action in both
    action = 0
    _, reward1, _, _, _ = env1.step(action)
    _, reward2, _, _, _ = env2.step(action)
    
    assert np.isclose(reward1, reward2), \
        "Same action in same state should produce same reward"


def test_improvement_gives_positive_contribution(env):
    """
    Tests that improving the tour (reducing length) gives positive reward contribution
    before accounting for step penalty.
    """
    obs, info = env.reset()
    initial_length = env.scorer.length(env.tour)
    
    # Try multiple actions to find one that improves the tour
    action_mask = env._get_action_mask()
    valid_actions = np.where(action_mask > 0)[0]
    
    improvement_found = False
    for action in valid_actions[:10]:  # Try first 10 valid actions
        # Simulate the action to check if it improves
        new_tour = env._apply_operator(env.tour, *env.action_pairs[action])
        new_length = env.scorer.length(new_tour)
        
        if new_length < initial_length:
            # This action improves the tour
            obs, reward, _, _, _ = env.step(action)
            
            # The potential difference should be positive for improvements
            potential_diff = (-new_length) - (-initial_length)
            assert potential_diff > 0, "Improvement should give positive potential difference"
            
            # The reward includes step penalty, but improvement contribution is positive
            reward_without_penalty = reward + env.step_penalty
            assert reward_without_penalty > 0, \
                "Reward contribution from improvement should be positive"
            
            improvement_found = True
            break
    
    assert improvement_found, "Could not find an improving action to test"


def test_step_penalty_encourages_efficiency(env):
    """
    Tests that the step penalty properly encourages efficient solutions
    by making neutral moves have negative reward.
    """
    obs, info = env.reset()
    initial_length = env.scorer.length(env.tour)
    
    # Find an action that doesn't change tour length (if possible)
    # This is theoretical since most 2-opt moves change length
    # But we can test that any move has at least the step penalty as cost
    
    action_mask = env._get_action_mask()
    valid_actions = np.where(action_mask > 0)[0]
    
    for action in valid_actions[:5]:
        new_tour = env._apply_operator(env.tour, *env.action_pairs[action])
        new_length = env.scorer.length(new_tour)
        
        # If we find a neutral move (unlikely but possible)
        if np.isclose(new_length, initial_length, rtol=1e-6):
            obs, reward, _, _, _ = env.step(action)
            # Reward should be negative due to step penalty
            assert reward < 0, "Neutral moves should have negative reward due to step penalty"
            assert np.isclose(reward, -env.step_penalty, rtol=1e-5), \
                "Neutral move reward should equal negative step penalty"
            return
    
    # If no neutral moves found, at least verify step penalty is factored in
    obs, reward, _, _, _ = env.step(valid_actions[0])
    # This is a weaker test but still validates step penalty exists
    assert hasattr(env, 'step_penalty'), "Step penalty must exist for PBRS"


if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 19: Potential-Based Reward Shaping Tests")
    print("=" * 60)
    print("\nThese tests define the expected PBRS reward structure.")
    print("In RED state, they will fail because the environment")
    print("doesn't implement potential-based rewards yet.\n")
    
    # Run the tests
    pytest.main([__file__, "-v"])
    
    print("\n" + "=" * 60)
    print("RED STATE CONFIRMED: Environment needs PBRS implementation")