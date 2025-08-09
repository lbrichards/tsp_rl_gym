import numpy as np
import pytest
from tsp_rl_gym.envs.core_scorer import CoreScorer

def test_immediate_reversal_is_illegal():
    """
    Tests that after taking an action (i, j), that same action is
    masked as illegal on the immediately following step.
    """
    # This will fail because OperatorEnv doesn't exist yet
    try:
        from tsp_rl_gym.envs.operator_env import OperatorEnv
    except ImportError:
        pytest.fail("OperatorEnv not found. Environment needs to be created first.")
    
    # Create a simple test environment
    coords = np.random.rand(10, 2)
    scorer = CoreScorer(coords=coords)
    env = OperatorEnv(scorer=scorer, max_span=5, max_steps=20, patience=5)
    
    obs, _ = env.reset()
    action_space_size = env.action_space.n

    # Find a valid, non-trivial action
    action_index = -1
    original_action = None
    action_mask = obs.get("action_mask")
    
    if action_mask is None:
        pytest.fail("Environment does not provide action_mask in observation")
    
    for i in range(action_space_size - 1, -1, -1):
        if action_mask[i]:
            action_index = i
            # Assuming the environment has a way to convert action index to (i, j) pair
            if hasattr(env, 'action_pairs'):
                original_action = env.action_pairs[action_index]
            elif hasattr(env, 'action_to_pair'):
                original_action = env.action_to_pair(action_index)
            else:
                original_action = (action_index, None)  # Fallback
            break
    
    assert original_action is not None, "Could not find any valid action to test."

    # Take the action
    new_obs, reward, terminated, truncated, info = env.step(action_index)

    # On the next step, the same action should be masked
    next_action_mask = new_obs.get("action_mask")
    assert next_action_mask is not None, "Action mask not found in observation after step"
    
    assert not next_action_mask[action_index], \
        f"Action {original_action} (index {action_index}) was not masked after being taken."