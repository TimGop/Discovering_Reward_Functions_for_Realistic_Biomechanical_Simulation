# In a new file, e.g., 'tensor_reward_fns.py'
import torch


def walker_base_reward(root_states, progress_buf, dof_vel, dof_effort, termination_penalty):
    """Calculates the base components of the reward shared by all policies."""
    # Base forward velocity reward from isaacgym humanoid example
    # use lin vel x
    forward_vel = root_states[:, 7]
    forward_reward = 1.0 * forward_vel

    # Energy penalty
    energy_penalty = torch.sum(torch.square(dof_effort), dim=-1) * 0.005

    # Dof velocity penalty
    dof_vel_penalty = torch.sum(torch.square(dof_vel), dim=-1) * 0.001

    # Termination penalty (applied when episodes reset)
    # The `termination_penalty` is a tensor of shape (num_envs,) with values for terminated envs.
    total_reward = forward_reward - energy_penalty - dof_vel_penalty + termination_penalty
    return total_reward


# NEW TENSOR-BASED REWARD FUNCTIONS
def walker_less_speed_tensor(base_reward, root_states, **kwargs):
    """Custom reward logic that doesn't add extra speed incentive."""
    # In this case, it just returns the base reward.
    # We could add other penalties or bonuses here if needed.
    return base_reward + 0.0 * root_states[:, 7]  # No extra bonus for forward velocity


def walker_more_speed_tensor(base_reward, root_states, **kwargs):
    """Custom reward logic that encourages more speed."""
    # Add an extra incentive for forward velocity (root_states[:, 7] is linear vel x)
    forward_vel = root_states[:, 7]
    return base_reward + 0.1 * forward_vel