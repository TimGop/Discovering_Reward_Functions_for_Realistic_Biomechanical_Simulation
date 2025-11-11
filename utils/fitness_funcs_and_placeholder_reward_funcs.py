from typing import Any
import torch
import gymnasium as gym
import numpy as np
from numpy import floating
from ray.rllib.policy.sample_batch import SampleBatch


def calculate_fitness_walker2d(env: gym.Env, policy, num_episodes: int = 3) -> floating[Any]:
    all_episode_speeds = []

    for _ in range(num_episodes):
        observation, info = env.reset()
        terminated, truncated = False, False
        x_velocities = []

        while not terminated and not truncated:
            obs_tensor = torch.from_numpy(observation).float()

            obs_tensor = obs_tensor.unsqueeze(0)

            batch = {SampleBatch.OBS: obs_tensor}

            output = policy.forward_inference(batch)

            action_log_std_tensor = output['action_dist_inputs']
            means, log_stds = torch.chunk(action_log_std_tensor, 2, dim=-1)

            # remove the batch dimension. Shape (1, 1) or (1,) -> (1,) or ()
            action_tensor = means.squeeze(0)

            action = action_tensor.cpu().numpy()

            observation, reward, terminated, truncated, info = env.step(action)

            x_velocity = info.get('x_velocity', None)
            if not x_velocity:
                print("Warning: Could not access MuJoCo data directly. Skipping velocity logging.")
                break  # Stop episode if we can't measure x_velocity
            x_velocities.append(x_velocity)

        if x_velocities:
            episode_total_distance = sum(x_velocities)
            episode_duration = len(x_velocities)

            episode_average_speed = episode_total_distance / episode_duration
            all_episode_speeds.append(episode_average_speed)
        else:
            # If the agent falls on the first step, fitness is 0
            all_episode_speeds.append(0.0)

    overall_average_speed = np.mean(all_episode_speeds)
    return overall_average_speed


def walker2d_original_reward(observation, action, original_reward, next_observation, terminated, truncated):
    # forward_vel = info.get('x_velocity', 0.0)
    info = {"base_reward": original_reward}
    return original_reward, info
