from typing import Any
import torch
import gymnasium as gym
import numpy as np
from numpy import floating
from ray.rllib.policy.sample_batch import SampleBatch


def calculate_fitness_walker2d(env: gym.Env, policy, num_episodes: int = 3) -> floating[Any]:
    # Calculates the fitness score for a given policy in the Walker2d environment.
    all_episode_speeds = []

    for _ in range(num_episodes):
        observation, info = env.reset()
        terminated, truncated = False, False
        x_velocities = []

        while not terminated and not truncated:
            # A1. Convert numpy array to a torch tensor
            obs_tensor = torch.from_numpy(observation).float()

            # A2. Add a batch dimension (B=1). Shape (4,) -> (1, 4)
            obs_tensor = obs_tensor.unsqueeze(0)

            # A3. Create the input batch dictionary
            batch = {SampleBatch.OBS: obs_tensor}

            output = policy.forward_inference(batch)

            # B1. Extract the actions tensor from the output dict
            action_log_std_tensor = output['action_dist_inputs']
            means, log_stds = torch.chunk(action_log_std_tensor, 2, dim=-1)

            # B2. Remove the batch dimension. Shape (1, 1) or (1,) -> (1,) or ()
            action_tensor = means.squeeze(0)

            # B3. Move tensor to CPU and convert to a numpy array
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

            # The requested metric: Average Speed
            episode_average_speed = episode_total_distance / episode_duration
            all_episode_speeds.append(episode_average_speed)
        else:
            # If the agent falls on the first step, fitness is 0
            all_episode_speeds.append(0.0)

    overall_average_speed = np.mean(all_episode_speeds)
    return overall_average_speed
