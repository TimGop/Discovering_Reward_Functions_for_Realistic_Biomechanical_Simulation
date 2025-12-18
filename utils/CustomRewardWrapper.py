import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Callable, Dict, Tuple
import re
import tempfile
import importlib.util
import sys
import torch
from pathlib import Path

EXPECTED_NAME = "custom_reward_fn"


class JustFitnessWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.x_velocities = []

    def reset(self, *, seed=None, options=None):
        observation_np, info = super().reset(seed=seed, options=options)
        self.x_velocities = []
        return observation_np, info

    def step(self, action):
        observation_np, reward_np, terminated_np, truncated_np, info = self.env.step(action)
        self.x_velocities.append(info.get('x_velocity', 0.0))  # x_vel for fitness
        is_done = terminated_np or truncated_np
        if is_done:
            episode_duration = len(self.x_velocities)
            if episode_duration > 0:
                velocity_sum = sum(self.x_velocities)
                fitness_score = np.float64(velocity_sum / 1000)  # full episode would have length 1000
            else:
                fitness_score = np.float64(-1.0)

            info["fitness_score"] = fitness_score
            info["episode_length"] = np.float64(episode_duration)

        return observation_np, reward_np, terminated_np, truncated_np, info


# creates a version of a given gym environment where we can add a custom reward function
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn: Callable, policy_id: str):
        super().__init__(env)
        self.custom_reward_fn = reward_fn
        self.policy_id = policy_id
        self.x_velocities = []
        self.reward_components = {}
        self.current_observation = None

    def reset(self, *, seed=None, options=None):
        observation_np, info = super().reset(seed=seed, options=options)
        self.x_velocities = []
        self.reward_components = {}
        self.current_observation = observation_np
        return observation_np, info

    def step(self, action):
        observation_np, _, terminated_np, truncated_np, info = self.env.step(action)

        observation = torch.as_tensor(self.current_observation, dtype=torch.float32)
        next_observation = torch.as_tensor(observation_np, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool)
        truncated = torch.as_tensor(truncated_np, dtype=torch.bool)

        new_reward, reward_components = self.custom_reward_fn(
            observation, action, next_observation, terminated, truncated
        )

        self.current_observation = observation_np

        # update reward components or create new list if first append
        for key, value in reward_components.items():
            self.reward_components.setdefault(key, []).append(value)
        self.x_velocities.append(info.get('x_velocity', 0.0))  # x_vel for fitness

        if torch.isnan(new_reward).any() or torch.isinf(new_reward).any():
            new_reward = torch.zeros_like(new_reward)

        is_done = terminated_np or truncated_np
        if is_done:
            episode_duration = len(self.x_velocities)
            if episode_duration > 0:
                velocity_sum = sum(self.x_velocities)
                fitness_score = np.float64(velocity_sum / 1000)  # full episode would have length 1000
            else:
                fitness_score = np.float64(-1.0)

            info["fitness_score"] = fitness_score
            info["episode_length"] = np.float64(episode_duration)
            info["reward_components"] = {key: np.float64(sum(value_list))
                                         for key, value_list in self.reward_components.items()}

        return observation_np, new_reward.cpu().numpy(), terminated_np, truncated_np, info


class FlattenStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        is_done = terminated or truncated
        if is_done and "reward_components" in info:
            for key, value in info["reward_components"].items():
                info[f"reward_components/{key}"] = value
        # sb3 monitor will now see these new scalar keys and log them.
        # will still ignore the original "reward_components" dict.
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


class CustomMultiPolicyWalker(MultiAgentEnv):
    def __init__(self, env_config: Dict):
        super().__init__()
        self.num_policies = env_config["num_policies"]
        self.reward_fn_list = env_config["reward_fn_list"]
        self.env_id = env_config["env_id"]

        # dictionary to hold all independent environment instances
        self.envs = {}
        self.policy_ids = [f"policy_{i}" for i in range(self.num_policies)]
        self.possible_agents = self.policy_ids.copy()

        # instantiate N environments with own reward function and policy ID
        for i, p_id in enumerate(self.policy_ids):
            base_env = gym.make(self.env_id)
            # Apply custom reward wrapper, passing the specific policy ID
            wrapped_env = CustomRewardWrapper(base_env, self.reward_fn_list[i], p_id)
            self.envs[p_id] = wrapped_env

        single_obs_space = self.envs[self.policy_ids[0]].observation_space
        single_act_space = self.envs[self.policy_ids[0]].action_space

        self.action_space = gym.spaces.Dict(
            {p_id: single_act_space for p_id in self.policy_ids}
        )
        self.observation_space = gym.spaces.Dict(
            {p_id: single_obs_space for p_id in self.policy_ids}
        )

        self._agent_ids = set(self.policy_ids)
        self.agents = self.policy_ids.copy()
        self.info = {}

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        obs = {}
        info = {}
        for p_id, env in self.envs.items():
            o, i = env.reset(seed=seed, options=options)
            obs[p_id] = o
            info[p_id] = i
        self.agents = self.policy_ids.copy()
        self.info = {}
        return obs, info

    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}

        for p_id, action in action_dict.items():
            env = self.envs[p_id]
            o, r, t, tr, i = env.step(action)

            obs[p_id] = o
            rewards[p_id] = r
            terminated[p_id] = t
            truncated[p_id] = tr
            info[p_id] = i
            if t or tr:
                self.agents.remove(p_id)
                self.info[p_id] = info[p_id]

        # RLlib requires a single __all__ flag when all agents are done
        terminated["__all__"] = not self.agents
        truncated["__all__"] = not self.agents

        return obs, rewards, terminated, truncated, info


def compile_func_from_string(
        code_string: str) -> Callable:
    # find the expected function name
    name_pattern = re.compile(r"def\s+([a-zA-Z_]\w*)")
    match = name_pattern.search(code_string)
    if not match:
        raise ValueError("Could not find any function definition 'def ...' in code string.")

    function_name = match.group(1)

    # Validate that the *first* function found is the one we expect.
    if function_name != EXPECTED_NAME:
        raise ValueError(f"Function name is incorrect. Expected: '{EXPECTED_NAME}', Got: '{function_name}'")

    tmp_path = None
    module_name = None

    try:
        # delete=False is crucial for compatibility, especially on Windows
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(code_string)
            tmp.flush()

            module_name = tmp_path.stem
            spec = importlib.util.spec_from_file_location(module_name, tmp_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec from {tmp_path}")

            imported_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = imported_module

            # executes the code, including imports and decorators
            spec.loader.exec_module(imported_module)

            # get the specific function we're looking for
            if not hasattr(imported_module, function_name):
                raise AttributeError(f"Module was imported but does not have function '{function_name}'.")

            func_object = getattr(imported_module, function_name)
            return func_object
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        if module_name and module_name in sys.modules:
            del sys.modules[module_name]


class RewardFunctionWrapper:
    def __init__(self, code_string: str, full_string: str):
        self.code_string = code_string
        self.full_string = full_string
        self._compiled_func = None

    def __call__(self, *args, **kwargs):
        if self._compiled_func is None:
            """print statement is helpful for confirming that compilation
            is happening on the worker, not the main process."""
            print(f"Compiling reward function on worker...")
            self._compiled_func = compile_func_from_string(self.code_string)
            print("Compilation complete.")

        return self._compiled_func(*args, **kwargs)
