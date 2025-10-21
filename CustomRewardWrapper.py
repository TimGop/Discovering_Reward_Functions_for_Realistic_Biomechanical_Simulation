import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Callable, Dict, Tuple
import numpy as np
import re
import torch
import tempfile
import importlib.util
import sys
from pathlib import Path


# creates a version of a given gym environment where we can add a custom reward function
class CustomRewardWrapper(gym.Wrapper):
    # ... (Implementation remains the same, but we add a policy ID) ...
    def __init__(self, env: gym.Env, reward_fn: Callable, policy_id: str):
        super().__init__(env)
        self.custom_reward_fn = reward_fn
        self.policy_id = policy_id  # Store the specific policy ID

    def step(self, action):
        observation_np, original_reward_np, terminated_np, truncated_np, info = self.env.step(action)

        # torch.jit.script reward function expects tensors
        observation = torch.as_tensor(observation_np, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        original_reward = torch.as_tensor(original_reward_np, dtype=torch.float32)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool)
        truncated = torch.as_tensor(truncated_np, dtype=torch.bool)

        # Calculate new reward
        new_reward_tuple = self.custom_reward_fn(
            observation, action_tensor, original_reward, terminated, truncated
        )
        new_reward = new_reward_tuple[0]

        info["custom_policy_id"] = self.policy_id

        return observation_np, new_reward.cpu().numpy(), terminated_np, truncated_np, info


# --- RLlib Custom Environment Class ---
# RLlib works best when you define a custom environment that inherits from MultiAgentEnv
# to handle the policy routing.

class CustomMultiPolicyWalker(MultiAgentEnv):
    def __init__(self, env_config: Dict):
        super().__init__()

        self.num_policies = env_config["num_policies"]
        self.reward_fn_list = env_config["reward_fn_list"]
        self.env_id = env_config["env_id"]

        # Create a dictionary to hold all independent environment instances
        self.envs = {}
        self.policy_ids = [f"policy_{i}" for i in range(self.num_policies)]
        self.possible_agents = self.policy_ids

        # Instantiate N environments, each with its own reward function and policy ID
        for i, p_id in enumerate(self.policy_ids):
            base_env = gym.make(self.env_id)
            # Apply your custom reward wrapper, passing the specific policy ID
            wrapped_env = CustomRewardWrapper(base_env, self.reward_fn_list[i], p_id)
            self.envs[p_id] = wrapped_env

        # 1. Get the single-agent space from an instance
        single_obs_space = self.envs[self.policy_ids[0]].observation_space
        single_act_space = self.envs[self.policy_ids[0]].action_space

        # 2. Define the *required* multi-agent spaces as dictionaries
        self.action_space = gym.spaces.Dict(
            {p_id: single_act_space for p_id in self.policy_ids}
        )
        self.observation_space = gym.spaces.Dict(
            {p_id: single_obs_space for p_id in self.policy_ids}
        )

        # RLlib maps policies to agents. Here, each 'agent' is an environment instance.
        self._agent_ids = set(self.policy_ids)
        self.agents = self.policy_ids

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        obs = {}
        info = {}
        for p_id, env in self.envs.items():
            o, i = env.reset(seed=seed, options=options)
            obs[p_id] = o
            info[p_id] = i
        self.agents = self.policy_ids
        return obs, info

    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}

        # Iterate over all policies and run one step for each environment
        for p_id, action in action_dict.items():
            env = self.envs[p_id]
            o, r, t, tr, i = env.step(action)

            obs[p_id] = o
            rewards[p_id] = r
            terminated[p_id] = t
            truncated[p_id] = tr
            # Pass the info back, which now contains the "custom_policy_id" tag
            info[p_id] = i

        # RLlib requires a single __all__ flag when all agents are done
        terminated["__all__"] = any(terminated.values())
        truncated["__all__"] = any(truncated.values())

        done = terminated["__all__"] or truncated["__all__"]
        terminated["__all__"] = done
        truncated["__all__"] = done

        return obs, rewards, terminated, truncated, info


def compile_func_from_string(
    code_string: str) -> callable:
    name_pattern = re.compile(r"def\s+([a-zA-Z_]\w*)")
    match = name_pattern.search(code_string)
    if not match:
        raise ValueError("Could not find function definition in code string.")

    function_name = match.group(1)

    try:
        # Using delete=False is crucial for compatibility, especially on Windows
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(code_string)
            tmp.flush()

            module_name = tmp_path.stem
            spec = importlib.util.spec_from_file_location(module_name, tmp_path)
            imported_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = imported_module
            spec.loader.exec_module(imported_module)
            func_object = getattr(imported_module, function_name)
            return func_object
    finally:
        if 'tmp_path' in locals() and tmp_path.exists():
            tmp_path.unlink()
        if 'module_name' in locals() and module_name in sys.modules:
            del sys.modules[module_name]


class RewardFunctionWrapper:
    """A serializable wrapper that compiles a function from a string on a worker."""

    def __init__(self, code_string: str):
        self.code_string = code_string
        self._compiled_func = None

    def __call__(self, *args, **kwargs):
        if self._compiled_func is None:
            # This print statement is helpful for confirming that compilation
            # is happening on the worker, not the main process.
            print(f"Compiling reward function on worker...")
            self._compiled_func = compile_func_from_string(self.code_string)
            print("Compilation complete.")

        return self._compiled_func(*args, **kwargs)
