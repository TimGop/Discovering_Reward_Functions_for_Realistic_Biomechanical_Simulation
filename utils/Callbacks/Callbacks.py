import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# Note: The BaseEnv type hint might be needed depending on your setup
# from ray.rllib.env import BaseEnv
from typing import Dict, Any, Optional

from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


class LogMetrics(DefaultCallbacks):

    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["policy_dicts"] = {}

    def on_episode_step(self, *, episode, env: Any, **kwargs):
        # access CustomMultiPolicyWalker through env.envs[0].unwrapped
        info = env.envs[0].unwrapped.info
        if len(info) == len(env.envs[0].unwrapped.policy_ids):
            for policy_id in env.envs[0].unwrapped.policy_ids:
                policy_info = info[policy_id]

                if policy_id not in episode.custom_data["policy_dicts"]:
                    episode.custom_data["policy_dicts"][policy_id] = {
                        "reward_components": {}
                    }
                target_policy_dict = episode.custom_data["policy_dicts"][policy_id]
                target_policy_dict["fitness_score"] = policy_info["fitness_score"]
                target_policy_dict["episode_length"] = policy_info["episode_length"]
                source_rewards = policy_info["reward_components"]
                target_rewards_dict = target_policy_dict["reward_components"]
                for comp_name, comp_value in source_rewards.items():
                    target_rewards_dict[comp_name] = comp_value

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        metrics_logger.log_dict(episode.custom_data["policy_dicts"], key="policy_dicts", reduce=None)


