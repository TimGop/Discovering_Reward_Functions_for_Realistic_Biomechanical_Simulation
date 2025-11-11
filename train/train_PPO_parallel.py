import ray
from ray.rllib.algorithms.ppo import PPOConfig

from utils.Callbacks.Callbacks import LogMetrics
from utils.utils import CustomMultiPolicyWalker  # Assuming this file exists
from utils.utils import walker2d_original_reward
from utils.utils import RewardFunctionWrapper
import gymnasium as gym
import warnings
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _update_stats(training_results, result, policy_names):
    for idx, p_id in enumerate(policy_names):
        # print(f"result is {result}")
        ep_scores = result["env_runners"]["policy_dicts"][p_id]["fitness_score"]
        ep_lens = result["env_runners"]["policy_dicts"][p_id]["episode_length"]
        reward_components = result["env_runners"]["policy_dicts"][p_id]["reward_components"]

        # ep_scores
        training_results[idx]["score"]["min"] += [np.min(ep_scores)]
        training_results[idx]["score"]["mean"] += [np.mean(ep_scores)]
        training_results[idx]["score"]["max"] += [np.max(ep_scores)]

        # ep_lens
        training_results[idx]["ep_lens"]["min"] += [np.min(ep_lens)]
        training_results[idx]["ep_lens"]["mean"] += [np.mean(ep_lens)]
        training_results[idx]["ep_lens"]["max"] += [np.max(ep_lens)]

        # rew components
        for key, returns_list in reward_components.items():
            component_dict = training_results[idx]["reward_components"].setdefault(key, {})
            component_dict.setdefault("min", []).append(np.min(returns_list))
            component_dict.setdefault("mean", []).append(np.mean(returns_list))
            component_dict.setdefault("max", []).append(np.max(returns_list))
    return training_results


def _finalize_stats(results):
    for p_idx in range(len(results)):
        results[p_idx]["score"]["value_list"] = results[p_idx]["score"]["mean"]
        results[p_idx]["score"]["mean"] = np.mean(results[p_idx]["score"]["mean"])
        results[p_idx]["score"]["min"] = np.mean(results[p_idx]["score"]["min"])
        results[p_idx]["score"]["max"] = np.mean(results[p_idx]["score"]["max"])

        results[p_idx]["ep_lens"]["value_list"] = results[p_idx]["ep_lens"]["mean"]
        results[p_idx]["ep_lens"]["mean"] = np.mean(results[p_idx]["ep_lens"]["mean"])
        results[p_idx]["ep_lens"]["min"] = np.mean(results[p_idx]["ep_lens"]["min"])
        results[p_idx]["ep_lens"]["max"] = np.mean(results[p_idx]["ep_lens"]["max"])

        for key in results[p_idx]["reward_components"]:
            results[p_idx]["reward_components"][key]["value_list"] = results[p_idx]["reward_components"][key]["mean"]
            results[p_idx]["reward_components"][key]["mean"] = np.mean(results[p_idx]["reward_components"][key]["mean"])
            results[p_idx]["reward_components"][key]["min"] = np.mean(results[p_idx]["reward_components"][key]["min"])
            results[p_idx]["reward_components"][key]["max"] = np.mean(results[p_idx]["reward_components"][key]["max"])

    return results


def train_rllib_multi_policy(
        env_id: str,
        reward_list: list,
        max_iterations: int = 3_000,
        hidden_layers: list[int] = None,
        stat_frequency: int = 300
):
    if not hidden_layers:
        hidden_layers = [64, 64]

    # ray.init(ignore_reinit_error=True)

    num_parallel_policies = len(reward_list)

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id

    temp_env = gym.make(env_id)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    policy_names = [f"policy_{i}" for i in range(num_parallel_policies)]

    # create training_results dict to be populated further below
    training_results = []
    for i, reward_fn in enumerate(reward_list):
        code_str = ""
        # assumes reward_list contains RewardFunctionWrapper objects
        if isinstance(reward_fn, RewardFunctionWrapper):
            code_str = reward_fn.code_string
        else:
            print(f"Warning: reward_fn for policy_{i} is not a RewardFunctionWrapper. Code will not be saved.")

        training_results.append({
            "score": {"min": [], "mean": [], "max": []},
            "ep_lens": {"min": [], "mean": [], "max": []},
            "reward_components": {},
            "error": None,
            "code": code_str
        })

    # policy specs
    module_specs = {}
    for policy_id in policy_names:
        module_specs[policy_id] = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "fcnet_hiddens": hidden_layers,
                "fcnet_activation": "tanh",
                "vf_share_layers": True,
            })

    # define policy spaces and run config
    policy_spaces = {
        f"policy_{i}": (
            None,
            obs_space,
            act_space,
            {}
        ) for i in range(num_parallel_policies)
    }

    config = (
        PPOConfig()
        .environment(
            CustomMultiPolicyWalker,
            env_config={
                "env_id": env_id,
                "num_policies": num_parallel_policies,
                "reward_fn_list": reward_list,
            },
        )
        .callbacks(LogMetrics)
        .framework("torch")
        .env_runners(
            num_env_runners=8,  # CRITICAL CHANGE: Set to 8 workers
            num_envs_per_env_runner=1,
            observation_filter="MeanStdFilter",
            rollout_fragment_length=1024
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=module_specs
            )
        )
        .training(
            lr=2.15e-4,
            train_batch_size=8192,  # Eureka uses 131,072
            minibatch_size=64,
            num_epochs=10,  # num_sgd_iter
            lambda_=0.73,
            clip_param=0.21,
            entropy_coeff=0.0,  # 0.01
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            gamma=0.99,
        )
        .resources(
            num_gpus=1
        )
        .multi_agent(
            policies=policy_spaces,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policy_names
        )
        .debugging(seed=0)
    )
    config.env_runners(sample_timeout_s=180)
    algo = config.build_algo()

    iteration = 0

    print(f"Starting Multi-Policy PPO Training for {num_parallel_policies} Policies...")

    # [{"score": dict, "reward_components": dict, "ep_lens": dict, "error": str, "code": str}, ...]
    while iteration < max_iterations:
        result = algo.train()
        timesteps_so_far = result["num_env_steps_sampled_lifetime"]
        env_runner_stats = result.get("env_runners", {})

        per_policy_rewards = env_runner_stats.get("agent_episode_returns_mean", {})
        global_reward_mean = env_runner_stats.get("episode_return_mean", float('nan'))

        if iteration % stat_frequency == 0:
            training_results = _update_stats(training_results, result, policy_names)

        print(
            f"\nIter: {iteration + 1}, total timesteps: {timesteps_so_far}, "
            f"total reward mean: {global_reward_mean:.2f}"
        )

        if per_policy_rewards:
            print(" Mean rewards per policy:")
            for policy_id, reward in per_policy_rewards.items():
                print(f" {policy_id}: {reward:.2f}\n")

        iteration += 1

    algo.stop()
    ray.shutdown()

    training_results = _finalize_stats(training_results)
    return training_results


if __name__ == '__main__':
    # for testing purposes
    reward_fn_list = [walker2d_original_reward for _ in range(1)]
    fitness_per_rew_func = train_rllib_multi_policy(env_id="Walker2d-v4",
                                                    reward_list=reward_fn_list, stat_frequency=1000, max_iterations=3000
                                                    )
    print(fitness_per_rew_func)
