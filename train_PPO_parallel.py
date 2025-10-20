import ray
from ray.rllib.algorithms.ppo import PPOConfig
from CustomRewardWrapper import CustomMultiPolicyWalker  # Assuming this file exists
from fitness_funcs_and_placeholder_reward_funcs import calculate_fitness_walker2d, walker2d_less_speed
import gymnasium as gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_rllib_multi_policy(
        reward_list: list,
        total_timesteps: int = 1_000_000,
        hidden_layers: list[int] = None
):
    if not hidden_layers:
        hidden_layers = [64, 64]

    ray.init(ignore_reinit_error=True)

    num_parallel_policies = len(reward_list)

    def policy_mapping_fn(agent_id, episode, **kwargs):
        return agent_id

    env_id = "Walker2d-v4"
    temp_env = gym.make(env_id)
    fitness_func = calculate_fitness_walker2d
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    policy_names = [f"policy_{i}" for i in range(num_parallel_policies)]

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
        .framework("torch")
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=1,
            observation_filter="MeanStdFilter",
            rollout_fragment_length=2048
        )
        .training(
            lr=3e-4,
            train_batch_size=8192,
            minibatch_size=64,
            num_sgd_iter=10,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.0,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model={
                "fcnet_hiddens": hidden_layers,
                "fcnet_activation": "tanh",
                # match the separate policy_net and value_net
                "vf_share_layers": False,
            },
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

    algo = config.build_algo()

    timesteps_so_far = 0
    iteration = 0
    max_fitness = [0 for _ in range(num_parallel_policies)]

    print(f"Starting Multi-Policy PPO Training for {num_parallel_policies} Policies...")

    while timesteps_so_far < total_timesteps:
        result = algo.train()
        timesteps_so_far = result["num_env_steps_sampled_lifetime"]
        env_runner_stats = result.get("env_runners", {})

        # per-policy rewards
        per_policy_rewards = env_runner_stats.get("agent_episode_returns_mean", {})

        # overall reward across all policies
        global_reward_mean = env_runner_stats.get("episode_return_mean", float('nan'))

        print(
            f"Iter: {iteration + 1}, total timesteps: {timesteps_so_far}, "
            f"total reward mean: {global_reward_mean:.2f}"
        )

        if per_policy_rewards:
            print(" Mean rewards per policy:")
            for policy_id, reward in per_policy_rewards.items():
                print(f" {policy_id}: {reward:.2f}")

        if iteration % 20 == 0:
            fitness_env = gym.make(env_id)
            for idx, policy_name in enumerate(policy_names):
                policy = algo.get_module(policy_name)
                curr_fitness = fitness_func(fitness_env, policy, num_episodes=1)
                if curr_fitness > max_fitness[idx]:
                    max_fitness[idx] = curr_fitness
                    print("new max fitness " + str(max_fitness[idx]) + " achieved by " + policy_name + "...")
        iteration += 1

    algo.stop()
    ray.shutdown()
    return max_fitness


if __name__ == '__main__':
    # for testing purposes
    reward_fn_list = [walker2d_less_speed for _ in range(4)]
    fitness_per_rew_func = train_rllib_multi_policy(
        reward_list=reward_fn_list,
        total_timesteps=3_000_000
    )
    print(fitness_per_rew_func)
