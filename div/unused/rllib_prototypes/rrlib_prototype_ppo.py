from ray.rllib.algorithms.ppo import PPOConfig


def train_rllib_single_policy(
        env_id: str,
        max_timesteps: int = 3_000_000,
        hidden_layers: list[int] = None,
):
    if not hidden_layers:
        hidden_layers = [64, 64]

    # ray.init(ignore_reinit_error=True) # Uncomment if not running in a cluster
    # Total steps per .train() call. Matches SB3 (n_steps * n_envs = 2048 * 4 = 8192)

    rollout_fragment_length = 1024
    num_env_runners = 8
    train_batch_size = 8192

    max_iterations = 3_000

    config = (
        PPOConfig()
        .environment(env_id)  # Use the raw env_id, just like SB3
        .framework("torch")
        .env_runners(
            num_env_runners=16,
            num_envs_per_env_runner=1,
            rollout_fragment_length=rollout_fragment_length,
            observation_filter="MeanStdFilter",
            batch_mode="truncate_episodes"
        )
        .training(
            lr=3e-4,
            train_batch_size=65536,  # 8192
            # minibatch_size=64,
            # num_epochs=10,
            lambda_=0.95,
            kl_coeff=1.0,
            clip_param=0.2,
            # entropy_coeff=0.0,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            gamma=0.99,
            num_sgd_iter=32,
            minibatch_size=4096
        )

        .rl_module(
            model_config={
                "fcnet_hiddens": hidden_layers,
                "fcnet_activation": "tanh",
                "vf_share_layers": True,
            }
        )
        .resources(num_gpus=1)
        .debugging(seed=0)
    )

    algo = config.build_algo()

    print(f"Starting single-policy training for {max_iterations} iterations...")

    for i in range(max_iterations):
        result = algo.train()

        print(
            f"Iter: {i + 1}/{max_iterations}, "
            f"Timesteps: {result['num_env_steps_sampled_lifetime']}, "
            f"Reward Mean: {result['env_runners']['episode_return_mean']:.2f}"
        )

    algo.stop()

    # ray.shutdown() # Uncomment if you called ray.init()

    print("--- Training Complete ---")


if __name__ == '__main__':
    train_rllib_single_policy(env_id="Walker2d-v4")
