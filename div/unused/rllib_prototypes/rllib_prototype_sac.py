from ray.rllib.algorithms.sac import SACConfig


def train_rllib_single_policy(
        env_id: str,
        max_timesteps: int = 3_000_000,  # This parameter is not used in the loop
        hidden_layers: list[int] = None,
):
    if not hidden_layers:
        # SAC generally requires larger networks than PPO for good performance
        hidden_layers = [256, 256]

    # ray.init(ignore_reinit_error=True) # Uncomment if not running in a cluster

    # --- SAC (Off-Policy) Data Collection & Training ---
    # Collect small fragments frequently from each worker
    rollout_fragment_length = 50
    num_env_runners = 8
    # Sample this many transitions from the replay buffer for each training step
    train_batch_size = 256

    max_iterations = 3000  # Keep the original number of .train() calls

    config = (
        SACConfig()  # Use SACConfig
        .environment(env_id)
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            rollout_fragment_length=rollout_fragment_length,
            observation_filter="MeanStdFilter",
        )
        .training(
            # --- SAC-specific hyperparameters ---
            lr=3e-4,  # Common SAC learning rate
            train_batch_size=train_batch_size,
            gamma=0.99,
            grad_clip=0.5,  # Kept from original
            tau=0.005,  # Soft-update coefficient for target networks
            num_steps_sampled_before_learning_starts=10000,  # Replay buffer warmup
            num_training_intensity_updates_per_env_step=1.0,  # Train 1 step per 1 env step

            # --- PPO params (removed) ---
            # lr=2e-4,
            # minibatch_size=64,
            # num_epochs=10,
            # lambda_=0.9,
            # clip_param=0.2,
            # entropy_coeff=0.0,
            # vf_loss_coeff=0.5,
        )
        .rl_module(
            model_config={
                "policy_model_config": {
                    "fcnet_hiddens": hidden_layers,
                    "fcnet_activation": "relu",
                },
                "q_model_config": {
                    "fcnet_hiddens": hidden_layers,
                    "fcnet_activation": "relu",
                },
            }
            # --- PPO model config (removed) ---
            # model_config={
            #     "fcnet_hiddens": hidden_layers,
            #     "fcnet_activation": "tanh",
            #     "vf_share_layers": True,
            # }
        )
        .resources(num_gpus=0)
        .debugging(seed=0)
    )

    algo = config.build_algo()

    print(f"Starting single-policy training for {max_iterations} iterations...")

    # The training loop remains the same
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