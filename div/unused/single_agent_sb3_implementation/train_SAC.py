import gymnasium as gym
import os
from sbx import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

# Create directories to hold models and logs
model_dir = "models2/sac_humanoid"
log_dir = "logs2/sac_humanoid"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def make_env():
    """
    Utility function for multiprocess env.
    """
    # Humanoid-v4 is the gymnasium standard
    gym_env = gym.make("Humanoid-v4", render_mode="rgb_array")
    gym_env = Monitor(gym_env, log_dir)  # Wraps env to track episode rewards/lengths
    return gym_env


if __name__ == "__main__":
    # 1. Create the environment
    # Using a vectorized environment is generally more efficient for sampling
    env = make_vec_env(make_env, n_envs=1)

    # 2. Define the Hyperparameters (mapped from rl_zoo3)
    model = SAC(
        policy="MlpPolicy",
        env=env,

        # --- Optimization ---
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',

        # --- Replay Buffer ---
        buffer_size=1_000_000,  # Warning: This requires significant RAM (~10GB+)
        batch_size=256,
        learning_starts=10_000,

        # --- Training Frequency ---
        train_freq=1,
        gradient_steps=1,

        # --- Exploration ---
        use_sde=False,  # State Dependent Exploration disabled

        # --- Logging ---
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"  # Uses GPU if available
    )

    # 3. Set up Callbacks
    # Save a checkpoint every 200,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=model_dir,
        name_prefix="sac_humanoid"
    )

    # Evaluate the agent every 20,000 steps on a separate test environment
    eval_env = Monitor(gym.make("Humanoid-v4", render_mode="rgb_array"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=log_dir,
        eval_freq=20_000,
        deterministic=True,
        render=False
    )

    # 4. Start Training
    print("Starting training on Humanoid-v4...")
    model.learn(
        total_timesteps=2_000_000,
        callback=[checkpoint_callback, eval_callback]
    )

    # 5. Save Final Model
    model.save(f"{model_dir}/sac_humanoid_final")
    print("Training complete. Model saved.")
