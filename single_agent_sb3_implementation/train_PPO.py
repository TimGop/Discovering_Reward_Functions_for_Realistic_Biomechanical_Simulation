import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time

# --- 1. Define Constants and Configuration ---
# Environment ID for the Walker2d task
ENV_ID = "Walker2d-v4"
# Total number of training steps
TOTAL_TIMESTEPS = 3_000_000
# Directory for logs and the trained model
LOG_DIR = "../logs"
MODEL_DIR = "../models"

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Unique filename for the model and environment stats
timestamp = int(time.time())
model_name = f"ppo_{ENV_ID}_{TOTAL_TIMESTEPS}_{timestamp}"
MODEL_PATH = os.path.join(MODEL_DIR, model_name)
STATS_PATH = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")


# --- 2. Create the Vectorized and Normalized Environment ---
# Vectorized environments allow for parallel training and are standard in SB3
# We use make_vec_env to easily create a vectorized environment
# n_envs=4 means we are running 4 environments in parallel
print(f"Creating vectorized environment for {ENV_ID}...")
vec_env = make_vec_env(ENV_ID, n_envs=4, seed=0)

# VecNormalize is a wrapper that normalizes observations and rewards
# This is a common and effective technique for continuous control tasks
print("Normalizing the environment...")
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)


# --- 3. Define and Train the PPO Model ---
# We use the MlpPolicy because the observation space is a vector of continuous values
# For environments with image-based observations, you would use CnnPolicy
# Key PPO Hyperparameters:
#   - learning_rate: How much to update the policy weights at each step.
#   - n_steps: The number of steps to run for each environment per update.
#   - batch_size: The number of samples used for each policy update.
#   - gamma: The discount factor for future rewards.
# verbose=1 will print training progress.
# tensorboard_log creates logs that can be visualized with TensorBoard.
print("Defining the PPO model...")
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
# The learn() method starts the training process
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
    tb_log_name=model_name
)

# --- 4. Save the Model and Environment Statistics ---
print("Training finished. Saving model and environment stats...")
model.save(MODEL_PATH)
vec_env.save(STATS_PATH)

print(f"Model saved to: {MODEL_PATH}.zip")
print(f"Environment stats saved to: {STATS_PATH}")
print("--- Training Complete ---")

# Close the environment
vec_env.close()
