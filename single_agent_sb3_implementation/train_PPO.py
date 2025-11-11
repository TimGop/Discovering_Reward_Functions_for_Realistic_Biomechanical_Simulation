import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time

ENV_ID = "Walker2d-v4"
TOTAL_TIMESTEPS = 3_000_000
LOG_DIR = "../logs"
MODEL_DIR = "../models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Unique filename for the model and environment stats
timestamp = int(time.time())
model_name = f"ppo_{ENV_ID}_{TOTAL_TIMESTEPS}_{timestamp}"
MODEL_PATH = os.path.join(MODEL_DIR, model_name)
STATS_PATH = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")

print(f"Creating vectorized environment for {ENV_ID}...")
vec_env = make_vec_env(ENV_ID, n_envs=4, seed=0)


print("Normalizing the environment...")
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

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
