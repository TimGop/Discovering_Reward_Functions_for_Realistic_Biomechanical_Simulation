import gymnasium as gym
import torch.nn as nn
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback

# --- Configuration based on YAML ---
ENV_ID = "Humanoid-v4"
TOTAL_TIMESTEPS = int(1e7)
N_ENVS = 1  # STRICT MATCH: Set to 1 to match YAML batch dynamics
LOG_DIR = "logs"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

run_name = f"ppo_{ENV_ID}_{int(time.time())}"

# --- 1. Create Environments ---
print(f"Creating environments...")
vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=0)
# YAML: normalize: true
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

eval_env = make_vec_env(ENV_ID, n_envs=1, seed=12345)
# Important: norm_reward=False for eval so we see the "real" game score
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)


# --- CRITICAL FIX: Callback to Sync Normalization Stats ---
class Sync2VecNormalizeCallback(BaseCallback):
    """Syncs the normalization stats from train_env to eval_env"""

    def __init__(self, train_env, eval_env, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # Sync the running mean and variance
        self.eval_env.obs_rms = self.train_env.obs_rms
        # Also sync reward normalization if you were using it, but we aren't for eval
        return True


# --- 2. Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=100_000 // N_ENVS,
    save_path=os.path.join(MODEL_DIR, "checkpoints"),
    name_prefix=run_name
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=50_000 // N_ENVS,
    deterministic=True,
    render=False
)

# Add the sync callback BEFORE the eval callback
sync_callback = Sync2VecNormalizeCallback(vec_env, eval_env)
callbacks = CallbackList([sync_callback, checkpoint_callback, eval_callback])

# --- 3. Define Model (Strict YAML Match) ---
policy_kwargs = dict(
    log_std_init=-2,
    ortho_init=False,
    activation_fn=nn.ReLU,  # YAML specifies ReLU, not Tanh
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3.56987e-05,  # YAML value is constant (no 'lin_' prefix)
    n_steps=512,
    batch_size=256,
    n_epochs=5,
    gamma=0.95,
    gae_lambda=0.9,
    clip_range=0.3,
    ent_coef=0.00238306,
    max_grad_norm=2,
    vf_coef=0.431892,
    policy_kwargs=policy_kwargs
)

# --- 4. Train ---
print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
    tb_log_name=run_name,
    callback=callbacks
)

model.save(os.path.join(MODEL_DIR, f"{run_name}_final"))
vec_env.save(os.path.join(MODEL_DIR, f"{run_name}_vecnormalize.pkl"))