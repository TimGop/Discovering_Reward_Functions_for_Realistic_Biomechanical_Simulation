import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time
from CustomRewardWrapper import CustomRewardWrapper, FlattenStatsWrapper
import functools
import numpy as np
import pandas as pd
from Callbacks_sb3 import StatsCallback


def create_custom_reward_env(p_id, reward_func, ENV_ID):
    base_env = gym.make(ENV_ID)
    env = CustomRewardWrapper(base_env, reward_func, p_id)
    env = FlattenStatsWrapper(env)
    env = RecordEpisodeStatistics(env)
    return env


def _finalize_single_stat(result):
    def safe_mean(lst):
        return np.mean(lst) if len(lst) > 0 else 0.0

    if result["score"]["mean"]:  # Only finalize if we have data
        result["score"]["value_list"] = result["score"]["mean"]
        result["score"]["mean"] = safe_mean(result["score"]["mean"])
        result["score"]["min"] = safe_mean(result["score"]["min"])
        result["score"]["max"] = safe_mean(result["score"]["max"])

        result["ep_lens"]["value_list"] = result["ep_lens"]["mean"]
        result["ep_lens"]["mean"] = safe_mean(result["ep_lens"]["mean"])
        result["ep_lens"]["min"] = safe_mean(result["ep_lens"]["min"])
        result["ep_lens"]["max"] = safe_mean(result["ep_lens"]["max"])

        for key in result["reward_components"]:
            result["reward_components"][key]["value_list"] = result["reward_components"][key]["mean"]
            result["reward_components"][key]["mean"] = safe_mean(result["reward_components"][key]["mean"])
            result["reward_components"][key]["min"] = safe_mean(result["reward_components"][key]["min"])
            result["reward_components"][key]["max"] = safe_mean(result["reward_components"][key]["max"])
    return result


def process_callback_stats(episode_stats, n_steps, n_envs, reward_func_code, stat_frequency=1):
    """
    Replaces parse_sb3_monitor_logs.
    Processes the list of dicts from the StatsCallback.
    """
    stats = {
        "score": {"min": [], "mean": [], "max": []},
        "ep_lens": {"min": [], "mean": [], "max": []},
        "reward_components": {},
        "error": None,
        "code": reward_func_code
    }

    if not episode_stats:
        print("Warning: No episode stats were collected by the callback.")
        return _finalize_single_stat(stats)

    df = pd.DataFrame(episode_stats)

    # Re-use the same batching logic as before
    batch_size_timesteps = n_steps * n_envs
    df['timesteps_cumsum'] = df['l'].cumsum()
    df['batch'] = (df['timesteps_cumsum'] - 1) // batch_size_timesteps

    reward_component_cols = [col for col in df.columns if col.startswith("reward_components/")]

    if 'fitness_score' not in df.columns:
        print(f"Warning: 'fitness_score' not found in callback stats. Check wrappers.")
        return _finalize_single_stat(stats)

    grouped = df.groupby('batch')
    for batch_idx, batch_df in grouped:
        # Note: will crash if batch_idx not an integer
        if int(batch_idx) % stat_frequency == 0:
            stats["score"]["min"].append(batch_df['fitness_score'].min())
            stats["score"]["mean"].append(batch_df['fitness_score'].mean())
            stats["score"]["max"].append(batch_df['fitness_score'].max())

            len_col = 'episode_length' if 'episode_length' in batch_df.columns else 'l'
            stats["ep_lens"]["min"].append(batch_df[len_col].min())
            stats["ep_lens"]["mean"].append(batch_df[len_col].mean())
            stats["ep_lens"]["max"].append(batch_df[len_col].max())

            for col_name in reward_component_cols:
                key = col_name.split('/')[-1]
                component_dict = stats["reward_components"].setdefault(key, {})
                component_dict.setdefault("min", []).append(batch_df[col_name].min())
                component_dict.setdefault("mean", []).append(batch_df[col_name].mean())
                component_dict.setdefault("max", []).append(batch_df[col_name].max())

    return _finalize_single_stat(stats)


def train_ppo(p_id, reward_func, ENV_ID="Walker2d-v4", TOTAL_TIMESTEPS: int = 3_000_000, stat_frequency: int = 300):
    N_ENVS = 4
    N_STEPS = 2048
    # TODO register run stats like rllib version
    LOG_DIR = "../logs"
    MODEL_DIR = "../models"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Unique filename for the model and environment stats
    timestamp = int(time.time())
    model_name = f"ppo_{ENV_ID}_{TOTAL_TIMESTEPS}_{timestamp}"
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    STATS_PATH = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")

    print(f"creating vectorized environment for {ENV_ID}...")
    custom_reward_creation_func = functools.partial(create_custom_reward_env, p_id, reward_func, ENV_ID)
    vec_env = make_vec_env(custom_reward_creation_func, n_envs=N_ENVS, seed=0)

    print("normalizing the environment...")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print("defining the PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    stats_callback = StatsCallback()

    print(f"starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        tb_log_name=model_name,
        callback=stats_callback
    )

    # --- 4. Save the Model and Environment Statistics ---
    print("training finished... saving model and environment stats...")
    model.save(MODEL_PATH)
    vec_env.save(STATS_PATH)
    vec_env.close()

    print(f"model saved to: {MODEL_PATH}.zip")
    print(f"environment stats saved to: {STATS_PATH}")
    print("parsing monitor logs...")

    code_str = ""
    if hasattr(reward_func, 'code_string'):
        code_str = reward_func.code_string
    else:
        print("warning: reward_func has no 'code_string' attribute. Saving empty code.")

    stats = process_callback_stats(
        stats_callback.episode_stats,
        n_steps=N_STEPS,
        n_envs=N_ENVS,
        reward_func_code=code_str,
        stat_frequency=stat_frequency
    )

    print("--- training Complete ---")
    return stats


def train_sb3_sequnetial_policies(reward_funcs, env_id, max_its, stat_frequency: int = 300):
    all_stats = []
    for idx, reward_func in enumerate(reward_funcs):
        stats = train_ppo(idx, reward_func, ENV_ID=env_id, TOTAL_TIMESTEPS=max_its, stat_frequency=stat_frequency)
        all_stats.append(stats)
    return all_stats
