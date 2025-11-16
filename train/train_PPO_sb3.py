import multiprocessing

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time
from utils.CustomRewardWrapper import CustomRewardWrapper, FlattenStatsWrapper
import functools
import numpy as np
import pandas as pd
from utils.Callbacks.Callbacks_sb3 import StatsCallback


def create_custom_reward_env(p_id, reward_func, ENV_ID):
    base_env = gym.make(ENV_ID)
    env = CustomRewardWrapper(base_env, reward_func, p_id)
    env = FlattenStatsWrapper(env)
    env = RecordEpisodeStatistics(env)
    return env


def _finalize_single_stat(result):
    def safe_mean(lst):
        return np.mean(lst) if len(lst) > 0 else 0.0

    def safe_min(lst):
        return np.min(lst) if len(lst) > 0 else 0.0

    def safe_max(lst):
        return np.max(lst) if len(lst) > 0 else 0.0

    if result["score"]["mean"]:  # only finalize if we have data
        result["score"]["value_list"] = result["score"]["mean"]
        result["score"]["mean"] = safe_mean(result["score"]["mean"])
        result["score"]["min"] = safe_min(result["score"]["min"])
        result["score"]["max"] = safe_max(result["score"]["max"])

        result["ep_lens"]["value_list"] = result["ep_lens"]["mean"]
        result["ep_lens"]["mean"] = safe_mean(result["ep_lens"]["mean"])
        result["ep_lens"]["min"] = safe_min(result["ep_lens"]["min"])
        result["ep_lens"]["max"] = safe_max(result["ep_lens"]["max"])

        for key in result["reward_components"]:
            result["reward_components"][key]["value_list"] = result["reward_components"][key]["mean"]
            result["reward_components"][key]["mean"] = safe_mean(result["reward_components"][key]["mean"])
            result["reward_components"][key]["min"] = safe_min(result["reward_components"][key]["min"])
            result["reward_components"][key]["max"] = safe_max(result["reward_components"][key]["max"])
    return result


def process_callback_stats(episode_stats, n_steps, n_envs, reward_func_code, stat_frequency=1):
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


def train_ppo(p_id, reward_func, args, stat_frequency: int):
    ENV_ID = args.env_id
    N_ENVS = args.n_envs
    N_STEPS = args.n_steps
    TOTAL_TIMESTEPS = args.n_rollouts * N_ENVS * N_STEPS
    LOG_DIR = args.log_dir
    MODEL_DIR = args.model_dir

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())
    model_name = f"ppo_{ENV_ID}_{p_id}_{TOTAL_TIMESTEPS}_{timestamp}"
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    STATS_PATH = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")

    print(f"[{p_id}] creating vectorized environment for {ENV_ID}...")
    custom_reward_creation_func = functools.partial(create_custom_reward_env, p_id, reward_func, ENV_ID)
    vec_env = make_vec_env(custom_reward_creation_func, n_envs=N_ENVS, seed=args.vec_env_seed)

    print(f"[{p_id}] normalizing the environment...")
    vec_env = VecNormalize(vec_env, norm_obs=args.vec_env_norm_obs, norm_reward=args.vec_env_norm_reward,
                           clip_obs=args.vec_env_clip_obs)

    print(f"[{p_id}] defining the PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=args.ppo_learning_rate,
        n_steps=N_STEPS,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_n_epochs,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_gae_lambda,
        clip_range=args.ppo_clip_range,
        ent_coef=args.ppo_ent_coef,
        vf_coef=args.ppo_vf_coef,
        max_grad_norm=args.ppo_max_grad_norm,
        device='cpu'   # sb3 ppo made to train on cpu (actually faster)
    )
    stats_callback = StatsCallback()

    print(f"[{p_id}] starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        tb_log_name=model_name,
        callback=stats_callback
    )

    print(f"[{p_id}] training finished... saving model and environment stats...")
    model.save(MODEL_PATH)
    vec_env.save(STATS_PATH)
    vec_env.close()

    print(f"[{p_id}] model saved to: {MODEL_PATH}.zip")
    print(f"[{p_id}] environment stats saved to: {STATS_PATH}")
    print(f"[{p_id}] parsing monitor logs...")

    full_string = ""
    if hasattr(reward_func, 'full_string'):
        full_string = reward_func.full_string
    else:
        print(f"[{p_id}] warning: reward_func has no 'code_string' attribute. Saving empty code.")

    stats = process_callback_stats(
        stats_callback.episode_stats,
        n_steps=N_STEPS,
        n_envs=N_ENVS,
        reward_func_code=full_string,
        stat_frequency=stat_frequency
    )

    print(f"--- [{p_id}] training Complete ---")
    return stats


def train_sb3_parallel_policies(reward_funcs, args, stat_frequency: int):
    """
    Trains multiple PPO policies in parallel using multiprocessing.
    """
    num_workers = args.num_parallel_trains

    # create a list of arguments for each train_ppo call
    # each item is a tuple: (p_id, reward_func, args, stat_frequency)
    tasks = []
    for idx, reward_func in enumerate(reward_funcs):
        tasks.append((idx, reward_func, args, stat_frequency))

    print(f"\n--- Training {len(reward_funcs)} Policies ({num_workers} in Parallel at a time) ---")

    # use 'spawn' start method for CUDA safety. This is crucial!
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_workers) as pool:
        # use starmap to pass the tuples of arguments to train_ppo
        all_stats = pool.starmap(train_ppo, tasks)

    print(f"--- All {len(all_stats)} Parallel Training Jobs Complete ---")
    return all_stats
