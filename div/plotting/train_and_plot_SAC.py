import argparse
import os
import platform
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from reward_functions.eureka_reward_funcs import custom_reward_fn_with_video, custom_reward_fn_without_video

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

import time
import functools

# SBX / SB3 Imports
from sbx import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# custom Utils
from utils.Callbacks.Callbacks_sb3 import AutoRecordStatsCallback
from utils.utils import create_custom_reward_env

# for recording
system_os = platform.system()  # Returns 'Windows', 'Linux', or 'Darwin' (Mac)

# For main process
if system_os == "Windows":
    # Windows: Do nothing.
    # MuJoCo will default to "glfw" (which creates a hidden window for recording).
    print("Detected Windows: Using default MuJoCo backend (GLFW).")

elif system_os == "Linux":
    # Linux: Check if we are headless (no monitor attached)
    # The 'DISPLAY' environment variable is usually missing on headless servers.
    print("Detected Headless Linux: Force-enabling EGL backend.")
    os.environ["MUJOCO_GL"] = "egl"


def train_sac(args, name, run_no: int, reward_func=None):
    # For every subprocess as well!
    if system_os == "Windows":
        # Windows: Do nothing.
        # MuJoCo will default to "glfw" (which creates a hidden window for recording).
        print("Detected Windows(subprocess): Using default MuJoCo backend (GLFW).")

    elif system_os == "Linux":
        # Linux: Check if we are headless (no monitor attached)
        # The 'DISPLAY' environment variable is usually missing on headless servers.
        if "DISPLAY" not in os.environ:
            print("Detected Headless Linux(subprocess): Force-enabling EGL backend.")
            os.environ["MUJOCO_GL"] = "egl"
        else:
            print("Detected Linux with Display(subprocess): Using default backend.")

    ENV_ID = args.env_id
    N_ENVS = args.n_envs
    # SAC is off-policy and doesn't strictly use n_steps for updates,
    # but we use it here to calculate total duration and stats aggregation.
    N_STEPS = args.n_steps
    TOTAL_TIMESTEPS = args.n_rollouts * N_ENVS * N_STEPS
    LOG_DIR = args.log_dir
    MODEL_DIR = args.model_dir

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = int(time.time())
    model_name = f"sac_{ENV_ID}_{run_no}_{TOTAL_TIMESTEPS}_{timestamp}"
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    STATS_PATH = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")

    print(f"creating vectorized environment for {ENV_ID}...")
    if reward_func is not None:
        custom_reward_creation_func = functools.partial(create_custom_reward_env, 0, reward_func, ENV_ID)
        vec_env = make_vec_env(custom_reward_creation_func, n_envs=N_ENVS, seed=args.vec_env_seed)
    else:
        custom_reward_creation_func = functools.partial(create_custom_reward_env, 0, None, ENV_ID)
        vec_env = make_vec_env(custom_reward_creation_func, n_envs=N_ENVS, seed=args.vec_env_seed)

    print(f"normalizing the environment...")
    # Note: SAC often works well without normalization, but we keep it to match ppo workflow
    vec_env = VecNormalize(vec_env, norm_obs=args.vec_env_norm_obs, norm_reward=False,
                           clip_obs=args.vec_env_clip_obs)

    ent_coef_arg = args.sac_ent_coef
    if ent_coef_arg != "auto":
        try:
            ent_coef_arg = float(ent_coef_arg)
        except ValueError:
            print(f"Warning: Could not parse ent_coef '{ent_coef_arg}', defaulting to 'auto'")
            ent_coef_arg = "auto"

    print(f"defining the sbx SAC model...")

    # Using the hyperparameters from your reference script
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,

        # --- Optimization ---
        learning_rate=args.sac_learning_rate,
        gamma=args.sac_gamma,
        tau=args.sac_tau,
        ent_coef=ent_coef_arg,

        # --- Replay Buffer ---
        buffer_size=args.sac_buffer_size,  # Large RAM usage warning applies here
        batch_size=args.sac_batch_size,
        learning_starts=args.sac_learning_starts,

        # --- Training Frequency ---
        train_freq=args.sac_train_freq,
        gradient_steps=args.sac_gradient_steps,

        # --- Exploration ---
        use_sde=args.sac_use_sde,

        # --- Logging ---
        verbose=args.sac_verbose,
        tensorboard_log=LOG_DIR,
        device="auto"  # SBX (JAX) will try to use GPU
    )

    if reward_func is not None:
        def eval_env_creator():
            # create gym env with rgb for recording videos
            func = functools.partial(create_custom_reward_env, 0, reward_func, ENV_ID, "rgb_array")
            venv = make_vec_env(func, n_envs=1, seed=args.vec_env_seed)
            venv = VecNormalize(venv, norm_obs=args.vec_env_norm_obs, norm_reward=False,
                                clip_obs=args.vec_env_clip_obs, training=False)
            return venv
    else:
        def eval_env_creator():
            # create gym env with rgb for recording videos
            func = functools.partial(create_custom_reward_env, 0, None, ENV_ID, "rgb_array")
            venv = make_vec_env(func, n_envs=1, seed=args.vec_env_seed)
            venv = VecNormalize(venv, norm_obs=args.vec_env_norm_obs, norm_reward=False,
                                clip_obs=args.vec_env_clip_obs, training=False)
            return venv

    stats_callback = AutoRecordStatsCallback(env_creator=eval_env_creator,
                                             video_folder=f"videos_final/SAC_{name}_{run_no}")

    print(f"starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        tb_log_name=model_name,
        callback=stats_callback
    )

    print(f"training finished... saving model and environment stats...")
    model.save(MODEL_PATH)
    vec_env.save(STATS_PATH)
    vec_env.close()

    print(f"model saved to: {MODEL_PATH}.zip")
    print(f"environment stats saved to: {STATS_PATH}")
    print(f"parsing monitor logs...")

    print(f"--- training Complete ---")
    return stats_callback.episode_stats


def plot_combined_fitness_curves(experiments_dict, save_dir, plot_name="combined_fitness"):
    plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')

    # each model gets a distinct color
    colors = ['#1f77b4', '#e377c2', '#2ca02c', '#ff7f0e', '#9467bd']

    all_max_scores = {}

    for idx, (model_name, episode_stats_list) in enumerate(experiments_dict.items()):
        dfs = []
        max_fitness_scores = []
        color = colors[idx % len(colors)]  # cycle through colors

        if not episode_stats_list:
            print(f"No stats for {model_name}")
            continue

        for run_idx, episode_stats in enumerate(episode_stats_list):
            if not episode_stats:
                continue

            df = pd.DataFrame(episode_stats)

            if 'fitness_score' not in df.columns or 'l' not in df.columns:
                continue

            max_fitness_scores.append(df['fitness_score'].max())
            df['total_steps'] = df['l'].cumsum()
            dfs.append(df)

        if not dfs:
            print(f"No valid DataFrames for {model_name}")
            continue

        all_max_scores[model_name] = max_fitness_scores

        # interpolation & alignment
        max_steps = max(df['total_steps'].iloc[-1] for df in dfs)
        min_steps = min(df['total_steps'].iloc[0] for df in dfs)

        # create common grid for THIS model
        common_x = np.linspace(min_steps, max_steps, num=200)
        interpolated_y_values = []

        for df in dfs:
            y_interp = np.interp(common_x, df['total_steps'], df['fitness_score'])
            interpolated_y_values.append(y_interp)

        y_matrix = np.array(interpolated_y_values)
        y_mean = np.mean(y_matrix, axis=0)
        y_std = np.std(y_matrix, axis=0)

        # plot the Mean Line
        plt.plot(common_x, y_mean, color=color, linewidth=2, label=f"{model_name} (Mean)")

        # plot the Standard Deviation Shading
        plt.fill_between(common_x, y_mean - y_std, y_mean + y_std,
                         color=color, alpha=0.15, label=f"_nolegend_")  # _nolegend_ hides it from legend

    # Finalize Figure Layout
    plt.title("Fitness Score Comparison: Video Feedback vs Base")
    plt.xlabel("Number of Steps")
    plt.ylabel("Fitness Score")
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.png")

    plt.savefig(save_path, dpi=300)  # Increased DPI for better quality
    plt.close()

    print(f"Combined fitness plot saved to: {save_path}")
    return all_max_scores


def main(args):
    all_experiments = {}

    # 1. Eureka WITH video feedback
    print("Running: Eureka with video feedback...")
    custom_reward_fn = custom_reward_fn_with_video
    stats_vid = []
    for i in range(10):
        stats_vid.append(train_sac(args, name="eureka_with_vid", run_no=i, reward_func=custom_reward_fn))

    all_experiments["Eureka (With Video)"] = stats_vid

    # 2. Base reward function
    print("Running: Base reward function...")
    stats_base = []
    for i in range(10):
        stats_base.append(train_sac(args, name="base", run_no=i))

    all_experiments["Base Reward"] = stats_base

    # 3. Eureka WITHOUT video feedback
    print("Running: Eureka without video feedback...")
    custom_reward_fn = custom_reward_fn_without_video
    stats_no_vid = []
    for i in range(10):
        stats_no_vid.append(train_sac(args, name="eureka_without_vid", run_no=i, reward_func=custom_reward_fn))

    all_experiments["Eureka (No Video)"] = stats_no_vid

    print("Generating combined plot...")
    print(plot_combined_fitness_curves(all_experiments, "plots", "comparison_fitness_curve"))


if __name__ == "__main__":
    print("training for plotting and videos started...")
    parser = argparse.ArgumentParser(description="Testing and plotting args match eureka script")

    parser.add_argument("--rl_algorithm", type=str, default="SAC", help="rl algorithm to use for training",
                        choices=["PPO", "SAC"])

    parser.add_argument("--library", type=str, default="stable_baselines_3",
                        help="stable baselines 3 or ray.rllib", choices=["stable_baselines_3", "ray_rllib"])

    # --- Evolution Loop Parameters ---
    parser.add_argument("--env_id", type=str, default="Humanoid-v5", help="Gymnasium environment ID",
                        choices=["Walker2d-v5", "Humanoid-v5"])
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of reward evolution iterations")

    # --- RL Training Parameters ---
    # steps per rl agent rollout is n_envs * n_steps and therefore total steps is n_rollouts * n_envs * n_steps
    parser.add_argument("--n_rollouts", type=int, default=250,
                        help="Total rollouts for *each* PPO policy training")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments for PPO")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per environment per PPO update (rollout buffer size)")

    # --- Logging ---
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for Stable Baselines3 logs (Tensorboard)")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save trained models and VecNormalize stats")

    # --- PPO Hyperparameters ---
    parser.add_argument("--ppo_learning_rate", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--ppo_batch_size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--ppo_n_epochs", type=int, default=10, help="PPO number of epochs per update")
    parser.add_argument("--ppo_gamma", type=float, default=0.99, help="PPO discount factor")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="PPO GAE lambda")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ppo_ent_coef", type=float, default=0.0, help="PPO entropy coefficient")
    parser.add_argument("--ppo_vf_coef", type=float, default=0.5, help="PPO value function coefficient")
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="PPO max gradient norm")
    parser.add_argument("--ppo_verbose", type=float, default=1,
                        help="PPO verbosity level: 0 for no output, 1 for info messages "
                             "(such as device or wrappers used), 2 for debug messages")

    # --- SAC Hyperparameters ---
    parser.add_argument("--sac_learning_rate", type=float, default=3e-4, help="SAC learning rate")
    parser.add_argument("--sac_buffer_size", type=int, default=1_000_000, help="SAC replay buffer size")  # 1_000_000
    parser.add_argument("--sac_batch_size", type=int, default=256, help="SAC mini-batch size")
    parser.add_argument("--sac_gamma", type=float, default=0.99, help="SAC discount factor")
    parser.add_argument("--sac_tau", type=float, default=0.005, help="SAC soft update coefficient (polyak averaging)")
    parser.add_argument("--sac_ent_coef", type=str, default="auto",
                        help="SAC entropy coefficient (use 'auto' or a float string like '0.1')")
    parser.add_argument("--sac_learning_starts", type=int, default=10_000, help="SAC steps before learning starts")
    parser.add_argument("--sac_train_freq", type=int, default=1, help="SAC training frequency (in steps)")
    parser.add_argument("--sac_gradient_steps", type=int, default=1, help="SAC gradient steps per training trigger")
    parser.add_argument("--sac_use_sde", action="store_true", default=False,
                        help="Enable State Dependent Exploration (SDE)")
    parser.add_argument("--sac_verbose", type=int, default=1, help="SAC verbosity level")

    # --- Vectorized environment parameters ---
    parser.add_argument("--vec_env_norm_obs", type=bool, default=True, help="normalize observations in env")
    parser.add_argument("--vec_env_norm_reward", type=bool, default=False, help="normalize rewards in env")
    parser.add_argument("--vec_env_clip_obs", type=float, default=10.0, help="observation clipping param")
    parser.add_argument("--vec_env_seed", type=int, default=0, help="initial random seed for env")

    parsed_args = parser.parse_args()
    main(parsed_args)
