import multiprocessing
import os
import platform

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
from utils.utils import create_custom_reward_env, process_callback_stats

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


def train_sac(p_id, reward_func, args, stat_frequency: int, eureka_it: int):

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
    model_name = f"sac_{ENV_ID}_{eureka_it}_{p_id}_{TOTAL_TIMESTEPS}_{timestamp}"
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    STATS_PATH = os.path.join(MODEL_DIR, f"{model_name}_vecnormalize.pkl")

    print(f"[{p_id}] creating vectorized environment for {ENV_ID}...")
    custom_reward_creation_func = functools.partial(create_custom_reward_env, p_id, reward_func, ENV_ID)
    vec_env = make_vec_env(custom_reward_creation_func, n_envs=N_ENVS, seed=args.vec_env_seed)

    print(f"[{p_id}] normalizing the environment...")
    vec_env = VecNormalize(vec_env, norm_obs=args.vec_env_norm_obs, norm_reward=args.vec_env_norm_reward,
                           clip_obs=args.vec_env_clip_obs)

    ent_coef_arg = args.sac_ent_coef
    if ent_coef_arg != "auto":
        try:
            ent_coef_arg = float(ent_coef_arg)
        except ValueError:
            print(f"Warning: Could not parse ent_coef '{ent_coef_arg}', defaulting to 'auto'")
            ent_coef_arg = "auto"

    print(f"[{p_id}] defining the sbx SAC model...")

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

    def eval_env_creator():
        # create gym env with rgb for recording videos
        func = functools.partial(create_custom_reward_env, p_id, reward_func, ENV_ID, "rgb_array")
        venv = make_vec_env(func, n_envs=1, seed=args.vec_env_seed)
        venv = VecNormalize(venv, norm_obs=args.vec_env_norm_obs, norm_reward=False,
                            clip_obs=args.vec_env_clip_obs, training=False)
        return venv

    stats_callback = AutoRecordStatsCallback(env_creator=eval_env_creator,
                                             video_folder=f"videos/SAC_eureka_{eureka_it}_policy_{p_id}")

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


def train_sbx_parallel_policies_SAC(reward_funcs, args, stat_frequency: int, eureka_it: int):
    """
    Trains multiple SAC policies in parallel using multiprocessing.
    """
    num_workers = args.num_parallel_trains

    tasks = []
    for idx, reward_func in enumerate(reward_funcs):
        tasks.append((idx, reward_func, args, stat_frequency, eureka_it))

    print(f"\n--- Training {len(reward_funcs)} Policies (SAC) ({num_workers} in Parallel at a time) ---")

    # 'spawn' is required for CUDA/JAX safety
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_workers) as pool:
        all_stats = pool.starmap(train_sac, tasks)

    print(f"--- All {len(all_stats)} Parallel Training Jobs Complete ---")
    return all_stats
