import multiprocessing
import platform
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time
import functools
from utils.Callbacks.Callbacks_sb3 import AutoRecordStatsCallback
from utils.utils import create_custom_reward_env, process_callback_stats

# for recording
system_os = platform.system()  # Returns 'Windows', 'Linux', or 'Darwin' (Mac)

if system_os == "Windows":
    # Windows: Do nothing.
    # MuJoCo will default to "glfw" (which creates a hidden window for recording).
    print("Detected Windows: Using default MuJoCo backend (GLFW).")

elif system_os == "Linux":
    # Linux: Check if we are headless (no monitor attached)
    # The 'DISPLAY' environment variable is usually missing on headless servers.
    if "DISPLAY" not in os.environ:
        print("Detected Headless Linux: Force-enabling EGL backend.")
        os.environ["MUJOCO_GL"] = "egl"
    else:
        print("Detected Linux with Display: Using default backend.")


def train_ppo(p_id, reward_func, args, stat_frequency: int, eureka_it: int):
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
        device='cpu'  # sb3 ppo made to train on cpu (actually faster)
    )

    def eval_env_creator():
        # create gym env with rgb for recording videos
        func = functools.partial(create_custom_reward_env, p_id, reward_func, ENV_ID, "rgb_array")
        venv = make_vec_env(func, n_envs=1, seed=args.vec_env_seed)
        venv = VecNormalize(venv, norm_obs=args.vec_env_norm_obs, norm_reward=False,
                            clip_obs=args.vec_env_clip_obs, training=False)
        return venv

    stats_callback = AutoRecordStatsCallback(env_creator=eval_env_creator,
                                             video_folder=f"videos/PPO_eureka_{eureka_it}_policy_{p_id}")

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


def train_sb3_parallel_policies_PPO(reward_funcs, args, stat_frequency: int, eureka_it: int):
    """
    Trains multiple PPO policies in parallel using multiprocessing.
    """
    num_workers = args.num_parallel_trains

    # create a list of arguments for each train_ppo call
    # each item is a tuple: (p_id, reward_func, args, stat_frequency)
    tasks = []
    for idx, reward_func in enumerate(reward_funcs):
        tasks.append((idx, reward_func, args, stat_frequency, eureka_it))

    print(f"\n--- Training {len(reward_funcs)} Policies ({num_workers} in Parallel at a time) ---")

    # use 'spawn' start method for CUDA safety. This is crucial!
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_workers) as pool:
        # use starmap to pass the tuples of arguments to train_ppo
        all_stats = pool.starmap(train_ppo, tasks)

    print(f"--- All {len(all_stats)} Parallel Training Jobs Complete ---")
    return all_stats
