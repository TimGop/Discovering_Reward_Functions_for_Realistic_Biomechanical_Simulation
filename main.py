import os

from train.train_PPO_parallel import train_rllib_multi_policy
import argparse
from train.train_PPO_sb3 import train_sb3_parallel_policies_PPO
from train.train_SAC_sbx import train_sbx_parallel_policies_SAC
from utils.utils import (ChatSession, get_funcs, get_reflection)
from utils.prompts_and_env_code.prompts import (init_sys_prompt, code_formatting_tip,
                                                init_user_prompt)
from utils.prompts_and_env_code.environment_code import (walker_2d_v5_code, walker_2d_v5_description,
                                                         Humanoid_v5_code, Humanoid_v5_description)
import ray
from google import genai
API_KEY = os.environ.get("GEMINI_API_KEY")


def reward_evolution(alg_args):
    """
    main eureka reward evolution loop
    """
    best_fitness = float("-inf")  # across all eureka iterations
    if alg_args.num_iterations < 1:
        alg_args.num_iterations = 1

    epoch_stat_freq = max(int(alg_args.n_rollouts // 10), 1)  # stat frequency, e.g., 10% of total batches

    gpt = ChatSession(model=alg_args.gpt_model)
    gemini = genai.Client(api_key=API_KEY)

    sys_prompt = (init_sys_prompt + "\n" + code_formatting_tip)  # "\n" + reward_func_context

    env_code = walker_2d_v5_code if args.env_id == "Walker2d-v5" else Humanoid_v5_code
    env_description = walker_2d_v5_description if args.env_id == "Walker2d-v5" else Humanoid_v5_description

    user_prompt = init_user_prompt.format(task_obs_code_string=env_code,
                                          task_description=env_description)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    for iteration in range(alg_args.num_iterations):
        print(f"Running iteration {iteration + 1} of reward iteration process...")
        reward_funcs = get_funcs(env_id=alg_args.env_id, gpt=gpt, messages=messages,
                                 num_funcs=alg_args.num_funcs_per_iteration)

        while True:
            try:
                if alg_args.library == "stable_baselines_3":
                    if args.rl_algorithm == "PPO":
                        training_results = train_sb3_parallel_policies_PPO(reward_funcs=reward_funcs, args=alg_args,
                                                                           stat_frequency=epoch_stat_freq,
                                                                           eureka_it=iteration)
                    else:  # using SAC
                        training_results = train_sbx_parallel_policies_SAC(reward_funcs=reward_funcs, args=alg_args,
                                                                           stat_frequency=epoch_stat_freq,
                                                                           eureka_it=iteration)
                else:  # using rllib library (PPO) --> Note: no video reward reflection
                    training_results = train_rllib_multi_policy(reward_list=reward_funcs, hidden_layers=[64, 64],
                                                                env_id=alg_args.env_id, stat_frequency=300,
                                                                max_iterations=3_000)
                break
            except Exception as e:
                print(f"Training failed with exception: {e}")
                # sometimes when using ray, workers fail and retrying the training loop "magically" fixes the problem
                print("Retrying training loop...")

        messages, max_fitness = get_reflection(training_results, messages, epoch_stat_freq, args, iteration,
                                               video_llm=gemini, previous_best_fitness=best_fitness)
        if best_fitness < max_fitness:
            best_fitness = max_fitness


def main(alg_args):
    if alg_args.library == "ray_rllib":
        print("training with rllib...")
        ray.init(ignore_reinit_error=True)
    else:
        print("training with sb3...")
    try:
        reward_evolution(alg_args)
    finally:
        print("eureka done...")
        if alg_args.library == "ray_rllib":
            ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolve reward functions using PPO or SAC and GPT.")

    parser.add_argument("--rl_algorithm", type=str, default="SAC", help="rl algorithm to use for training",
                        choices=["PPO", "SAC"])

    # only PPO works with RLLib and sb3 to use SAC you must use sb3 (actually sbx to be precise)
    # Note: video reward reflection variant only implemented for sb3 PPO and SAC
    # --- Library ---
    parser.add_argument("--library", type=str, default="stable_baselines_3",
                        help="stable baselines 3 or ray.rllib", choices=["stable_baselines_3", "ray_rllib"])

    # Note: args below are only for sb3 implementation (aside from env_id)
    # sb3 delivers more stable performant learning vs rllib ppo implementation (rllib is faster per episode multipolicy)
    # For Humanoid env SAC performs alot better

    # --- Evolution Loop Parameters ---
    parser.add_argument("--env_id", type=str, default="Humanoid-v5", help="Gymnasium environment ID",
                        choices=["Walker2d-v5", "Humanoid-v5"])
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of reward evolution iterations")
    parser.add_argument("--gpt_model", type=str, default="gpt-5.1",
                        choices=["gpt-4", "gpt-4-turbo", "gpt-5-nano-2025-08-07", "gpt-5", "gpt-5.1"],
                        help="GPT model name for ChatSession")
    parser.add_argument("--num_funcs_per_iteration", type=int, default=16,
                        help="Number of reward functions to generate per iteration")
    parser.add_argument("--video_feedback", type=bool, default=False,
                        help="have video feedback to incentivize human-like movement")

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

    # --- Multiprocessing parameters ---
    parser.add_argument("--num_parallel_trains", type=int, default=2,
                        help="Number of PPO training processes to run in parallel.")

    args = parser.parse_args()

    main(args)
