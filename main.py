from train.train_PPO_parallel import train_rllib_multi_policy
import argparse
from train.train_PPO_sb3 import train_sb3_sequnetial_policies
from utils.utils import (ChatSession, parse_and_validate_code_blocks)
from utils.prompts_and_env_code.prompts import (init_sys_prompt, code_formatting_tip, rew_reflection_1,
                                                rew_reflection_2, walker_2d_v4_description,
                                                reward_func_context, code_formatting_tip_bonus,
                                                init_user_prompt)
from utils.prompts_and_env_code.environment_code import walker_2d_v4_code
from utils.CustomRewardWrapper import RewardFunctionWrapper
import ray


def get_funcs(env_id, gpt, messages, num_funcs=16):
    reward_funcs = []
    while len(reward_funcs) < num_funcs:
        reward_string = gpt.ask(messages=messages)
        reward_funcs += [RewardFunctionWrapper(string)
                         for string in parse_and_validate_code_blocks(env_id, reward_string)]
    assert len(reward_funcs) > 0
    return reward_funcs


def get_reflection(training_results, messages, epoch_freq):
    best_index = None
    max_fitness = float("-inf")
    for idx, result in enumerate(training_results):
        curr_max_fitness = result["score"]["max"]
        if curr_max_fitness > max_fitness:
            max_fitness = curr_max_fitness
            best_index = idx

    assert best_index is not None  # not None
    best_result = training_results[best_index]
    separator = "\n"
    key_val_format = "{k}: {v}, max: {max}, mean: {mean}, min: {min}"

    reward_component_string = separator.join(key_val_format.format(k=k, v=str(dictionary["value_list"]),
                                                                   max=dictionary["max"], mean=dictionary["mean"],
                                                                   min=dictionary["min"])
                                             for k, dictionary in best_result["reward_components"].items())

    fitness_and_ep_lens_string = separator.join(
        [key_val_format.format(k="fitness score", v=best_result["score"]["value_list"],
                               max=best_result["score"]["max"],
                               mean=best_result["score"]["mean"],
                               min=best_result["score"]["min"]),
         key_val_format.format(k="episode lengths",
                               v=best_result["ep_lens"]["value_list"],
                               max=best_result["ep_lens"]["max"],
                               mean=best_result["ep_lens"]["mean"],
                               min=best_result["ep_lens"]["min"])
         ])

    best_code_string = best_result["code"]

    reflection_string = (rew_reflection_1.format(epoch_freq=epoch_freq) + "\n" + reward_component_string + "\n"
                         + fitness_and_ep_lens_string + "\n\n" + rew_reflection_2 + " " + code_formatting_tip + "\n" +
                         code_formatting_tip_bonus)

    print("\n\n"+reflection_string+"\n\n")

    if len(messages) == 2:
        messages += [{"role": "assistant", "content": best_code_string}]
        messages += [{"role": "user", "content": reflection_string}]
    else:
        assert len(messages) == 4
        messages[-2] = {"role": "assistant", "content": best_code_string}
        messages[-1] = {"role": "user", "content": reflection_string}

    return messages


def reward_evolution(alg_args):
    """
    main eureka reward evolution loop
    """
    if alg_args.num_iterations < 1:
        alg_args.num_iterations = 1

    epoch_stat_freq = max(int(alg_args.n_rollouts // 10), 1)  # stat frequency, e.g., 10% of total batches

    gpt = ChatSession(model=alg_args.gpt_model)

    sys_prompt = (
            init_sys_prompt + "\n" + reward_func_context + "\n" + code_formatting_tip + "\n"
            + code_formatting_tip_bonus)
    user_prompt = init_user_prompt.format(task_obs_code_string=walker_2d_v4_code,
                                          task_description=walker_2d_v4_description)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    for iteration in range(alg_args.num_iterations):
        print(f"Running iteration {iteration + 1} of reward iteration process...")
        reward_funcs = get_funcs(env_id=alg_args.env_id, gpt=gpt, messages=messages,
                                 num_funcs=alg_args.num_funcs_per_iteration)

        while True:
            try:
                if alg_args.library == "stable_baselines_3":
                    training_results = train_sb3_sequnetial_policies(reward_funcs=reward_funcs, args=alg_args,
                                                                     stat_frequency=epoch_stat_freq)
                else:
                    training_results = train_rllib_multi_policy(reward_list=reward_funcs, hidden_layers=[64, 64],
                                                                env_id=alg_args.env_id, stat_frequency=300,
                                                                max_iterations=3_000)
                break
            except Exception as e:
                print(f"Training failed with exception: {e}")
                # sometimes when using ray workers fail and retrying the training loop "magically fixes the problem
                print("Retrying training loop...")

        messages = get_reflection(training_results, messages, epoch_stat_freq)


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
    parser = argparse.ArgumentParser(description="Evolve reward functions using PPO and GPT.")

    # --- Library ---
    parser.add_argument("--library", type=str, default="stable_baselines_3", help="stable baselines 3 or ray.rllib"
                        , choices=["stable_baselines_3", "ray_rllib"])

    # Note: args below are only for sb3 implementation (aside from env_id)
    # sb3 delivers more stable performant learning vs rllib ppo implementation (rllib is faster per episode multipolicy)

    # --- Evolution Loop Parameters ---
    parser.add_argument("--env_id", type=str, default="Walker2d-v4", help="Gymnasium environment ID")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of reward evolution iterations")
    parser.add_argument("--gpt_model", type=str, default="gpt-5-nano-2025-08-07",
                        choices=["gpt-4-turbo", "gpt-5-nano-2025-08-07", "gpt-5"],
                        help="GPT model name for ChatSession")
    parser.add_argument("--num_funcs_per_iteration", type=int, default=16,
                        help="Number of reward functions to generate per iteration")

    # --- RL Training Parameters ---
    # steps per rl agent rollout is n_envs * n_steps and therefore total steps is n_rollouts * n_envs * n_steps
    parser.add_argument("--n_rollouts", type=int, default=250,
                        help="Total rollouts for *each* PPO policy training")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments for PPO")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per environment per PPO update (rollout buffer size)")

    # --- Logging ---
    parser.add_argument("--log_dir", type=str, default="../../logs",
                        help="Directory for Stable Baselines3 logs (Tensorboard)")
    parser.add_argument("--model_dir", type=str, default="../../models",
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

    # --- Vectorized environment parameters ---
    parser.add_argument("--vec_env_norm_obs", type=bool, default=True, help="normalize observations in env")
    parser.add_argument("--vec_env_norm_reward", type=bool, default=True, help="normalize rewards in env")
    parser.add_argument("--vec_env_clip_obs", type=float, default=10.0, help="observation clipping param")
    parser.add_argument("--vec_env_seed", type=int, default=0, help="initial random seed for env")

    args = parser.parse_args()

    main(args)
