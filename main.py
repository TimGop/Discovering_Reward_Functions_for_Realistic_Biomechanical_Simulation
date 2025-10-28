from train_PPO_parallel import train_rllib_multi_policy
from utils import (ChatSession, save_string_to_file, load_string_from_file,
                   parse_and_validate_code_blocks)
from prompts import (init_sys_prompt, code_formatting_tip, rew_reflection_1, rew_reflection_2, walker_2d_v4_description,
                     pre_env, reward_func_context, code_formatting_tip_bonus, init_user_prompt)
from environment_code import walker_2d_v4_code
from CustomRewardWrapper import RewardFunctionWrapper


def get_funcs(gpt, messages, num_funcs=16):
    reward_funcs = []
    while len(reward_funcs) < num_funcs:
        reward_string = gpt.ask(messages=messages)
        reward_funcs += [RewardFunctionWrapper(string) for string in parse_and_validate_code_blocks(reward_string)]
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

    assert best_index  # not None
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

    if len(messages) == 2:
        messages += [{"role": "assistant", "content": best_code_string}]
        messages += [{"role": "user", "content": reflection_string}]
    else:
        assert len(messages) == 4
        messages += [{"role": "assistant", "content": best_code_string}]
        messages += [{"role": "user", "content": reflection_string}]

    return messages


def reward_evolution(num_iterations=5, max_its_rl_run=3_000):
    if num_iterations < 1:
        num_iterations = 1
    epoch_freq = max(int(max_its_rl_run // 10), 1)
    gpt = ChatSession(model="gpt-5-nano-2025-08-07")  # -nano-2025-08-07
    # TODO at a later stage try removing the code_formatting_tip_bonus again
    sys_prompt = (
            init_sys_prompt + "\n" + reward_func_context + "\n" + code_formatting_tip + "\n"
            + code_formatting_tip_bonus)
    user_prompt = init_user_prompt.format(task_obs_code_string=walker_2d_v4_code,
                                          task_description=walker_2d_v4_description)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    reward_funcs = get_funcs(gpt, messages=messages)
    # TODO: This function must be modified to return detailed results: i.e. a list containing a dict for each rl_reward
    #       [{"score": dict, "reward_components": dict, "ep_lens": dict, "error": str, "code": str}, ...]
    training_results = train_rllib_multi_policy(reward_list=reward_funcs, max_iterations=4,
                                                stat_frequency=2)
    messages = get_reflection(training_results, messages, epoch_freq)
    print(f"first full reflection prompt looks like this:\n {messages}")
    for iteration in range(num_iterations - 1):
        print(f"Running iteration {iteration + 1} of reward iteration process...")
        reward_funcs = get_funcs(gpt, messages=messages)
        training_results = train_rllib_multi_policy(reward_list=reward_funcs, max_iterations=4,
                                                    stat_frequency=epoch_freq)
        messages = get_reflection(training_results, messages, epoch_freq)


if __name__ == '__main__':
    reward_evolution(num_iterations=5, max_its_rl_run=3_000)
