from train_PPO_parallel import train_rllib_multi_policy
from utils import (ChatSession, save_string_to_file, load_string_from_file,
                   parse_and_validate_code_blocks)
from prompts import (init_sys_prompt, code_formatting_tip, rew_reflection_1, rew_reflection_2, walker_2d_v4_description,
                     pre_env, reward_func_context, num_rew_funcs_directive, code_formatting_tip_bonus, init_user_prompt)
from environment_code import walker_2d_v4_code
from CustomRewardWrapper import RewardFunctionWrapper


def get_funcs(gpt, messages, num_funcs=16):
    reward_funcs = []
    while len(reward_funcs) < num_funcs:
        reward_string = gpt.ask(messages=messages)
        reward_funcs += [RewardFunctionWrapper(string) for string in parse_and_validate_code_blocks(reward_string)]
    assert len(reward_funcs) > 0
    return reward_funcs


def reward_evolution(num_iterations=5):
    if num_iterations < 1:
        num_iterations = 1
    gpt = ChatSession(model="gpt-5-nano-2025-08-07")  # -nano-2025-08-07

    sys_prompt = (
                init_sys_prompt + "\n" + reward_func_context + "\n" + code_formatting_tip + "\n"
                + code_formatting_tip_bonus)
    user_prompt = init_user_prompt.format(task_obs_code_string=walker_2d_v4_code,
                                          task_description=walker_2d_v4_description)
    # print(sys_prompt + "\n\n\n" + user_prompt)

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    reward_funcs = get_funcs(gpt, messages=messages)

    fitness_scores = train_rllib_multi_policy(reward_list=reward_funcs, total_timesteps=32768)

    reflection_string = ""
    for iteration in range(num_iterations - 1):
        print(f"Running iteration {iteration + 1} of reward iteration process...")
        # TODO create reward reflection prompt

        reward_funcs = get_funcs(gpt, messages=messages)

        fitness_scores = train_rllib_multi_policy(reward_list=reward_funcs, total_timesteps=32768)


if __name__ == '__main__':
    reward_evolution()
