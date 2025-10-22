from train_PPO_parallel import train_rllib_multi_policy
from utils import (ChatSession, save_string_to_file, load_string_from_file,
                   parse_and_validate_code_blocks)
from prompts import (init_sys_prompt, code_formatting_tip, rew_reflection_1, rew_reflection_2, walker_2d_v4_description,
                     pre_env, reward_func_context, num_rew_funcs_directive, code_formatting_tip_bonus)
from environment_code import walker_2d_v4_code
from CustomRewardWrapper import RewardFunctionWrapper


def reward_evolution(num_iterations=5):
    if num_iterations < 1:
        num_iterations = 1
    gpt = ChatSession(model="gpt-5")  # -nano-2025-08-07
    original_prompt = (
            init_sys_prompt + "\n" + reward_func_context + "\n" + pre_env + "\n" + walker_2d_v4_code + "\n" +
            walker_2d_v4_description + "\n" + code_formatting_tip + "\n" + code_formatting_tip_bonus + "\n" +
            num_rew_funcs_directive)
    reward_string = gpt.ask(original_prompt)
    print(reward_string)
    reward_strings_list = parse_and_validate_code_blocks(reward_string)
    reward_funcs = [RewardFunctionWrapper(reward_string) for reward_string in reward_strings_list]
    assert len(reward_funcs) > 0
    fitness_scores = train_rllib_multi_policy(reward_list=reward_funcs, total_timesteps=16384)
    reflection_string = ""
    for _ in range(num_iterations - 1):
        # TODO create reward reflection prompt
        new_prompt = """test new prompt"""
        prompt = new_prompt + reflection_string
        # reward_string = llm_placeholder(prompt)
        # reward_funcs = parse_all_code_blocks(reward_string)
        reward_string = gpt.ask(original_prompt)
        print(reward_string)
        reward_strings_list = parse_and_validate_code_blocks(reward_string)
        reward_funcs = [RewardFunctionWrapper(reward_string) for reward_string in reward_strings_list]
        assert len(reward_funcs) > 0
        fitness_scores = train_rllib_multi_policy(reward_list=reward_funcs, total_timesteps=16384)
        reflection_string = ""


if __name__ == '__main__':
    reward_evolution()
