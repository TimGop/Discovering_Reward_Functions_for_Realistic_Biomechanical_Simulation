import openai
import os

from google import genai

from .prompts_and_env_code.prompts import vid_comp_prompt
from openai import OpenAI
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from .CustomRewardWrapper import compile_func_from_string, CustomRewardWrapper, FlattenStatsWrapper, \
    RewardFunctionWrapper, JustFitnessWrapper
import re
from typing import List, Tuple, Dict
import torch
import inspect
import numpy as np
import pandas as pd
import time

from .prompts_and_env_code.prompts import rew_reflection_1, rew_reflection_2, code_formatting_tip_bonus, \
    code_formatting_tip

EXPECTED_NAME = "custom_reward_fn"
EXPECTED_PARAMS = ["observation", "action", "next_observation", "terminated", "truncated"]
EXPECTED_RETURN_TYPE = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


# GPT for obtaining reward functions in Eureka loop
class ChatSession:
    def __init__(self, model="gpt-5-nano-2025-08-07"):
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception("Error: The OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI()
        self.model = model

    def ask(self, messages) -> str:
        try:
            # Send the entire conversation to the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            reply = response.choices[0].message.content

        except openai.NotFoundError as e:
            return f"An error occurred: The model does not exist. Please use a valid model name. Details: {e}"

        except Exception as e:
            return f"An unexpected error occurred: {e}"

        return reply


def get_funcs(env_id, gpt, messages, num_funcs=16):
    reward_funcs = []
    while len(reward_funcs) < num_funcs:
        reward_string = gpt.ask(messages=messages)
        reward_str_list, full_str = parse_and_validate_code_blocks(env_id, reward_string)
        if len(reward_str_list) > 1:
            print("Multiple valid reward functions detected in single llm output..."
                  " Will ignore this iteration...")
        elif len(reward_str_list) == 1:
            reward_funcs += [RewardFunctionWrapper(reward_str_list[0], full_str)]
    return reward_funcs


def parse_and_validate_code_blocks(env_id, text_blob: str, required_imports=None) -> Tuple[List[str], str]:
    if required_imports is None:
        required_imports = ["import torch", "from typing import Tuple", "from typing import Dict"]

    patterns = [
        r'"""python(.*?)"""',
        r"‘‘‘python(.*?)‘‘‘",
        r"'''python(.*?)'''",
        r'```python(.*?)```',
        r'"""(.*?)"""',
        r"‘‘‘(.*?)‘‘‘",
        r"'''(.*?)'''",
        r'```(.*?)```',
    ]
    combined_pattern = re.compile("|".join(patterns), re.DOTALL)

    # print("\n\ntext blob:\n"+text_blob+"\n")

    code_blocks = []
    for match in combined_pattern.finditer(text_blob):
        code_content = next((g for g in match.groups() if g is not None), None)
        if code_content:
            code_blocks.append(code_content.strip())

    print(f"Found {len(code_blocks)} potential code blocks. Now validating...")

    valid_code_strings = []
    for i, code_str in enumerate(code_blocks):

        try:
            if f"def {EXPECTED_NAME}" not in code_str:
                raise ValueError(f"Could not find function definition 'def {EXPECTED_NAME}'")

            if "@torch.jit.script" not in code_str:
                raise ValueError("Could not find decorator '@torch.jit.script'")

            compilation_code_str = code_str.replace("@torch.jit.script", "")

            import_statements = ""
            if required_imports:
                import_statements = "\n".join([f"{lib}" for lib in required_imports]) + "\n\n"

            full_code_for_inspection = import_statements + compilation_code_str
            original_full_code = import_statements + code_str

            print(f"Validating block {i + 1}...")
            function_jit = compile_func_from_string(original_full_code)

            function = compile_func_from_string(full_code_for_inspection)

            # ["observation", "action", "original_reward", "next_observation", "terminated", "truncated"]
            temp_env = gym.make(env_id)
            obs_np, info = temp_env.reset()
            action_np = temp_env.action_space.sample()
            next_obs_np, _, terminated, truncated, info = temp_env.step(action_np)
            temp_env.close()

            obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
            next_obs_tensor = torch.as_tensor(next_obs_np, dtype=torch.float32)
            act_tensor = torch.as_tensor(action_np, dtype=torch.float32)
            term_tensor = torch.as_tensor(terminated, dtype=torch.bool)
            trunc_tensor = torch.as_tensor(truncated, dtype=torch.bool)

            new_reward, components = function_jit(
                obs_tensor, act_tensor, next_obs_tensor, term_tensor, trunc_tensor
            )
            assert components

            if not inspect.isfunction(function):
                raise TypeError(f"Loaded object is not a Python function. Got: {type(function)}")

            sig = inspect.signature(function)

            actual_params = list(sig.parameters.keys())
            if actual_params != EXPECTED_PARAMS:
                raise ValueError(f"Function has incorrect parameters. "
                                 f"Expected: {EXPECTED_PARAMS}, Got: {actual_params}")

            actual_return = sig.return_annotation
            if actual_return != EXPECTED_RETURN_TYPE:
                raise ValueError(f"Function has incorrect return type. "
                                 f"Expected: {EXPECTED_RETURN_TYPE}, Got: {actual_return}")

            # If all checks pass, the code is valid.
            print(f"Block {i + 1} is valid.")
            valid_code_strings.append(original_full_code)  # Save the original jit version

        except Exception as e:
            print(f"Validation failed for block {i + 1}. Error: {e}")
            print("-" * 20)

    print(f"Validation complete. Found {len(valid_code_strings)} valid functions.\n")
    return valid_code_strings, text_blob


def get_reflection(training_results, messages, epoch_freq, args, eureka_iteration, video_llm, previous_best_fitness):
    best_index = None
    max_fitness = float("-inf")
    for idx, result in enumerate(training_results):
        curr_max_fitness = result["score"]["max"]
        if curr_max_fitness > max_fitness:
            max_fitness = curr_max_fitness
            best_index = idx

    assert best_index is not None  # not None
    print("highest max fitness for policy"+str(best_index))

    if args.library == "ray_rllib" or args.video_feedback is False:
        policy_video_path = None
        target_video_path = None
    else:
        target_video_path = "videos/target_run.mp4"
        if args.rl_algorithm == "PPO":
            policy_video_path = (f"videos/PPO_eureka_{eureka_iteration}_policy_{best_index}/"
                                 f"best_fitness-step-0-to-step-1000.mp4")
        else:  # SAC
            policy_video_path = (f"videos/SAC_eureka_{eureka_iteration}_policy_{best_index}/"
                                 f"best_fitness-step-0-to-step-1000.mp4")

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

    if args.library != "ray_rllib" and args.video_feedback is True:
        vid_reflection = compare_videos([policy_video_path, target_video_path], video_llm, best_code_string)
        reflection_string += f"\n\n{vid_reflection}"

    length = len(messages)
    if length == 2:
        messages += [{"role": "assistant", "content": best_code_string}]
        messages += [{"role": "user", "content": reflection_string}]
        print("\n\n" + best_code_string + "\n\n")
        print("\n\n" + reflection_string + "\n\n")
    else:
        messages_suffix = [{"role": "assistant", "content": best_code_string},
                           {"role": "user", "content": reflection_string}]
        assert (length == 4 or length == 6)

        if previous_best_fitness <= max_fitness:
            messages = messages[:2] + messages_suffix
            print("\n\n" + best_code_string + "\n\n")
            print("\n\n" + reflection_string + "\n\n")
        else:
            messages = messages[:4] + messages_suffix
            print("\n\n" + str(messages[2]) + "\n\n" + str(messages[3]) + "\n\n")
            print("\n\n" + best_code_string + "\n\n")
            print("\n\n" + reflection_string + "\n\n")

    return messages, max_fitness


def wait_for_files_active(file_names, client):
    """Waits for uploaded files to transition to the ACTIVE state."""
    print("Waiting for file processing...", end="")

    for name in file_names:
        while True:
            # Check file status
            file_obj = client.files.get(name=name)

            if file_obj.state == "ACTIVE":
                # File is ready
                break
            elif file_obj.state == "FAILED":
                raise Exception(f"File {name} failed to process.")

            # If still PROCESSING, wait and check again
            print(".", end="", flush=True)
            time.sleep(2)

    print("\nAll files ready.")


def compare_videos(video_paths, client, reward_code=None):
    uploaded_files = []

    # 1. Upload
    for path in video_paths:
        print(f"Uploading: {path}...")
        file_ref = client.files.upload(file=path)
        uploaded_files.append(file_ref)

    # 2. CRITICAL STEP: WAIT FOR PROCESSING
    # We extract the names from the file objects to pass to our wait function
    file_names = [f.name for f in uploaded_files]
    wait_for_files_active(file_names, client)

    # 3. Request
    request_content = []

    # Add the actual file objects to the request
    for f in uploaded_files:
        request_content.append(f)

    # Add prompt
    prompt_text = vid_comp_prompt
    if reward_code:
        prompt_text += reward_code
    request_content.append(prompt_text)

    print("Generating comparison...")

    # Use the correct model
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=request_content
    )

    # 4. Cleanup
    print("Cleaning up files...")
    for name in file_names:
        client.files.delete(name=name)

    return response.text


def save_string_to_file(filepath, content):
    try:
        # 'w' mode opens the file for writing.
        # If the file already exists, its contents are discarded.
        # The 'with' statement ensures the file is properly closed.
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully saved string to '{filepath}'")
    except IOError as e:
        print(f"Error: Could not write to file '{filepath}'. Reason: {e}")


def load_string_from_file(filepath):
    try:
        # 'r' mode opens the file for reading.
        # The 'with' statement ensures the file is properly closed.
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Successfully loaded string from '{filepath}'")
        return content
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except IOError as e:
        print(f"Error: Could not read from file '{filepath}'. Reason: {e}")
        return None


def create_custom_reward_env(p_id, reward_func, ENV_ID, render_mode=None):
    if reward_func is None:
        base_env = gym.make(ENV_ID, render_mode=render_mode)
        env = RecordEpisodeStatistics(FlattenStatsWrapper(JustFitnessWrapper(base_env)))
    else:
        base_env = gym.make(ENV_ID, render_mode=render_mode)
        env = CustomRewardWrapper(base_env, reward_func, p_id)
        env = FlattenStatsWrapper(env)
        env = RecordEpisodeStatistics(env)
    return env


def finalize_single_stat(result):
    def safe_mean(lst):
        return np.mean(lst) if len(lst) > 0 else 0.0

    def safe_min(lst):
        return np.min(lst) if len(lst) > 0 else 0.0

    def safe_max(lst):
        return np.max(lst) if len(lst) > 0 else 0.0

    if result["score"]["mean"]:
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
        return finalize_single_stat(stats)

    df = pd.DataFrame(episode_stats)

    batch_size_timesteps = n_steps * n_envs
    df['timesteps_cumsum'] = df['l'].cumsum()
    df['batch'] = (df['timesteps_cumsum'] - 1) // batch_size_timesteps

    reward_component_cols = [col for col in df.columns if col.startswith("reward_components/")]

    if 'fitness_score' not in df.columns:
        print(f"Warning: 'fitness_score' not found in callback stats. Check wrappers.")
        return finalize_single_stat(stats)

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

    return finalize_single_stat(stats)


def list_available_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables.")
        return

    client = genai.Client(api_key=api_key)

    print(f"{'Model Name':<30} | {'Display Name':<30}")
    print("-" * 65)

    try:
        # Paginator for listing models
        for model in client.models.list():
            print(model)

    except Exception as e:
        print(f"An error occurred: {e}")


"""if __name__ == "__main__":
    # Run `set GEMINI_API_KEY=your_key` in terminal before running script
    API_KEY = os.environ.get("GEMINI_API_KEY")

    if not API_KEY:
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        client = genai.Client(api_key=API_KEY)

        video_locations = [
            "../videos/eureka_0_policy_0/best_fitness-step-0-to-step-1000.mp4",
            "../videos/eureka_0_policy_1/best_fitness-step-0-to-step-1000.mp4"
        ]

        # Ensure files actually exist before running to avoid other errors
        if all(os.path.exists(p) for p in video_locations):
            print(compare_videos(video_locations, client))
        else:
            print("Video files not found. Check paths.")

    # list_available_models()"""
