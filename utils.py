import openai
import os
from openai import OpenAI
import gymnasium as gym

from CustomRewardWrapper import compile_func_from_string
import re
from typing import List, Callable, Any, Tuple, Dict
import torch
import tempfile
import importlib.util
import sys
from pathlib import Path
import inspect

EXPECTED_NAME = "custom_reward_fn"
EXPECTED_PARAMS = ["observation", "action", "original_reward", "terminated", "truncated"]
EXPECTED_RETURN_TYPE = Tuple[torch.Tensor, Dict[str, torch.Tensor]]

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


def parse_and_validate_code_blocks(env_id, text_blob: str, required_imports=None) -> List[str]:
    """
    Parses a text blob to find all Python code blocks.
    Validates that each block *contains* '@torch.jit.script' and
    matches the required Python function signature.
    Returns a list of the valid original code strings (with decorator intact).
    """
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

            # Compile the *non-JIT* function for inspection
            function = compile_func_from_string(full_code_for_inspection)

            # ["observation", "action", "original_reward", "terminated", "truncated"]
            temp_env = gym.make(env_id)
            obs_np, info = temp_env.reset()
            action_np = temp_env.action_space.sample()
            next_observation_np, reward_np, terminated, truncated, info = temp_env.step(action_np)
            temp_env.close()

            obs_tensor = torch.as_tensor(next_observation_np, dtype=torch.float32)
            act_tensor = torch.as_tensor(action_np, dtype=torch.float32)
            rew_tensor = torch.as_tensor(reward_np, dtype=torch.float32)
            term_tensor = torch.as_tensor(terminated, dtype=torch.bool)
            trunc_tensor = torch.as_tensor(truncated, dtype=torch.bool)

            # Call the *scripted_function*
            new_reward, components = function_jit(
                obs_tensor, act_tensor, rew_tensor, term_tensor, trunc_tensor
            )
            assert components

            if not inspect.isfunction(function):
                # This might happen if the decorator removal fails or code is unusual
                raise TypeError(f"Loaded object is not a Python function. Got: {type(function)}")

            # Validate signature directly on the Python function
            sig = inspect.signature(function)

            # Validate parameters
            actual_params = list(sig.parameters.keys())
            if actual_params != EXPECTED_PARAMS:
                raise ValueError(f"Function has incorrect parameters. "
                                 f"Expected: {EXPECTED_PARAMS}, Got: {actual_params}")

            # Validate return type
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
    return valid_code_strings


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
