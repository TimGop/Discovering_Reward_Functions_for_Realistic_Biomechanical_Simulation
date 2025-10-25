import openai
import os
from openai import OpenAI

from CustomRewardWrapper import compile_func_from_string
import re
from typing import List, Callable, Any

# New imports needed for the solution
import tempfile
import importlib.util
import sys
from pathlib import Path


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



def parse_and_validate_code_blocks(text_blob: str, required_imports=None) -> List[str]:
    """
    Parses a text blob to find all Python code blocks, validates that each
    can be compiled by torch.jit.script, and returns a list of the valid
    code STRINGS.
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
            # Attempt to compile the function here in the main process.
            # We don't keep the result; this is purely for validation.
            # Prepend any required import statements
            import_statements = ""
            if required_imports:
                import_statements = "\n".join([f"{lib}" for lib in required_imports]) + "\n\n"

            full_code = import_statements + code_str

            print(f"Validating block {i + 1}...")
            compile_func_from_string(full_code)

            # If the line above doesn't raise an error, the code is valid.
            print(f"Block {i + 1} is valid.")
            valid_code_strings.append(full_code)

        except Exception as e:
            # If compilation fails, report the error and skip this function.
            print(f"Validation failed for block {i + 1}. Error: {e}")
            print("-" * 20)

    print(f"\nValidation complete. Found {len(valid_code_strings)} valid functions.")
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
