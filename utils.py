import openai
import os
from openai import OpenAI

from CustomRewardWrapper import compile_func_from_string
from placeholders import return_reward_string_placeholder
import re
import torch
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
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def ask(self, user_input: str) -> str:
        try:
            self.messages.append({"role": "user", "content": user_input})

            # Send the entire conversation to the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )

            reply = response.choices[0].message.content

            self.messages.append({"role": "assistant", "content": reply})

        except openai.NotFoundError as e:
            return f"An error occurred: The model does not exist. Please use a valid model name. Details: {e}"

        except Exception as e:
            return f"An unexpected error occurred: {e}"

        return reply

    def reset(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


def parse_all_code_blocks(text_blob: str) -> List[Callable[..., Any]]:
    patterns = [
        r'```python(.*?)```',  # Markdown python code block
        r'```(.*?)```',  # Generic markdown code block
        r'"""(.*?)"""',  # Triple-double quotes
        r"'''(.*?)'''",  # Triple-single quotes
    ]
    combined_pattern = re.compile("|".join(patterns), re.DOTALL)

    code_blocks = []
    for match in combined_pattern.finditer(text_blob):
        code_content = next((g for g in match.groups() if g is not None), None)
        if code_content:
            code_blocks.append(code_content)

    print(f"Found {len(code_blocks)} code blocks.")

    callable_functions = []
    name_pattern = re.compile(r"def\s+([a-zA-Z_]\w*)")

    for i, code in enumerate(code_blocks):
        code = code.strip()
        if not code:
            continue

        match = name_pattern.search(code)
        if not match:
            print(f"Info: Block {i + 1} does not contain a Python function definition. Skipping.")
            continue

        function_name = match.group(1)

        try:
            # Create a named temporary file that stays open.
            # Suffix is '.py' so Python recognizes it as a module.
            # delete=False is important on Windows to allow re-opening.
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(code)
                # Flush writes to disk to ensure the importer can read it.
                tmp.flush()

                module_name = tmp_path.stem
                spec = importlib.util.spec_from_file_location(module_name, tmp_path)
                imported_module = importlib.util.module_from_spec(spec)

                sys.modules[module_name] = imported_module

                spec.loader.exec_module(imported_module)

                # get the function object from the now-loaded module.
                func_object = getattr(imported_module, function_name)

                print(f"Successfully imported function '{function_name}' from temporary module.")
                # TODO add code instead of function object because torch.jit func needs to be compiled
                #  in ray.rllib (let each worker compile the function independently when it initializes)
                callable_functions.append(func_object)

        except Exception as e:
            print(f"Error parsing or creating function from block {i + 1}: {e}")
        finally:
            # Clean up the temporary file and remove it from sys.modules
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            if 'module_name' in locals() and module_name in sys.modules:
                del sys.modules[module_name]

    return callable_functions


def parse_and_validate_code_blocks(text_blob: str, required_imports=None) -> List[str]:
    """
    Parses a text blob to find all Python code blocks, validates that each
    can be compiled by torch.jit.script, and returns a list of the valid
    code STRINGS.
    """
    if required_imports is None:
        required_imports = ["import torch", "from typing import Tuple"]
    patterns = [
        r"'''python(.*?)'''",
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r"'''(.*?)'''",
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


# Test LLM API
if __name__ == '__main__':
    string = """```python
import torch
from typing import Dict, Tuple

@torch.jit.script
def compute_reward(
    x_velocity: torch.Tensor,
    action: torch.Tensor,
    z: torch.Tensor,
    angle: torch.Tensor,
    _forward_reward_weight: float,
    _ctrl_cost_weight: float,
    _healthy_reward: float,
    _terminate_when_unhealthy: bool,
    _healthy_z_range: torch.Tensor,      # shape [2]: [min_z, max_z]
    _healthy_angle_range: torch.Tensor,  # shape [2]: [min_angle, max_angle]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Ensure we use a consistent device/dtype for all created tensors
    device = x_velocity.device
    dtype = x_velocity.dtype

    # Temperatures for transformed components (must be local named variables, not inputs)
    vel_temp = torch.tensor(1.2, device=device, dtype=dtype)      # scales tanh sharpness for forward speed
    angle_temp = torch.tensor(0.5, device=device, dtype=dtype)    # scales exp sensitivity for uprightness
    height_temp = torch.tensor(0.05, device=device, dtype=dtype)  # scales exp sensitivity for height band

    # Weights for shaping terms (local tensors for device alignment)
    upright_w = torch.tensor(0.2, device=device, dtype=dtype)
    height_w = torch.tensor(0.1, device=device, dtype=dtype)

    # Unpack healthy ranges
    min_z = _healthy_z_range[0]
    max_z = _healthy_z_range[1]
    min_angle = _healthy_angle_range[0]
    max_angle = _healthy_angle_range[1]

    # Forward progress: raw and shaped
    r_forward_raw = _forward_reward_weight * x_velocity
    r_forward_tanh = _forward_reward_weight * torch.tanh(x_velocity / vel_temp)

    # Healthy bonus following env logic
    # healthy_mask is 1 if within both z and angle limits, else 0
    healthy_mask = ((z > min_z) & (z < max_z) & (angle > min_angle) & (angle < max_angle)).to(dtype)
    healthy_full = torch.tensor(_healthy_reward, device=device, dtype=dtype).expand_as(healthy_mask)
    r_healthy = healthy_full * healthy_mask
    # If environment does not terminate when unhealthy, typically healthy reward is always provided
    if not _terminate_when_unhealthy:
        r_healthy = healthy_full

    # Upright shaping: large when |angle| is small
    r_upright = torch.exp(-torch.abs(angle) / angle_temp)

    # Height shaping: large when z is inside or near the healthy band; softly penalize outside
    dev_low = torch.clamp(min_z - z, min=0.0)
    dev_high = torch.clamp(z - max_z, min=0.0)
    z_deviation = dev_low + dev_high
    r_height = torch.exp(-z_deviation / height_temp)

    # Control cost
    r_ctrl_cost = _ctrl_cost_weight * torch.sum(action * action, dim=-1)

    # Total reward composition
    total_reward = r_forward_tanh + r_healthy + upright_w * r_upright + height_w * r_height - r_ctrl_cost

    # Components dictionary (all tensors)
    components: Dict[str, torch.Tensor] = {
        "forward_raw": r_forward_raw,
        "forward_tanh": r_forward_tanh,
        "healthy_bonus": r_healthy,
        "upright_bonus": upright_w * r_upright,
        "height_bonus": height_w * r_height,
        "control_cost": r_ctrl_cost,
        "total": total_reward,
    }
    return total_reward, components
```"""
    ph = return_reward_string_placeholder("empty")
    parsed_funcs = parse_all_code_blocks(string)
    print(parsed_funcs)

    """chat = ChatSession()
    print(chat.ask("Hello, who won the 2024 World Cup in soccer?"))
    print(chat.ask("And who was the top scorer?"))"""
