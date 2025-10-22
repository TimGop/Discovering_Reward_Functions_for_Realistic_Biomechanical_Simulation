init_sys_prompt = """You are a reward engineer trying to write reward functions to solve reinforcement learning
tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the
task described in text.
Your reward function should use useful variables from the environment as inputs. As an example
the reward function signature can be:
‘‘‘
@torch.jit.script
def custom_reward_fn(observation, action, original_reward, terminated, truncated)
 -> Tuple[torch.Tensor,Dict[str, torch.Tensor]]:
...
return reward, {}‘‘‘
Since the reward function will be decorated with @torch.jit.script,
please make sure that the code is compatible with TorchScript (e.g., use torch tensor instead
of numpy array). Crucially, you must add explicit type hints to any empty dictionaries or lists.
For example: info: Dict[str, torch.Tensor] = {}.
Make sure any new tensor or variable you introduce is on the same device as the input tensors."""
reward_func_context = """Note that the reward function will be called in the following context in a wrapper class that 
overrides the original step method of an environment to implement its own custom reward function:
    def step(self, action):
        observation_np, original_reward_np, terminated_np, truncated_np, info = self.env.step(action)
        observation = torch.as_tensor(observation_np, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.float32)
        original_reward = torch.as_tensor(original_reward_np, dtype=torch.float32)
        terminated = torch.as_tensor(terminated_np, dtype=torch.bool)
        truncated = torch.as_tensor(truncated_np, dtype=torch.bool)
        # Calculate new reward
        new_reward_tuple = self.custom_reward_fn(
            observation, action_tensor, original_reward, terminated, truncated
        )
        new_reward = new_reward_tuple[0]
        info["custom_policy_id"] = self.policy_id
        return observation_np, new_reward.cpu().numpy(), terminated_np, truncated_np, info"""
pre_env = """Below is the code for the environment:"""
code_formatting_tip = """The output of a reward function should consist of two items:
(1) the total reward,
(2) a dictionary of each individual reward component.
The code output for each reward function should be formatted as a separate python code string: "‘‘‘python ... ‘‘‘".
Some helpful tips for writing the reward function code:
(1) You may find it helpful to normalize the reward to a fixed range by applying
transformations like torch.exp to the overall reward or its components
(2) If you choose to transform a reward component, then you must also introduce a
temperature parameter inside the transformation function; this parameter must be a named
variable in the reward function and it must not be an input variable. Each transformed
reward component should have its own temperature variable
(3) Make sure the type of each input variable is correctly specified; a float input
variable should not be specified as torch.Tensor
(4) Most importantly, the reward code’s input variables must contain only attributes of
the provided environment class definition (namely, variables that have prefix self.).
Under no circumstance can you introduce new input variables.
"""
code_formatting_tip_bonus = """When generating the reward function, follow this critical rule for numerical stability: 
The final reward must NOT be the sum of the original_reward and a function of the original_reward 
(i.e., avoid reward = base + f(base)). Instead, use one of these two stable patterns:
Reward Replacement: Create a new reward value by applying a bounded function to the original reward 
(e.g., final_reward = torch.tanh(original_reward) + action_penalty).
Independent Addition: Add or subtract terms that depend on the action or observation, not the original_reward itself 
(e.g., final_reward = original_reward - 0.01 * torch.norm(action))."""
num_rew_funcs_directive = """Please design 16 seperate reward functions."""
rew_reflection_1 = """We trained a RL policy using the provided reward function code and tracked the values of the
individual components in the reward function as well as global policy metrics such as
success rates and episode lengths after every {epoch_freq} epochs and the maximum, mean,
minimum values encountered:"""
rew_reflection_2 = """Please carefully analyze the policy feedback and provide a new, improved reward function that
can better solve the task. Some helpful tips for analyzing the policy feedback:
(1) If the success rates are always near zero, then you must rewrite the entire reward
function
(2) If the values for a certain reward component are near identical throughout, then this
means RL is not able to optimize this component as it is written. You may consider
(a) Changing its scale or the value of its temperature parameter
(b) Re-writing the reward component
(c) Discarding the reward component
(3) If some reward components’ magnitude is significantly larger, then you must re-scale
its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then
write the reward function code.
"""

walker_2d_v4_description = """The walker is a two-dimensional two-legged figure that consist of seven main body parts 
- a single torso at the top (with the two legs splitting after the torso), two thighs in the middle below the torso, 
two legs in the bottom below the thighs, and two feet attached to the legs on which the entire body rests. The goal 
is to walk in the in the forward (right) direction by applying torques on the six hinges connecting the seven body 
parts."""
