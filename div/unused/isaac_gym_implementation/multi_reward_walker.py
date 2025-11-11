# In a new file, e.g., 'multi_reward_walker.py'
from fitness_funcs_and_placeholder_reward_funcs_isaac import walker_base_reward
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

# Assuming you have the Humanoid task from isaacgymenvs/tasks/humanoid.py
# If not, you can copy it and adapt it. For simplicity, we'll assume it exists
# and we will override its reward function.
from isaacgymenvs.tasks.humanoid import Humanoid


class MultiRewardWalkerTask(Humanoid):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # This is where we receive the list of reward functions from the training script
        self.reward_fn_list = cfg["env"]["reward_fn_list"]
        self.num_policies = len(self.reward_fn_list)
        print(f"Initializing MultiRewardWalkerTask with {self.num_policies} policies.")

        # The rest of the config is passed to the parent
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # IMPORTANT: Ensure num_envs is divisible by num_policies
        if self.num_envs % self.num_policies != 0:
            raise ValueError("num_envs must be divisible by the number of reward functions (num_policies).")

        envs_per_policy = self.num_envs // self.num_policies

        # Create a tensor that maps each environment index to a policy index
        # e.g., [0, 0, ..., 1, 1, ..., 2, 2, ...]
        self.policy_map = torch.arange(self.num_policies, device=self.device).repeat_interleave(envs_per_policy)

    def compute_reward(self, actions):
        """Overrides the parent's reward function to apply different rewards per policy."""

        # 1. Calculate base reward components common to all policies
        # This part is adapted from the original Humanoid reward logic.
        # You'd replace this with the logic specific to Walker2d.
        keypoint_dist = torch.sum(torch.square(self.key_body_pos - self.key_body_target_pos), dim=-1)

        # The `reset_buf` indicates which envs just terminated.
        # We give them a penalty.
        termination_penalty = -2.0 * self.reset_buf

        base_reward = walker_base_reward(
            self.root_states, self.progress_buf, self.dof_vel, self.dof_effort, termination_penalty
        )

        # 2. Calculate the specific rewards for each policy type
        reward_tensors = []
        for reward_fn in self.reward_fn_list:
            # Each reward function returns a tensor of shape (num_envs,)
            reward_tensors.append(reward_fn(
                base_reward=base_reward,
                root_states=self.root_states,
                # you can pass any other tensor the function might need
            ))

        # Stack rewards into a tensor of shape (num_policies, num_envs)
        stacked_rewards = torch.stack(reward_tensors, dim=0)

        # 3. Use the policy_map to select the correct reward for each environment
        # self.policy_map gives the row index (policy), and torch.arange gives the col index (env)
        final_rewards = stacked_rewards[self.policy_map, torch.arange(self.num_envs, device=self.device)]

        # 4. Store the final rewards in the buffer
        self.rew_buf[:] = final_rewards