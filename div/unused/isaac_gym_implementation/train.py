# In a new file, e.g., 'train.py'
# TODO: POTENTIALLY CONTAINS BUGS AND WILL NOT WORK YET SEE THE RAY.RLLIB IMPLEMENTATION INSTEAD!!!!!!!
import isaacgym
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os

# --- Import our custom classes ---
from multi_reward_walker import MultiRewardWalkerTask
from tensor_reward_fns import walker_less_speed_tensor, walker_more_speed_tensor


def train_isaacgym_multi_policy():
    # 1. Register the custom environment with rl-games
    # This allows the YAML config to find our 'MultiRewardWalker' task
    vecenv.register('MultiRewardWalker',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('MultiRewardWalker', {
        'vecenv_type': 'MultiRewardWalker',
        'env_creator': lambda **kwargs: MultiRewardWalkerTask(**kwargs),
    })

    # 2. Define the list of reward functions you want to use
    reward_functions = [
        walker_less_speed_tensor,
        walker_more_speed_tensor
    ]
    num_policies = len(reward_functions)

    # 3. Load and modify the configuration
    # Using Hydra to manage config files is standard in isaacgymenvs
    initialize(config_path="/")  # Assumes config yaml is in the same directory
    cfg = compose(config_name="config_walker_multi")

    # --- Key modification: Inject our reward functions and update player count ---
    # Convert the config to a dictionary to modify it
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Inject the list of reward functions into the env_config
    cfg_dict['params']['config']['env_config']['reward_fn_list'] = reward_functions

    # Automatically set the number of players/policies based on the list
    cfg_dict['params']['config']['player']['games_num'] = num_policies

    # Ensure num_envs is compatible
    num_envs = cfg_dict['params']['config']['env_config']['num_envs']
    if num_envs % num_policies != 0:
        print(f"Warning: num_envs ({num_envs}) is not divisible by num_policies ({num_policies}).")
        # Adjust num_envs to be a multiple
        new_num_envs = (num_envs // num_policies) * num_policies
        cfg_dict['params']['config']['env_config']['num_envs'] = new_num_envs
        cfg_dict['params']['config']['num_actors'] = new_num_envs
        print(f"Adjusted num_envs to {new_num_envs}.")

    # Convert back to OmegaConf DictConfig
    cfg = OmegaConf.create(cfg_dict)

    # 4. Create and run the rl-games Runner
    runner = Runner(RLGPUAlgoObserver())
    runner.load(cfg)
    runner.run({
        'train': True,
    })


if __name__ == '__main__':
    train_isaacgym_multi_policy()