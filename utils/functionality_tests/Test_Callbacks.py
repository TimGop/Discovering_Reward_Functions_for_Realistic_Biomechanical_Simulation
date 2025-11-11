import math
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

# Define a custom RLlibCallback.

class LogAcrobotAngle(RLlibCallback):

    def on_episode_created(self, *, episode, **kwargs):
        # Initialize an empty list in the `custom_data` property of `episode`.
        episode.custom_data["theta1"] = []

    def on_episode_step(self, *, episode, env, **kwargs):
        # Compute the angle at every episode step and store it temporarily in episode:
        state = env.envs[0].unwrapped.state
        deg_theta1 = math.degrees(math.atan2(state[1], state[0]))
        episode.custom_data["theta1"].append(deg_theta1)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        theta1s = episode.custom_data["theta1"]
        avg_theta1 = np.mean(theta1s)

        # Log the resulting average angle - per episode - to the MetricsLogger.
        # Report with a sliding window of 50.
        metrics_logger.log_value("theta1_mean", avg_theta1, reduce="mean", window=50)

config = (
    PPOConfig()
    .environment("Acrobot-v1")
    .callbacks(
        callbacks_class=LogAcrobotAngle,
    )
)
ppo = config.build()

# Train n times. Expect `theta1_mean` to be found in the results under:
# `env_runners/theta1_mean`
for i in range(30):
    results = ppo.train()
    print(
        f"iter={i} "
        f"theta1_mean={results['env_runners']['theta1_mean']} "
        f"R={results['env_runners']['episode_return_mean']}"
    )