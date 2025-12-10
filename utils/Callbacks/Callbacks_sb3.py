import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize


class AutoRecordStatsCallback(BaseCallback):

    def __init__(self, env_creator, video_folder: str, verbose=0):
        super(AutoRecordStatsCallback, self).__init__(verbose)
        self.episode_stats = []
        self.stat_keys = ["r", "l", "t", "fitness_score", "episode_length"]

        self.env_creator = env_creator
        self.video_folder = video_folder
        self.best_fitness = -float('inf')

        self.total_episodes = 0
        self.consecutive_short_episodes = 0

        os.makedirs(self.video_folder, exist_ok=True)

    def _record_eval_episode(self, step, score):
        eval_env = self.env_creator()
        training_env = self.model.get_env()
        if isinstance(training_env, VecNormalize):
            eval_env.obs_rms = training_env.obs_rms.copy()
            eval_env.ret_rms = training_env.ret_rms.copy()

        video_name = f"best_fitness"
        recorder = VecVideoRecorder(
            eval_env,
            self.video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=1000,
            name_prefix=video_name
        )

        obs = recorder.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = recorder.step(action)
            done = dones[0]
        recorder.close()
        if self.verbose > 0:
            print(f"New best score {score:.2f}! Video recorded at step {step}")

    def _on_step(self) -> bool:
        # check if any environments just finished
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]

                if "episode" in info:
                    self.total_episodes += 1

                    episode_data = info["episode"]
                    episode_len = episode_data["l"]

                    # Only check logic if we have passed the 7000th episode
                    if self.total_episodes > 2000:
                        if episode_len < 30:
                            self.consecutive_short_episodes += 1
                            if self.verbose > 0:
                                print(
                                    f"Short episode detected ({episode_len} steps)."
                                    f" Consecutive count: {self.consecutive_short_episodes}")
                        else:
                            self.consecutive_short_episodes = 0

                        if self.consecutive_short_episodes >= 10:
                            if self.verbose > 0:
                                print("Stopping training: 10 consecutive episodes < 30 steps detected after 7000 "
                                      "episodes.")
                            return False  # stops the training

                    stats = {
                        "r": episode_data["r"],
                        "l": episode_data["l"],
                        "t": episode_data["t"]
                    }

                    current_fitness = -float('inf')

                    if "fitness_score" in info:
                        stats["fitness_score"] = info["fitness_score"]
                        current_fitness = info["fitness_score"]

                    if "episode_length" in info:
                        stats["episode_length"] = info["episode_length"]

                    for key in info.keys():
                        if key.startswith("reward_components/"):
                            stats[key] = info[key]

                    self.episode_stats.append(stats)

                    if current_fitness > self.best_fitness:
                        self.best_fitness = current_fitness
                        self._record_eval_episode(self.num_timesteps, current_fitness)

        return True
