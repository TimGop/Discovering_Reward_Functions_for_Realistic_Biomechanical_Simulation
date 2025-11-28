import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize


class AutoRecordStatsCallback(BaseCallback):

    def __init__(self, env_creator, video_folder: str, verbose=0):
        super(AutoRecordStatsCallback, self).__init__(verbose)
        self.episode_stats = []
        self.stat_keys = ["r", "l", "t", "fitness_score", "episode_length"]

        self.env_creator = env_creator  # Save the recipe, not the cake
        self.video_folder = video_folder
        self.best_fitness = -float('inf')

        os.makedirs(self.video_folder, exist_ok=True)

    def _record_eval_episode(self, step, score):
        # This ensures the OpenGL context is fresh and hasn't been closed by previous runs
        eval_env = self.env_creator()

        # SYNC STATS
        training_env = self.model.get_env()
        if isinstance(training_env, VecNormalize):
            # We must sync stats to the new fresh env
            # Note: We assume the creator returns a VecNormalize wrapper
            eval_env.obs_rms = training_env.obs_rms.copy()
            eval_env.ret_rms = training_env.ret_rms.copy()  # Good practice to sync returns too

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

                # --- Your Existing Stats Logic ---
                if "episode" in info:
                    episode_data = info["episode"]
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
