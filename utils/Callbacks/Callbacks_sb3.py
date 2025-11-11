from stable_baselines3.common.callbacks import BaseCallback


class StatsCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(StatsCallback, self).__init__(verbose)
        self.episode_stats = []
        self.stat_keys = [
            "r", "l", "t", "fitness_score", "episode_length"
        ]
        self.reward_component_keys = []

    def _on_step(self) -> bool:
        # check if any environments just finished
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # get the info dict for this env
                info = self.locals["infos"][i]

                # check if "episode" stats exist (added by RecordEpisodeStatistics)
                if "episode" in info:
                    episode_data = info["episode"]
                    stats = {
                        "r": episode_data["r"],
                        "l": episode_data["l"],
                        "t": episode_data["t"]
                    }

                    # custom stats in the TOP-LEVEL info dict
                    if "fitness_score" in info:
                        stats["fitness_score"] = info["fitness_score"]

                    if "episode_length" in info:
                        stats["episode_length"] = info["episode_length"]

                    # add all flattened reward components
                    for key in info.keys():
                        if key.startswith("reward_components/"):
                            stats[key] = info[key]

                    self.episode_stats.append(stats)
        return True
