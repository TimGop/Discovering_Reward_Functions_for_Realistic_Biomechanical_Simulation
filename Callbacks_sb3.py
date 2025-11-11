from stable_baselines3.common.callbacks import BaseCallback


class StatsCallback(BaseCallback):
    """
    A custom callback to gather episodic statistics.
    This replaces the need to parse a monitor.csv file.
    """

    def __init__(self, verbose=0):
        super(StatsCallback, self).__init__(verbose)
        self.episode_stats = []
        # Find all keys that RecordEpisodeStatistics will log
        self.stat_keys = [
            "r", "l", "t", "fitness_score", "episode_length"
        ]
        self.reward_component_keys = []

    def _on_step(self) -> bool:
        # Check if any environments just finished
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # Get the info dict for this env
                info = self.locals["infos"][i]

                # Check if "episode" stats exist (added by RecordEpisodeStatistics)
                if "episode" in info:

                    # This is the info from RecordEpisodeStatistics
                    episode_data = info["episode"]

                    # Start a new stats dict with the basics
                    stats = {
                        "r": episode_data["r"],
                        "l": episode_data["l"],
                        "t": episode_data["t"]
                    }

                    # --- THIS IS THE FIX ---
                    # Now, look for ALL our custom stats in the TOP-LEVEL info dict

                    if "fitness_score" in info:
                        stats["fitness_score"] = info["fitness_score"]

                    if "episode_length" in info:
                        stats["episode_length"] = info["episode_length"]

                    # Find and add all flattened reward components
                    for key in info.keys():
                        if key.startswith("reward_components/"):
                            stats[key] = info[key]

                    self.episode_stats.append(stats)
        return True
