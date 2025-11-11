import ray
import random
from ray import tune
from ray.tune import sample_from
from ray.tune.schedulers import PopulationBasedTraining


# Postprocess the perturbed config to ensure it's still valid (from your script)
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config


def tune_rllib_policy(
        env_id: str,
        max_timesteps: int = 3_000_000,
        hidden_layers: list[int] = None,
        num_samples: int = 4,  # How many parallel trials to run
):
    horizon = 1000
    if not hidden_layers:
        hidden_layers = [64, 64]

    ray.init(ignore_reinit_error=True)

    # --- Scheduler Config (from your PBT example) ---
    # We use num_env_steps_sampled_lifetime as the time_attr to match Script 1's logging
    scheduler = PopulationBasedTraining(
        time_attr="num_env_steps_sampled_lifetime",
        metric="env_runners/episode_return_mean",
        mode="max",
        perturbation_interval=100_000,  # Perturb every 100k timesteps
        resample_probability=0.25,
        quantile_fraction=0.25,
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.1, 0.5),
            "lr": lambda: random.uniform(1e-5, 1e-3),
            # Search space from your script, centered around Script 1's 8192
            "train_batch_size": lambda: random.randint(1000, 60000),
        },
        custom_explore_fn=explore,
    )

    # --- Base Config (combining fixed values from Script 1) ---
    config = {
        # --- Environment & Framework ---
        "env": env_id,
        "framework": "torch",

        # --- Env Runners (Workers) from Script 1 ---
        "horizon": horizon,
        "num_workers": 8,
        "num_envs_per_env_runner": 1,
        "rollout_fragment_length": 'auto',
        "observation_filter": "MeanStdFilter",

        # --- Model (from Script 1) ---
        "model": {
            "fcnet_hiddens": hidden_layers,
            "fcnet_activation": "tanh",
            "vf_share_layers": True,
        },

        # --- Fixed Training Hyperparams (from Script 1) ---
        "sgd_minibatch_size": 64,
        "num_sgd_iter": 10,  # Renamed from num_epochs
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.0,
        "grad_clip": 0.5,
        "gamma": 0.99,

        # --- Tuned Hyperparams (using search space from Script 2) ---
        "lambda": tune.sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": tune.sample_from(lambda spec: random.uniform(0.1, 0.5)),
        "lr": tune.sample_from(lambda spec: random.uniform(1e-5, 1e-3)),
        "train_batch_size": tune.sample_from(lambda spec: random.randint(1000, 60000)),

        # --- Resources & Debugging ---
        "num_gpus": 1,
        "seed": 0,
    }

    # --- Run the Tune Experiment ---
    print(f"Starting Tune experiment for {env_id} with {num_samples} samples...")
    analysis = tune.run(
        "PPO",  # The algorithm to train
        name=f"PPO_tune_{env_id}",
        scheduler=scheduler,
        verbose=1,
        num_samples=num_samples,
        stop={"num_env_steps_sampled_lifetime": max_timesteps},
        config=config,
        resume=True
    )

    ray.shutdown()
    print("--- Tuning Complete ---")

    # Print the best result
    best_config = analysis.get_best_config(
        metric="env_runners/episode_return_mean", mode="max"
    )
    print("Best hyperparameters found: ", best_config)

    best_trial = analysis.get_best_trial(
        metric="env_runners/episode_return_mean", mode="max"
    )
    print(f"Best trial final reward: {best_trial.last_result['env_runners']['episode_return_mean']}")


if __name__ == '__main__':
    tune_rllib_policy(env_id="Walker2d-v4")
