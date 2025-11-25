import gymnasium as gym
from sbx import SAC
import os
import sys

# --- 1. FIX: Force JAX to use CPU to prevent Windows 0xC0000005 Crash ---
# This must happen before any other JAX/SBX operations.
# The crash is often caused by JAX trying to use the GPU while MuJoCo
# is trying to render on the same GPU context.
try:
    import jax

    jax.config.update("jax_platform_name", "cpu")
except ImportError:
    pass  # SBX will fail later if jax isn't there, but this is just config.

# --- Constants ---
ENV_ID = "Humanoid-v5"
MODEL_PATH = "sac_Humanoid-v5_0_2048000_1764069115.zip"


def run_test():
    # --- Debug System Info ---
    print(f"Current Python Version: {sys.version}", flush=True)
    print("JAX configured to use CPU for stability.", flush=True)

    # --- 2. Setup Environment ---
    print(f"Creating environment {ENV_ID} with human render mode...", flush=True)
    try:
        env = gym.make(ENV_ID)  # render_mode="human"
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # --- 3. Load the Agent ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...", flush=True)

    try:
        # We pass buffer_size=1 to save memory
        model = SAC.load(MODEL_PATH, env=env, buffer_size=1)
        print("Model loaded successfully.", flush=True)
    except ValueError as e:
        print("\nCRITICAL ERROR - PYTHON VERSION MISMATCH:", flush=True)
        print(f"Error details: {e}")
        return
    except Exception as e:
        print("\nCRITICAL ERROR LOADING MODEL:", flush=True)
        print(e)
        return

    # --- 4. Evaluation Loop ---
    episodes = 5
    print(f"Running {episodes} evaluation episodes...", flush=True)

    for i in range(episodes):
        print(f"Starting Episode {i + 1}...", end=" ", flush=True)
        obs, _ = env.reset()
        terminated = False
        truncated = False

        total_reward = 0
        steps = 0
        total_x_velocity = 0.0

        while not terminated and not truncated:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if 'x_velocity' in info:
                total_x_velocity += info['x_velocity']

        avg_speed = total_x_velocity / steps if steps > 0 else 0.0
        print(f"Done. Reward: {total_reward:.2f} | Avg Speed: {avg_speed:.4f}", flush=True)

    print("Evaluation finished.", flush=True)
    env.close()


if __name__ == "__main__":
    run_test()