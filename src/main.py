"""
Closed-Loop DBS via RL of Forced Temporal Spike-Time Stimulation (FTSTS).

Example usage of the DBS environment with a PPO agent in both sequential
and parallel environments.
"""

import time
import dbs_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def run_sequential():
    """Example usage of the DBSEnv with a PPO agent."""

    print("Running in sequential mode...")

    # Create the environment.
    env = gym.make('dbs_env/DBS-v0')

    # Create the PPO model.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
    )

    # Train the model.
    start = time.time()
    model.learn(total_timesteps=10000)
    end = time.time()
    print(f"Training took {end - start:.4f} seconds.")

    # Save and load the model.
    model.save("ppo_dbs")
    del model  # remove to demonstrate saving and loading
    model = PPO.load("ppo_dbs")

    # Test the trained model.
    print("\n\nTesting the trained model...\n")
    obs, _ = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, _, _, _ = env.step(action)
        print(f"\naction: {action},\nstate: {obs},\nreward: {reward}")


def run_parallel():
    """Example vectorized usage of the DBSEnv with a PPO agent."""

    print("Running in parallel mode...")

    # Parallel Environments.
    vec_env = make_vec_env(
        env_id='dbs_env/DBS-v0',
        n_envs=10,
        env_kwargs={'render_mode': None}
    )

    # Create the PPO model.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
    )

    # Train the model.
    start = time.time()
    model.learn(total_timesteps=0)
    end = time.time()
    print(f"Training took {end - start:.4f} seconds.")

    # Save and load the model.
    model.save("ppo_dbs")
    del model  # remove to demonstrate saving and loading
    model = PPO.load("ppo_dbs")

    # Test the trained model.
    print("\n\nTesting the trained model...\n")
    done = False
    obs = vec_env.reset()
    while not done:
        actions, _ = model.predict(obs)
        obs, rewards, dones, _ = vec_env.step(actions)
        print(f"\naction: {actions},\nstate: {obs},\nreward: {rewards}")

        done = dones.any()


if __name__ == '__main__':
    # run_sequential()
    run_parallel()
