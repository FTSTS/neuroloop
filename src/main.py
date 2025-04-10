"""
Forced Temporal Spike-Time Stimulation (FTSTS).
"""

import gymnasium as gym
from stable_baselines3 import PPO
import dbs_env


def main() -> None:
    """Example usage of the DBSEnv with a PPO agent."""

    env = gym.make('dbs_env/DBS-v0')

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_dbs")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_dbs")

    obs, _ = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(f"action: {action},\tstate: {obs},\treward: {reward}")


if __name__ == '__main__':
    main()
