import gymnasium as gym
from OnePlaneMultiAgent import OnePlaneMultiAgent
import numpy as np
import os
import torch
from pprint import pprint
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from save_simulation import sim_human, sim_rgb

env = OnePlaneMultiAgent(render_mode="human", dt=1, speed=120)


path = os.path.abspath("models/fly_towards_airport_policy")
algo = PPO.from_checkpoint(path)

sim_human(OnePlaneMultiAgent, algo, episodes=100, speed=120, difficulty=0.4)

