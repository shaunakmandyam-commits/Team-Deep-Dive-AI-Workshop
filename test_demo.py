import gymnasium as gym
from OnePlaneMultiAgent import OnePlaneMultiAgent
from TwoPlaneMultiAgent import TwoPlaneMultiAgent
import numpy as np
import os
import torch
from pprint import pprint
from save_simulation import sim_human, sim_rgb



import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

path = os.path.abspath("models/separate_plane_policy")
ray.init()
algo = PPO.from_checkpoint(path)

sim_human(TwoPlaneMultiAgent, algo, episodes=100, difficulty=6, speed=120)
"""
video = sim_rgb(TwoPlaneMultiAgent, algo, episodes=5, difficulty=6)
folder = "videos"
file = "tuff_vid"
full_path = os.path.join(folder, file)
np.save(full_path, video)
print("video saved in", file)
"""