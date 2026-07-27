import gymnasium as gym
from OnePlaneMultiAgent import OnePlaneMultiAgent
from TwoPlaneMultiAgent import TwoPlaneMultiAgent
import numpy as np
import os
import torch
from pprint import pprint

env = TwoPlaneMultiAgent(render_mode="human", dt=1, speed=120)


import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

path = os.path.abspath("models/separate_plane_policy")


ray.init(ignore_reinit_error=True)

algo = PPO.from_checkpoint(path)

episodes = 100
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    score = 0
    steps = 0
    pprint(obs)
    
    while not done:
        # Pass deterministic=True to exploit the learned policy without random exploration
        # For multi-agent or single-policy evaluation via algorithm instance:
        # If obs is a dictionary (multi-agent), extract or compute per agent ID:
        

        module = algo.get_module("plane_policy")

        actions = {}

        for agent_id, agent_obs in obs.items():

            obs_tensor = torch.tensor(
                agent_obs,
                dtype=torch.float32
            ).unsqueeze(0)

            output = module.forward_inference(
                {
                    "obs": obs_tensor
                }
            )

            dist_inputs = output["action_dist_inputs"]

            dist_cls = module.get_inference_action_dist_cls()

            dist = dist_cls.from_logits(dist_inputs)

            action = dist.to_deterministic().sample()

            action = action.detach().cpu().numpy().squeeze(0)

            actions[agent_id] =  np.clip(action, -1, 1) 
        #print(actions)
        
        # Step through the environment (render_mode="human" automatically displays the window)
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Handle multi-agent vs single-agent reward/done aggregations if needed
        done = terminated["__all__"] or truncated["__all__"]
        score += sum(reward.values()) if isinstance(reward, dict) else reward
        
            
        steps += 1

    print(f"Episode {episode + 1} Cleared. Total Score: {score:.2f}, Steps: {steps}")

# 4. Clean up resources and close the rendering window
env.close()
ray.shutdown()