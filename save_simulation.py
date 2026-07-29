import torch
import numpy as np
import random

def sim_rgb(env_class, algo, episodes=1, dt=1, speed=1, difficulty=0.1, environment = None):
    env = env_class(render_mode="rgb", dt=dt, speed=speed, difficulty = difficulty)
    videos = []
    if environment != None:
        environment.render_mode = "rgb"
        env = environment
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:

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
    
                actions[agent_id] = np.clip(np.array([output["action_dist_inputs"][0][0]]),-1, 1)
                    
            
            
            obs, reward, terminated, truncated, info = env.step(actions)
            
            done = terminated["__all__"] or truncated["__all__"]
            score += sum(reward.values()) 
            
                
            steps += 1

        print(env.get_images().shape)
        videos.append(env.get_images())
    return np.concatenate(videos, axis=0)





def sim_human(env_class, algo, episodes=1, dt=1, speed=1, difficulty=8.5, environment = None, show_actions = False, show_obs = False):
    env = env_class(render_mode="human", dt=dt, speed=speed, difficulty = difficulty)
    if environment != None:
        environment.render_mode = "human"
        env = environment
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        score = 0
        steps = 0
        if show_obs: print(obs)
        while not done:

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
    
                actions[agent_id] = np.clip(np.array([output["action_dist_inputs"][0][0]]),-1, 1)
            if show_actions: print(actions[agent_id])  
            
            
            obs, reward, terminated, truncated, info = env.step(actions)
            
            done = terminated["__all__"] or truncated["__all__"]
            score += sum(reward.values()) 
            
                
            steps += 1

        print(f"Episode {episode + 1} Cleared. Total Score: {score:.2f}, Steps: {steps}")
