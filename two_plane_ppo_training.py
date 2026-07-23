import math
import numpy as np
from TwoPlanesEnv import TwoPlanesEnv
env = TwoPlanesEnv()
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
check_env(env)

model = PPO.load("two_plane_agent",
                     env=env,
                     policy="MultiInputPolicy",
                     verbose=1,
                     learning_rate=0.0003,
                     ent_coef=0.001)

model = PPO(policy="MultiInputPolicy",env=env, verbose=1)
steps = 5_000_000
save = 50_000
for i in range(steps // save):
    model.learn(total_timesteps=save, 
                tb_log_name="two_plane_log",
                reset_num_timesteps=False)
    model.save("two_plane_agent")