import math
import numpy as np
from PlaneEnv import PlaneEnv
env = PlaneEnv()
from stable_baselines3.common.env_checker import check_env
check_env(env)

from stable_baselines3 import PPO
print('Starting')
"""
env.reset()
env.sim.plane.set_heading(0)
env.step([0.25])
env.sim.plane.x = 0
env.sim.plane.y = 0
env.sim.airport.x = 30
env.sim.airport.y=30
terminated = False

for i in range(10):
    observation, reward, terminated, truncated, info = env.step([0])
   #print(observation, reward, terminated, truncated, info)
"""




model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    ent_coef=0.001,
    tensorboard_log="./tensorboard_logs/"
)

model.learn(total_timesteps=5_000_000, tb_log_name="plane_agent")
model.save("plane_agent")
