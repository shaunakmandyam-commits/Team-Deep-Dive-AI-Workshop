import math
import numpy as np
from PlaneEnv import PlaneEnv
from TwoPlanesEnv import TwoPlanesEnv
env = PlaneEnv()
from stable_baselines3.common.env_checker import check_env
check_env(env)

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


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint = 100000
checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint,
    save_path= "./Team-Deep-Dive-AI-Workshop/",
    name_prefix="plane_agent"


)

model = PPO.load("plane_agent", 
                 env=env,
                 policy="MlpPolicy",
                 verbose=1,
                 learning_rate=0.0001,
                 ent_coef=0.005
)

"""
model = PPO(
    "MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0003,
    ent_coef=0.001
)
"""
model.learn(total_timesteps=500_000, tb_log_name="plane_agent")
model.save("plane_agent")
