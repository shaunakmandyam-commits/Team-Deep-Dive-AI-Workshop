import gymnasium as gym
from stable_baselines3 import PPO
from PlaneEnv import PlaneEnv
from TwoPlanesEnv import TwoPlanesEnv

# 1. Initialize the environment with visual rendering enabled
env = PlaneEnv(render_mode="human", dt=1, speed=60)

# 2. Load your saved PPO model and bind it to the environment
# (Do not instantiate a new PPO model first; call load directly on the class)
model = PPO.load("plane_agent_8500000_steps", env=env)

# 3. Run evaluation loops
episodes = 10
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    score = 0
    steps=0

    while not done:
        # Pass deterministic=True to exploit the learned policy without random exploration
        action, _states = model.predict(obs, deterministic=True)
        
        # Step through the environment (render_mode="human" automatically displays the window)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        steps+=1

    print(f"Episode {episode + 1} Cleared. Total Score: {score:.2f}", steps, env.sim.plane.speed)

# 4. Clean up resources and close the rendering window
env.close()
