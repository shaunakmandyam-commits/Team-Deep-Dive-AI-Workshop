import gymnasium as gym
from stable_baselines3 import PPO
from PlaneEnv import PlaneEnv

# 1. Initialize the environment with visual rendering enabled
env = PlaneEnv(render_mode="human")

# 2. Load your saved PPO model and bind it to the environment
# (Do not instantiate a new PPO model first; call load directly on the class)
model = PPO.load("plane_agent", env=env)

# 3. Run evaluation loops
episodes = 5
for episode in range(episodes):
    obs, info = env.reset()
    done = False
    score = 0
    
    while not done:
        # Pass deterministic=True to exploit the learned policy without random exploration
        action, _states = model.predict(obs, deterministic=True)
        
        # Step through the environment (render_mode="human" automatically displays the window)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        
    print(f"Episode {episode + 1} Cleared. Total Score: {score:.2f}")

# 4. Clean up resources and close the rendering window
env.close()
