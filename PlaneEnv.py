import math

import gymnasium as gym
import numpy as np
from Plane import Plane, Airport
from PlaneSim import PlaneSim




class PlaneEnv(gym.Env):

    def __init__(self, scale=40/800, dt=1, max_seconds=600):
        self.sim = PlaneSim(scale, dt)
        self.dt = dt
        self.max_steps = int(max_seconds / dt)
        self.steps = 0

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf,-np.inf,0,-1,-1,-1, 0,0], dtype=np.float32),
            high=np.array([
                np.inf,
                np.inf,
                1.0,
                1,
                1,
                1,
                1,
                1
            ], dtype=np.float32),
        )
        
        self.action_space = gym.spaces.Box(low=np.array([-1], dtype=np.float32),
                                           high=np.array([1], dtype=np.float32),
                                           dtype=np.float32)

    def observe(self):
        self.sim.plane.heading %= 360
        rad = np.deg2rad(self.sim.plane.heading)
        dx = self.sim.plane.x - self.sim.airport.x
        dy = self.sim.plane.y - self.sim.airport.y

        # Get angle in radians (-pi to pi)
        radians = np.arctan2(dy, dx)
        
        # Convert to degrees (-180 to 180)
        degrees = np.degrees(radians)
        
        # Map to 0-360 range
        deg = (degrees + 360) % 360
        bearing_error = (deg - self.sim.plane.heading + 180) % 360 - 180
       # print(type(rad), rad)
        return np.array([
            self.sim.plane.x / self.sim.x,
            self.sim.plane.y / self.sim.y,
            self.sim.plane.speed,
            np.sin(rad),
            np.cos(rad),
            bearing_error / 180,
            self.sim.airport.x / self.sim.x,
            self.sim.airport.y / self.sim.y
        ], dtype=np.float32)

    def info(self):
        return self.steps
    
    def step(self, action):
        heading = float(action[0]) * 180
        heading = (self.sim.plane.heading + heading) % 360
        self.sim.act(heading)

        old_distance = self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y)

        frequency = int(60 / self.dt)
        for i in range(0, frequency):
            self.steps += 1
            self.sim.step()
            if self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y) < 2:
                break
        
        new_distance = self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y)

        
        truncated = self.steps > self.max_steps  
        reward = (old_distance - new_distance)
        terminated = bool(2 > new_distance)
        if terminated:
            reward += 100
        if truncated:
            reward -= 50
        reward -= 10

        observation = self.observe()
        info = {
            "steps": self.info()
        }

        reward /= 100
        
        return observation, reward, terminated, truncated, info



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        x = self.sim.x / 2
        y = self.sim.y / 2
        speed = 0.2
        heading = self.np_random.integers(0, 360)
        
        margin_x = self.np_random.integers(0, self.sim.x / 10)
        margin_y = self.np_random.integers(0, self.sim.y / 10)

        if np.random.random() > 0.5:
            airport_x = margin_x
        else:
            airport_x = self.sim.x - margin_x

        if np.random.random() > 0.5:
            airport_y = margin_y
        else:
            airport_y = self.sim.y - margin_y
        

        plane = Plane("agent", "RL", 100, x, y, 0, heading, speed)
        airport = Airport(airport_x, airport_y)
        self.sim.reset(plane, airport)

        observation = self.observe()
        info = {
            "steps": self.info()
        }

        return observation, info
    
    

    def render(self):
        pass
