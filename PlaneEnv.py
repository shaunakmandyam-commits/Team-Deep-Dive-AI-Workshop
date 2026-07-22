import math
import pygame
import gymnasium as gym
import numpy as np
from Plane import Plane, Airport
from PlaneSim import PlaneSim




class PlaneEnv(gym.Env):
    scale = 40/800
    collision_distance=1
    max_seconds=600
    command_interval=60

    def __init__(self, dt=1, render_mode = None, speed=1):
        self.sim = PlaneSim(self.scale, dt)
        self.dt = dt
        self.steps = 0
        self.render_mode = render_mode
        self.window=None
        self.clock=None
        self.speed=speed

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

    @property
    def max_steps(self):
        return int(self.max_seconds / self.dt)

    @property 
    def frequency(self):
        return int(self.command_interval / self.dt)

    def observe(self):
        self.sim.plane.heading %= 360
        rad = np.deg2rad(self.sim.plane.heading)
        dx = self.sim.plane.x - self.sim.airport.x
        dy = self.sim.plane.y - self.sim.airport.y

        radians = np.arctan2(dy, dx)
        
        degrees = np.degrees(radians)
        
        deg = (degrees + 360) % 360
        bearing_error = (deg - self.sim.plane.heading + 180) % 360 - 180
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
        
        for i in range(self.frequency):
            self.steps += 1
            self.sim.step()
            if self.render_mode != None:
                self.render()
            if self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y) < self.collision_distance:
                break
        
        new_distance = self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y)
        
        truncated = self.steps > self.max_steps  
        reward = (old_distance - new_distance)
        terminated = bool(self.collision_distance > new_distance)
        if terminated:
            reward += 100
        if truncated:
            reward -= 50
        reward -= 15

        observation = self.observe()
        info = {
            "steps": self.info()
        }

        reward /= 100
        
        return observation, reward, terminated, truncated, info



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = PlaneSim(self.scale, self.dt)

        self.steps = 0
        x = self.np_random.uniform(0, self.sim.x)
        y = self.np_random.uniform(0, self.sim.y)
        speed = self.np_random.uniform(0.05, 0.35)
        heading = self.np_random.integers(0, 360)
        airport_x = self.np_random.uniform(0, self.sim.x)
        airport_y = self.np_random.uniform(0, self.sim.y)

       # margin_x = self.np_random.uniform(0, self.sim.x / 10)
       # margin_y = self.np_random.uniform(0, self.sim.y / 10)
        """
        if np.random.random() > 0.5:
            airport_x = margin_x
        else:
            airport_x = self.sim.x - margin_x

        if np.random.random() > 0.5:
            airport_y = margin_y
        else:
            airport_y = self.sim.y - margin_y
        """

        plane = Plane("agent", "RL", 100, x, y, 0, heading, speed)
        airport = Airport(airport_x, airport_y)
        self.sim.reset(plane, airport)

        observation = self.observe()
        info = {
            "steps": self.info()
        }

        return observation, info
    
    

    def render(self):

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.sim.width, self.sim.height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        screen_plane_x, screen_plane_y = self.world_to_screen(self.sim.plane.x, self.sim.plane.y)
        screen_airport_x, screen_airport_y = self.world_to_screen(self.sim.airport.x, self.sim.airport.y)

        canvas = pygame.Surface((self.sim.width, self.sim.height))
        canvas.fill((255, 255, 255))
        pygame.draw.circle(canvas, (255, 0, 0), (screen_plane_x, screen_plane_y), 5)
        pygame.draw.circle(canvas, (0, 255, 0), (screen_airport_x, screen_airport_y), 5)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(1 / self.dt * self.speed)
    
    def world_to_screen(self,x, y):
        screen_x = (x / self.sim.x) * self.sim.width
        screen_y = ((self.sim.y - y) / self.sim.y) * self.sim.height
        return screen_x, screen_y

