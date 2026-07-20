import math
import pygame
import gymnasium as gym
import numpy as np
from Plane import Plane, Airport
from PlaneSim import PlaneSim




class PlaneEnv(gym.Env):

    def __init__(self, scale=40/800, dt=1, max_seconds=600, render_mode = None):
        self.sim = PlaneSim(scale, dt)
        self.dt = dt
        self.max_steps = int(max_seconds / dt)
        self.steps = 0
        self.render_mode = render_mode
        self.window=None
        self.clock=None

        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf,-np.inf,0,-1,-1, 0,0], dtype=np.float32),
            high=np.array([
                np.inf,
                np.inf,
                1.0,
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
       # print(type(rad), rad)
        return np.array([
            self.sim.plane.x / self.sim.x,
            self.sim.plane.y / self.sim.y,
            self.sim.plane.speed,
            np.sin(rad),
            np.cos(rad),
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
            if self.render_mode != None:
                self.render()
            if self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y) < 2:
                break
        
        new_distance = self.sim.plane.distance(self.sim.airport.x, self.sim.airport.y)

        
        truncated = self.steps > self.max_steps  
        reward = old_distance - new_distance
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
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            #self.clock.tick(1 / self.dt * 25)
    
    def world_to_screen(self,x, y):
        screen_x = (x / self.sim.x) * self.sim.width
        screen_y = ((self.sim.y - y) / self.sim.y) * self.sim.height
        return screen_x, screen_y

