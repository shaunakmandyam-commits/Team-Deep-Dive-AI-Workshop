import random
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Sequence
import numpy as np
from Plane import Plane, Airport
from PlaneSim import PlaneSim
from ray.rllib.env.multi_agent_env import MultiAgentEnv

def angle_to_vector(angle):
    rad = np.radians(angle)
    return np.array([np.cos(rad), np.sin(rad)])


class OnePlaneMultiAgent(MultiAgentEnv):
    scale = 40/800
    airport_distance = 0.5
    max_seconds = 600
    command_interval = 30
    call_freq = 2048

    def __init__(self, config=None, dt=1, render_mode=None, speed=1, difficulty = 0.1):
        super().__init__()
        self.dt = dt
        self.steps = 0

        
        self.render_mode = render_mode
        self.window=None
        self.clock=None
        self.speed=speed

        self.images = []

        self.possible_agents = ["plane1"]
        self.agents = []

        self.observation_spaces, self.action_spaces = self._build_spaces()

        self.episodes = 0
        self.call_steps = 0
        self.losses = 0
        self.difficulty = difficulty


    @property
    def max_steps(self):
        return int(self.max_seconds / self.dt)
    @property
    def frequency(self):
        return int(self.command_interval / self.dt)

    def _build_spaces(self):
        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport, 7: time left
        low_agent = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high_agent = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, np.inf, 1.0], dtype=np.float32)


        obs_spaces = {}
        action_spaces = {}
        for agent in self.possible_agents:
            obs_spaces[agent] = Box(low_agent,
                                    high_agent,
                                    dtype=np.float32)
            
            action_spaces[agent] = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        return obs_spaces, action_spaces
    
    def _plane_observation(self, index):
        
        width, height = self.sim.to_scale()

        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport
        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy
        plane = self.sim.planes[index]
        agent = np.array([plane.dp[0] / width, 
                          plane.dp[1] / height, 
                          plane.speed,
                          plane.direction[1],
                          plane.direction[0],
                          plane.heading_error / 180,
                          plane.distance_to_airport / np.sqrt(width ** 2 + height ** 2),
                          (self.max_steps - self.steps) / self.max_steps
                          ], dtype=np.float32)
        
        
        return agent

    def raw_obs(self, index):
        width, height = self.sim.to_scale()
        
        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport
        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy
        plane = self.sim.planes[index]
        agent = {"dx": plane.dp[0], 
                 "dy": plane.dp[1], 
                 "speed": plane.speed,
                 "heading": plane.heading,
                 "heading error": plane.heading_error,
                 "distance": plane.distance_to_airport}
        
        
        return agent
        
        
    def _obs(self):
        observations = {}
        for agent in self.agents:
            observations[agent] = self._plane_observation(self.sim.plane_to_index[agent])
        return observations
    
    def _info(self):
        return {"plane1": {"episodes": self.episodes,
                "successes": self.episodes - self.losses,
                "losses": self.losses,
                "winrate": (self.episodes - self.losses) / (self.episodes + 0.0001),
                "difficulty": self.difficulty,
                "collision distance": self.airport_distance}}

    def _reset_agent(self, plane_angle, agent):
        width, height = self.sim.to_scale()
        small = min(width, height) / 2
        midpoint = np.array([width / 2, height / 2])

        plane_distance = self.np_random.uniform(small - 1, small) * min(self.difficulty * 7, 1)
        airport_distance = self.np_random.uniform(small - 1, small) * min(self.difficulty * 7, 1)

        plane_angle = self.np_random.uniform(0, 360)
        airport_angle = (plane_angle + 180) % 360

        plane_pos = midpoint + plane_distance * angle_to_vector(plane_angle)
        airport_pos = midpoint + airport_distance * angle_to_vector(airport_angle)

        airport_angle += np.clip(self.np_random.normal(0, 60), -180, 180)

        return (Plane(agent, "plane_policy", 100, plane_pos[0], plane_pos[1], heading=airport_angle % 360),
                Airport(airport_pos[0], airport_pos[1]))
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.episodes += 1
        if self.call_steps / self.call_freq > 1:
            self.call()


        self.steps = 0
        self.sim = PlaneSim(self.scale, self.dt)

        self.agents = self.possible_agents

        temp = 0
        for agent in self.agents:
            angle = (self.np_random.uniform(45, 270) + temp) % 360
            temp = angle
            plane, airport = self._reset_agent(angle, agent)
            self.sim.add(plane, airport)

        
        observation = self._obs()
        info = self._info()

        return observation, info

    def step(self, action:dict):
        self.call_steps += 1
        
        for name, heading in action.items():
            plane = self.sim.name_to_plane[name]
            self.sim.act(plane, (plane.heading + heading * 90) % 360)

        old_distance_airports = self.sim.distance_to_airport.copy()

        reward = 0
        terminated = False
        truncated = False
        step_penalty = 0.1

        for i in range(self.frequency):
            reward -= step_penalty / self.frequency
            self.steps += 1
            self.sim.step()

            if self.render_mode == "human":
                self.render()
            elif self.render_mode == "rgb":
                image = self.render()
                self.images.append(image)
            for plane, distance in zip(self.sim.planes, self.sim.distance_to_airport):
                if distance < self.airport_distance:
                    if self.render_mode == "human": print(plane.name, "landed")
                    plane.set_speed(0)
                    plane.landed = True
                    terminated = True
                    reward += 2
                    break
            if terminated:
                break

        

        if self.steps > self.max_steps:
            self.losses += 1
            if self.render_mode == "human": print("timeout")
            truncated = True
            reward -= 2

        
        new_distance_airports = self.sim.distance_to_airport

        reward += np.sum(old_distance_airports - new_distance_airports) / 100
        reward = float(reward)

        rewards={}
        terminateds={}
        truncateds={}

        for agent in self.agents:
            rewards[agent] = reward
            terminateds[agent] = terminated
            truncateds[agent] = truncated

        terminateds["__all__"] = terminated
        truncateds["__all__"] = truncated

        observation = self._obs()
        info = self._info()
        return observation, rewards, terminateds, truncateds, info

    def call(self):
        

        info = self._info()["plane1"]
        for key, val in info.items():
            print(key, val)
        
        self.losses = 0
        self.episodes = 0
        self.call_steps = 0
                    
    def render(self):

        width, height = self.sim.to_scale()

        pygame.init()
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.sim.width, self.sim.height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.sim.width, self.sim.height))
        canvas.fill((255, 255, 255))

        font = pygame.font.Font(None, 50)
        text_surface = font.render(str(self.steps * self.dt) + 's', False, (0,0,0))

        canvas.blit(text_surface, (0, 0))

        for plane in self.sim.planes_not_landed:
            x, y = self.world_to_screen(plane.x, plane.y)
            pygame.draw.circle(canvas, (255, 0, 0), (x, y), 5)
            pygame.draw.line(canvas, (0,0,255), (x,y), tuple(np.array([x,y]) + plane.direction * np.array([10,-10])), 1)

        for airport in self.sim.airports:
            x, y = self.world_to_screen(airport.x, airport.y)
            pygame.draw.circle(canvas, (0, 255, 0), (x, y), 5)
        

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(1 / self.dt * self.speed)

        elif self.render_mode == "rgb":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def world_to_screen(self,x, y):
        width, height = self.sim.to_scale()
        screen_x = (x / width) * self.sim.width
        screen_y = ((height - y) / height) * self.sim.height
        return screen_x, screen_y

    def get_images(self):
        final = np.stack(self.images)
        images = []
        return final

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None