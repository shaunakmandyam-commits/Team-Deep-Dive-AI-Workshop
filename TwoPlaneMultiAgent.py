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


class TwoPlaneMultiAgent(MultiAgentEnv):
    scale = 40/800
    inner_collision_distance = 1
    outer_collision_distance = 2
    airport_distance = 1
    max_seconds = 600
    command_interval = 30

    def __init__(self, config=None, dt=1, render_mode=None, speed=1):
        super().__init__()
        self.dt = dt
        self.steps = 0

        
        self.render_mode = render_mode
        self.window=None
        self.clock=None
        self.speed=speed

        self.possible_agents = ["plane1", "plane2"]
        self.agents = []

        self.observation_spaces, self.action_spaces = self._build_spaces()


    @property
    def max_steps(self):
        return int(self.max_seconds / self.dt)
    @property
    def frequency(self):
        return int(self.command_interval / self.dt)

    def _build_spaces(self):
        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport
        low_agent = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high_agent = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, np.inf, 1.0], dtype=np.float32)

        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy
        low_other = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf], dtype=np.float32)
        high_other = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1, np.inf, np.inf], dtype=np.float32)

        obs_spaces = {}
        action_spaces = {}
        for agent in self.possible_agents:
            obs_spaces[agent] = Box(np.concatenate([low_agent, low_other]),
                                    np.concatenate([high_agent, high_other]),
                                    dtype=np.float32)
            
            action_spaces[agent] = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        return obs_spaces, action_spaces
    
    def _plane_observation(self, index):
        
        width, height = self.sim.to_scale()

        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport
        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy
        plane1 = self.sim.planes[index]
        agent = np.array([self.sim.d_pos[index][0] / width, 
                          self.sim.d_pos[index][1] / height, 
                          plane1.speed,
                          plane1.direction[1],
                          plane1.direction[0],
                          self.sim.planes[index].heading_error / 180,
                          self.sim.distance_to_airport[index] / np.sqrt(width ** 2 + height ** 2),
                          (self.max_steps - self.steps) / self.max_steps
                          ], dtype=np.float32)
        i2 = (index + 1) % 2
        plane2 = self.sim.planes[i2]
        other = np.array([self.sim.relative_positions[index][i2][0] / width,
                          self.sim.relative_positions[index][i2][1] / height,
                          plane2.speed,
                          plane2.direction[1],
                          plane2.direction[0],
                          self.sim.angle_between_planes(index, i2) / 180,
                          self.sim.distance_matrix[index][i2] / width,
                          self.sim.d_pos[i2][0] / width, 
                          self.sim.d_pos[i2][1] / height
                          ], dtype=np.float32)
        
        return np.concatenate([agent, other])
        
    def _obs(self):
        observations = {}
        for agent in self.agents:
            observations[agent] = self._plane_observation(self.sim.plane_to_index[agent])
        return observations
    
    def _info(self):
        #return {"steps": self.steps}
        return {
                    "plane1": {"steps": self.steps},
                    "plane2": {"steps": self.steps}
                }

    def _reset_agent(self, plane_angle, agent):
        width, height = self.sim.to_scale()
        small = min(width, height) / 2
        midpoint = np.array([width / 2, height / 2])

        plane_distance = self.np_random.uniform(small - 1, small) 
        airport_distance = self.np_random.uniform(small - 1, small) 

        plane_angle = self.np_random.uniform(0, 360)
        airport_angle = (plane_angle + 180) % 360

        plane_pos = midpoint + plane_distance * angle_to_vector(plane_angle)
        airport_pos = midpoint + airport_distance * angle_to_vector(airport_angle)


        return (Plane(agent, "plane_policy", 100, plane_pos[0], plane_pos[1], heading=airport_angle),
                Airport(airport_pos[0], airport_pos[1]))
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
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
        
        for name, heading in action.items():
            plane = self.sim.name_to_plane[name]
            self.sim.act(plane, (plane.heading + heading * 90) % 360)

        old_distance_airports = self.sim.distance_to_airport

        reward = 0
        terminated = False
        truncated = False


        for i in range(self.frequency):
            self.steps += 1
            self.sim.step()

            if self.render_mode != None:
                self.render()
            for plane, distance in zip(self.sim.planes, self.sim.distance_to_airport):
                if distance < self.airport_distance and not plane.landed:
                    if self.render_mode == "human": print(plane.name, "landed")
                    plane.set_speed(0)
                    plane.landed = True
                    reward += 20

            if np.any(self.sim.no_diagonal_distances < self.inner_collision_distance):
                if self.render_mode == "human": print("collision")
                terminated = True
                reward -= 100
                break

        if len(self.sim.planes_not_landed) == 0:
            if self.render_mode == "human": print("all landed")
            terminated = True

        if self.steps > self.max_steps:
            if self.render_mode == "human": print("timeout")
            truncated = True
            reward -= 10

        
        new_distance_airports = self.sim.distance_to_airport

        reward += np.sum(old_distance_airports - new_distance_airports) / 100
        reward -= np.sum(3 / np.maximum(self.sim.no_diagonal_distances, 1) ** 2)
        reward -= 0.05

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

                    
    def render(self):

        width, height = self.sim.to_scale()

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.sim.width, self.sim.height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.sim.width, self.sim.height))
        canvas.fill((255, 255, 255))


        for plane in self.sim.planes_not_landed:
            x, y = self.world_to_screen(plane.x, plane.y)
            pygame.draw.circle(canvas, (255, 0, 0), (x, y), 5)

        for airport in self.sim.airports:
            x, y = self.world_to_screen(airport.x, airport.y)
            pygame.draw.circle(canvas, (0, 255, 0), (x, y), 5)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(1 / self.dt * self.speed)
    
    def world_to_screen(self,x, y):
        width, height = self.sim.to_scale()
        screen_x = (x / width) * self.sim.width
        screen_y = ((height - y) / height) * self.sim.height
        return screen_x, screen_y

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None