import math
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Sequence
import numpy as np
from Plane import Plane, Airport
from PlaneSim import PlaneSim

def angle_to_vector(angle):
    rad = np.radians(angle)
    return np.array([np.cos(rad), np.sin(rad)])

class TwoPlanesEnv(gym.Env):
    """A simple environment with two independent plane sims.

    Action: Box(shape=(2,)) with values in [-1,1] representing heading delta scale for each plane.
    Observation: concatenation of two plane observations (8 each) -> shape (16,)
    """

    scale = 80/800
    inner_collision_distance = 1
    outer_collision_distance = 2
    airport_distance = 1
    max_seconds = 600
    command_interval = 30

    def __init__(self, dt=1, render_mode=None, speed=1):
        self.dt = dt
        self.steps = 0

        
        self.render_mode = render_mode
        self.window=None
        self.clock=None
        self.speed=speed

        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport
        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy
        low_agent = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, -np.inf], dtype=np.float32)
        low_other = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf], dtype=np.float32)
        high_agent = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, np.inf], dtype=np.float32)
        high_other = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1, np.inf, np.inf], dtype=np.float32)

        self.observation_space = Dict({"agent": Box(np.tile(low_agent, (2,1)), np.tile(high_agent, (2,1)), shape=(2,7),dtype=np.float32),
                                       "other": Box(np.tile(low_other, (2,1)), np.tile(high_other, (2,1)), shape=(2,9), dtype=np.float32)})

        # Two continuous actions (one per plane)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                           high=np.array([1.0, 1.0], dtype=np.float32),
                                           dtype=np.float32)
    @property
    def max_steps(self):
        return int(self.max_seconds / self.dt)
    @property
    def frequency(self):
        return int(self.command_interval / self.dt)
    
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
                          ((self.sim.angle_to_airport[index] - self.sim.planes[index].heading + 180) % 360 - 180 ) / 180,
                          self.sim.distance_to_airport[index]
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
        
        return agent, other
        
    def _obs(self):
        agent0, other0 = self._plane_observation(0)
        agent1, other1 = self._plane_observation(1)
        agent_obs = np.array([agent0, agent1])
        other_obs = np.array([other0, other1])
        return {"agent": agent_obs, "other": other_obs}
    
    def _info(self):
        return {"steps": self.steps}
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.sim = PlaneSim(self.scale, self.dt)

        width, height = self.sim.to_scale()
        small = min(width, height) / 2
        midpoint = np.array([width / 2, height / 2])


        distance_plane = self.np_random.uniform(small - 1, small)

        airport_distance1 = self.np_random.uniform(small - 1, small)
        airport_distance2 = self.np_random.uniform(small - 1, small) 


        angle1 = self.np_random.uniform(0, 360)
        angle2 = (angle1 + self.np_random.uniform(45, 270)) % 360

        airport_angle1 = (angle1 + 180) % 360
        airport_angle2 = (angle2 + 180) % 360

        pos_plane1 = midpoint + distance_plane * angle_to_vector(angle1)
        pos_plane2 = midpoint + distance_plane * angle_to_vector(angle2)

        pos_airport1 = midpoint + airport_distance1 * angle_to_vector(airport_angle1)
        pos_airport2 = midpoint + airport_distance2 * angle_to_vector(airport_angle2)

        
        self.sim.add(Plane("Plane1", "RL", 100,
                            pos_plane1[0], pos_plane1[1], heading=self.np_random.uniform(0, 360)),
                     Airport(pos_airport1[0], pos_airport1[1]))
        self.sim.add(Plane("Plane2", "RL", 100,
                            pos_plane2[0], pos_plane2[1], heading=self.np_random.uniform(0, 360)),
                     Airport(pos_airport2[0], pos_airport2[1]))
        """
        self.sim.add(Plane("Plane1", "RL", 100,
                            pos_plane1[0], pos_plane1[1], heading=self.np_random.uniform(0, 360)),
                     Airport(pos_airport1[0], pos_airport1[1]))
        self.sim.add(Plane("Plane2", "RL", 100,
                            pos_airport2[0], pos_airport2[1], heading=airport_angle2),
                     Airport(pos_airport2[0], pos_airport2[1]))
        """
        observation = self._obs()
        info = self._info()

        return observation, info

    def step(self, action):
        
        heading = action * 90
        heading = np.array([(plane.heading + angle) % 360 for plane, angle in zip(self.sim.planes, heading)])
        self.sim.act(heading)

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
                    reward += 2

            if np.any(self.sim.no_diagonal_distances < self.inner_collision_distance):
                if self.render_mode == "human": print("collision")
                #terminated = True
                #reward -= 100
                #break

        if len(self.sim.planes_not_landed) == 0:
            if self.render_mode == "human": print("all landed")
            terminated = True

        if self.steps > self.max_steps:
            if self.render_mode == "human": print("timeout")
            truncated = True
            reward -= 1

        
        new_distance_airports = self.sim.distance_to_airport

        reward += np.sum(old_distance_airports - new_distance_airports) / 100
        #reward -= np.sum(3 / np.maximum(self.sim.no_diagonal_distances, 1) ** 2)
        reward -= 0.05

        reward = float(reward)


        observation = self._obs()
        info = self._info()
        return observation, reward, terminated, truncated, info

                    
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