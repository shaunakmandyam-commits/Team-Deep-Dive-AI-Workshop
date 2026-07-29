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
    airport_distance = 1
    max_seconds = 600
    command_interval = 30
    call_freq = 2048 * 2

    def __init__(self, config=None, dt=1, render_mode=None, speed=1, difficulty = 2):
        super().__init__()
        self.dt = dt
        self.steps = 0

        
        self.render_mode = render_mode
        self.window=None
        self.clock=None
        self.speed=speed

        self.images = []

        self.possible_agents = ["plane1", "plane2","plane3", "plane4", "plane5", "plane6"]
        self.agents = []

        self.observation_spaces, self.action_spaces = self._build_spaces()

        self.num_planes = difficulty
        self.difficulty = difficulty
        self.episodes = 0
        self.call_steps = 0
        self.losses = 0
        self.too_close = 0
        
        self.inner_collision_distance = 2 + (2 - self.num_planes) / 4
        self.outer_collision_distance = self.inner_collision_distance + 1


    @property
    def max_steps(self):
        return int(self.max_seconds / self.dt)
    @property
    def frequency(self):
        return int(self.command_interval / self.dt)

    def _build_spaces(self):
        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport,
        # 7: time left, 8: id
        low_agent = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_agent = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, np.inf, 1.0, np.inf], dtype=np.float32)

        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy, 9: other id
        low_other = np.array([-np.inf, -np.inf, 0.0, -1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, 0.0], dtype=np.float32)
        high_other = np.array([np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32)

        obs_spaces = {}
        action_spaces = {}
        for agent in self.possible_agents:
            obs_spaces[agent] = Box(np.concatenate([low_agent, low_other]),
                                    np.concatenate([high_agent, high_other]),
                                    dtype=np.float32)
            
            action_spaces[agent] = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        return obs_spaces, action_spaces
    
    def _plane_observation(self, agent):
        
        width, height = self.sim.to_scale()

        # 0: dx, 1: dy, 2: speed, 3: sin heading, 4: cos heading, 5: bearing error, 6: distance to airport,
        # 7: time left, 8: id

        plane = self.sim.name_to_plane[agent]
        agent_obs = np.array([plane.dp[0] / width, 
                              plane.dp[1] / height, 
                              plane.speed,
                              plane.direction[1],
                              plane.direction[0],
                              plane.heading_error / 180,
                              plane.distance_to_airport / np.sqrt(width ** 2 + height ** 2),
                              (self.max_steps - self.steps) / self.max_steps,
                              plane.atc_id
                              ], dtype=np.float32)
        
        
        return agent_obs

    def _other_observation(self, agent):

        width, height = self.sim.to_scale()

        
        # 0: dx, 1: dy, 2: other speed, 3: other sin, 4: other cos, 5: angle between planes, 6: distance,
        # 7: other plane airport dx, 8: other plane airport dy, 9: other id
        plane = self.sim.name_to_plane[agent]
        other_plane = self.sim.find_closest_plane(plane)

        plane_index = self.sim.name_to_index[agent]
        other_index = self.sim.name_to_index[other_plane.name]
        other_obs = np.array([self.sim.relative_positions[plane_index][other_index][0] / width,
                              self.sim.relative_positions[plane_index][other_index][1] / height,
                              0.2,
                              other_plane.direction[1],
                              other_plane.direction[0],
                              self.sim.angle_between_planes(plane, other_plane) / 180,
                              self.sim.distance_matrix[plane_index][other_index] / np.sqrt(width ** 2 + height ** 2),
                              other_plane.dp[0] / width, 
                              other_plane.dp[1] / height, 
                              other_plane.atc_id
                              ])
        return other_obs
    
    def _obs(self):
        observations = {}
        for agent in self.agents:
            observations[agent] = np.concatenate([self._plane_observation(agent),
                                                  self._other_observation(agent)])
        return observations
    
    def _info(self):
        infos = {}
        return infos

    def _reset_agent(self, plane_angle, agent, id):
        width, height = self.sim.to_scale()
        small = min(width, height) / 2
        midpoint = np.array([width / 2, height / 2])

        plane_distance = self.np_random.uniform(5, 10) + (self.difficulty - 1) +3
        airport_distance = self.np_random.uniform(5, 10) + (self.difficulty - 1)

    
        airport_angle = (plane_angle + 180) % 360

        plane_pos = midpoint + plane_distance * angle_to_vector(plane_angle)
        airport_pos = midpoint + airport_distance * angle_to_vector(airport_angle)

        return (Plane(agent, "plane_policy", 100, plane_pos[0], plane_pos[1], heading=airport_angle, id=id),
                Airport(airport_pos[0], airport_pos[1]))
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.images = []

        
        self.episodes += 1
        if self.call_steps / self.call_freq > 1:
            self.call()

        self.steps = 0
        self.sim = PlaneSim(self.scale, self.dt)

        self.agents = []
        l = [0,1,2,3,4,5]
        random.shuffle(l)
        for i in range(self.num_planes):
            num = l.pop()
            self.agents.append(self.possible_agents[num])
        

        angles = [0, 60, 120, 180, 240, 300]
        random.shuffle(angles)
        for agent in self.agents:
            angle = angles.pop() + self.np_random.uniform(-150, 150)
            plane, airport = self._reset_agent(angle, agent, int(angle / 30))
            self.sim.add(plane, airport)

        
        observation = self._obs()
        info = self._info()

        return observation, info

    def step(self, action:dict):
        
        self.call_steps += 1

        num_actions = 0
        for name, heading in action.items():
            a = heading[0]
            plane = self.sim.name_to_plane[name]
            if not plane.landed:
                self.sim.act(plane, (plane.heading + heading * 90) % 360)

        self.agents = []
        for plane in self.sim.planes_not_landed:
            self.agents.append(plane.name)

        collision = False
        reward = {}
        terminated = {}
        truncated = {}

        for agent in self.agents:
            reward[agent] = 0
            terminated[agent] = False
            truncated[agent] = False
        
        old_distances = self.sim.distance_to_airport.copy()
        step_penalty = 0.1

        for i in range(self.frequency):
            for agent in self.agents:
                reward[agent] -= step_penalty / self.frequency

            self.steps += 1
            self.sim.step()
            if self.render_mode == "human":
                self.render()
            elif self.render_mode == "rgb":
                image = self.render()
                self.images.append(image)

            for agent in self.agents:
                plane = self.sim.name_to_plane[agent]
                closest = self.sim.find_closest_plane(plane)
                index1 = self.sim.name_to_index[agent]
                index2 = self.sim.name_to_index[closest.name]
                distance = self.sim.distance_matrix[index1][index2]

                if distance < self.inner_collision_distance:
                    if self.render_mode == "human": print(plane.name, "collision")
                    terminated[agent] = True
                    reward[agent] -= 10
                    collision = True

                elif distance < self.outer_collision_distance:
                    margin = self.outer_collision_distance - self.inner_collision_distance
                    reward[agent] -= 0.3

            for plane, distance in zip(self.sim.planes, self.sim.distance_to_airport):
                if distance < self.airport_distance and not plane.landed:
                    if self.render_mode == "human": print(plane.name, "landed")
                    plane.set_speed(0)
                    plane.landed = True
                    terminated[plane.name] = True
                    reward[plane.name] += 2
                    
            if collision: 
                if self.steps < 5:
                    self.losses -= 1
                    self.too_close += 1
                    #print("too close")
                self.losses += 1
                break
            
        for agent in self.agents:
            index = self.sim.name_to_index[agent]
            new_distances = self.sim.distance_to_airport
            distance = old_distances[index] - new_distances[index]

            reward[agent] += distance / 50

        if self.steps > self.max_steps:
            self.too_close += 1
            self.losses += 1
            truncated["__all__"] = True
            if self.render_mode == "human": print("timeout")
            for agent in self.agents:
                truncated[agent] = True
                reward[agent] -= 2
        else:
            truncated["__all__"] = False
        

        terminated["__all__"] = collision or len(self.sim.planes_not_landed) == 0
        observation = self._obs()
        info = self._info()
        """
        print(reward)
        print(terminated)
        print(truncated)
        """
        return observation, reward, terminated, truncated, info

                    
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
            pygame.draw.circle(canvas, (255, 0, 0), (x, y), self.inner_collision_distance / self.scale - 5, 1)
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

    def get_images(self):
        final = np.stack(self.images)
        return final

    def call(self):
        if self.losses < 30:
            if self.difficulty < 6:
                self.difficulty += 1
            else:
                self.difficulty = 6
                self.call_freq = -1

        self.num_planes = self.difficulty
        self.inner_collision_distance = 2 + (2 - self.num_planes) / 4
        self.outer_collision_distance = self.inner_collision_distance + 1

        info = {"episodes": self.episodes,
                "successes": self.episodes - self.losses,
                "losses": self.losses,
                "winrate": (self.episodes - self.losses) / (self.episodes + 0.0001),
                "difficulty": self.difficulty,
                "collision distance": self.inner_collision_distance,
                "close": self.too_close}
        
        for key, val in info.items():
            print(key, val)
        
        self.losses = 0
        self.episodes = 0
        self.call_steps = 0
        self.too_close = 0
    
    def world_to_screen(self,x, y):
        width, height = self.sim.to_scale()
        screen_x = (x / width) * self.sim.width
        screen_y = ((height - y) / height) * self.sim.height
        return screen_x, screen_y

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None