import math
import pygame
import gymnasium as gym
import numpy as np
from Plane import Plane, Airport
from PlaneSim import PlaneSim


class TwoPlanesEnv(gym.Env):
    """A simple environment with two independent plane sims.

    Action: Box(shape=(2,)) with values in [-1,1] representing heading delta scale for each plane.
    Observation: concatenation of two plane observations (8 each) -> shape (16,)
    """

    scale = 40/800
    collision_distance = 1
    max_seconds = 600
    command_interval = 60

    def __init__(self, dt=1, render_mode=None, speed=1):
        self.dt = dt
        self.render_mode = render_mode
        self.speed = speed

        # Two separate sims (each manages a single Plane + airport)
        self.sim1 = PlaneSim(self.scale, dt)
        self.sim2 = PlaneSim(self.scale, dt)

        self.steps = 0
        self.window = None
        self.clock = None

        # Observations: plane_x, plane_y, speed, sin(rad), cos(rad), bearing_error/180, airport_x, airport_y (8 per plane)
        low = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0] * 2, dtype=np.float32)
        high = np.array([1.0] * 16, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

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

    def _plane_observation(self, sim):
        sim.plane.heading %= 360
        rad = np.deg2rad(sim.plane.heading)
        dx = sim.plane.x - sim.airport.x
        dy = sim.plane.y - sim.airport.y
        radians = np.arctan2(dy, dx)
        degrees = np.degrees(radians)
        deg = (degrees + 360) % 360
        bearing_error = (deg - sim.plane.heading + 180) % 360 - 180

        px = np.clip(sim.plane.x / sim.x, 0.0, 1.0)
        py = np.clip(sim.plane.y / sim.y, 0.0, 1.0)

        return np.array([
            px,
            py,
            sim.plane.speed,
            np.sin(rad),
            np.cos(rad),
            bearing_error / 180.0,
            sim.airport.x / sim.x,
            sim.airport.y / sim.y
        ], dtype=np.float32)

    def observe(self):
        obs1 = self._plane_observation(self.sim1)
        obs2 = self._plane_observation(self.sim2)
        return np.concatenate([obs1, obs2], axis=0)

    def step(self, action):
        # action expected shape (2,)
        a0 = float(action[0])
        a1 = float(action[1])

        # scale heading delta to +/-30 degrees per command
        h0 = (self.sim1.plane.heading + a0 * 30.0) % 360
        h1 = (self.sim2.plane.heading + a1 * 30.0) % 360

        self.sim1.act(h0)
        self.sim2.act(h1)

        old_d0 = self.sim1.plane.distance(self.sim1.airport.x, self.sim1.airport.y)
        old_d1 = self.sim2.plane.distance(self.sim2.airport.x, self.sim2.airport.y)

        for i in range(self.frequency):
            self.steps += 1
            self.sim1.step()
            self.sim2.step()
            if self.render_mode is not None:
                self.render()
            # stop early if any plane reaches its airport
            if (self.sim1.plane.distance(self.sim1.airport.x, self.sim1.airport.y) < self.collision_distance
                    or self.sim2.plane.distance(self.sim2.airport.x, self.sim2.airport.y) < self.collision_distance):
                break

        new_d0 = self.sim1.plane.distance(self.sim1.airport.x, self.sim1.airport.y)
        new_d1 = self.sim2.plane.distance(self.sim2.airport.x, self.sim2.airport.y)

        # collision between planes
        inter_plane_dist = math.hypot(self.sim1.plane.x - self.sim2.plane.x, self.sim1.plane.y - self.sim2.plane.y)
        collision = inter_plane_dist < self.collision_distance

        # Rewards: sum of distance reductions normalized by world size, small step penalty
        world_scale = max(self.sim1.x, self.sim1.y)
        reward = ((old_d0 - new_d0) + (old_d1 - new_d1)) / (world_scale if world_scale > 0 else 1.0)
        reward -= 0.01

        terminated = False
        if collision:
            reward -= 2.0
            terminated = True

        if new_d0 < self.collision_distance or new_d1 < self.collision_distance:
            reward += 1.0
            terminated = True

        truncated = self.steps > self.max_steps
        if truncated:
            reward -= 0.5

        obs = self.observe()
        info = {"steps": self.steps, "inter_plane_dist": inter_plane_dist}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # re-create sims so internal state is clean
        self.sim1 = PlaneSim(self.scale, self.dt)
        self.sim2 = PlaneSim(self.scale, self.dt)
        self.steps = 0

        # place planes in opposite halves and airports near edges
        x1 = self.sim1.x * 0.25
        y1 = self.sim1.y * 0.5
        x2 = self.sim2.x * 0.75
        y2 = self.sim2.y * 0.5

        heading1 = int(self.np_random.integers(0, 360))
        heading2 = int(self.np_random.integers(0, 360))

        speed = 0.2
        plane1 = Plane("p1", "RL", 100, x1, y1, 0, heading1, speed)
        plane2 = Plane("p2", "RL", 100, x2, y2, 0, heading2, speed)

        airport1 = Airport(self.sim1.x * 0.05, self.sim1.y * 0.1)
        airport2 = Airport(self.sim2.x * 0.95, self.sim2.y * 0.9)

        self.sim1.reset(plane1, airport1)
        self.sim2.reset(plane2, airport2)

        obs = self.observe()
        info = {"steps": self.steps}
        return obs, info

    def render(self):
        # reuse PlaneEnv rendering approach but draw both planes
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.sim1.width, self.sim1.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.sim1.width, self.sim1.height))
        canvas.fill((255, 255, 255))

        def world_to_screen(sim, x, y):
            sx = (x / sim.x) * sim.width
            sy = ((sim.y - y) / sim.y) * sim.height
            return int(sx), int(sy)

        p1x, p1y = world_to_screen(self.sim1, self.sim1.plane.x, self.sim1.plane.y)
        a1x, a1y = world_to_screen(self.sim1, self.sim1.airport.x, self.sim1.airport.y)

        p2x, p2y = world_to_screen(self.sim2, self.sim2.plane.x, self.sim2.plane.y)
        a2x, a2y = world_to_screen(self.sim2, self.sim2.airport.x, self.sim2.airport.y)

        pygame.draw.circle(canvas, (255, 0, 0), (p1x, p1y), 5)
        pygame.draw.circle(canvas, (0, 255, 0), (a1x, a1y), 5)
        pygame.draw.circle(canvas, (0, 0, 255), (p2x, p2y), 5)
        pygame.draw.circle(canvas, (255, 255, 0), (a2x, a2y), 5)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(1 / self.dt * self.speed)
