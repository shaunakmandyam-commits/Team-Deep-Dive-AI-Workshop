
from Plane import Plane, Airport
import numpy as np


class PlaneSim():
    def __init__(self, scale, dt, width=800, height=600):
        self.width = width
        self.height = height
        self.x, self.y = self.to_scale(scale)
        self.dt = dt

        self.planes: list[Plane] = []
        self.airports: list[Airport] = []


    @property
    def plane(self):
        return self.planes[0]
    @property
    def airport(self):
        return self.airports[0]

    def to_scale(self, scale):
        return self.width * scale, self.height * scale

    def step(self):
        for plane in self.planes:
            plane.move(self.dt)
    
    def act(self, action):
        self.plane.autopilot_input(self.plane.speed, action, 0)

    def observe(self):
        maksym = list(self.planes.observe().values())
        maksym.extend(self.aiport)
        return list(maksym)

    def distance(self, index1, index2):
        point1 = self.planes[index1].position
        point2 = self.planes[index2].position

        return np.linalg.norm(point1 - point2)

    def distance(self):
        pass


    def reset(self, plane, airport):
        self.planes.append(plane)
        self.airports.append(airport)

    
