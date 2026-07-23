
from Plane import Plane, Airport
import numpy as np

def angle_between_points(dp):
    return (np.degrees(np.arctan2(dp[1], dp[0])) + 180) % 360 -180

class PlaneSim():
    def __init__(self, scale, dt, width=800, height=600):
        self.width = width
        self.height = height
        self.scale = scale
        self.x, self.y = self.to_scale()
        self.dt = dt

        self.planes: list[Plane] = []
        self.airports: list[Airport] = []


    @property
    def plane(self):
        return self.planes[0]
    @property
    def airport(self):
        return self.airports[0]
    
    @property
    def distance_to_airport(self):
        return np.linalg.norm(self.d_pos, axis=1)
    
    @property
    def angle_to_airport(self):
            return np.array([
                angle_between_points(dp) for dp in self.d_pos
            ], dtype=np.float32)
    
    @property
    def direction_to_airport(self):
        d_pos = self.d_pos
        distances = np.linalg.norm(d_pos, axis=1, keepdims=True)
        return d_pos / np.maximum(distances, 1e-6)
    
    @property
    def d_pos(self):
        return np.array([
            airport.position - plane.position
            for plane, airport in zip(self.planes, self.airports)
        ], dtype=np.float32)

    @property
    def relative_positions(self):
        positions = np.array(
            [plane.position for plane in self.planes],
            dtype=np.float32
        )

        return positions[None,:,:] - positions[:,None,:]

    @property
    def distance_matrix(self):
        positions = np.array(
            [plane.position for plane in self.planes],
            dtype=np.float32
        )

        diff = positions[:, None, :] - positions[None, :, :]

        return np.linalg.norm(diff, axis=2)
    
    @property
    def planes_not_landed(self):
        return [plane for plane in self.planes if not plane.landed]

    @property
    def no_diagonal_distances(self):
        dist = self.distance_matrix.copy()
        np.fill_diagonal(dist, np.inf)
        return dist
    
    def to_scale(self):
        return self.width * self.scale, self.height * self.scale

    def step(self):
        for plane in self.planes:
            plane.move(self.dt)
    
    def act(self, actions):
        for plane, action in zip(self.planes, actions):
            plane.autopilot_input(plane.speed, action, 0)

    def observe(self):
        maksym = list(self.planes.observe().values())
        maksym.extend(self.aiport)
        return list(maksym)

    def angle_between_planes(self, index1, index2):
        point1 = self.planes[index1].position
        point2 = self.planes[index2].position
        return angle_between_points(point1 - point2)

    def add(self, plane, airport):
        self.planes.append(plane)
        self.airports.append(airport)

    
