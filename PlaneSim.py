
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
    def distance_to_airport(self):
        return np.array([
            plane.distance_to_airport for plane in self.planes
        ], dtype=np.float32)
    
    @property
    def angle_to_airport(self):
            return np.array([
                plane.angle_to_airport for plane in self.planes
            ], dtype=np.float32)

    @property
    def angle_error_to_airport(self):
            return np.array([
                plane.angle_error for plane in self.planes
            ], dtype=np.float32)

    @property
    def direction_to_airport(self):
        dp = self.dp
        distances = np.linalg.norm(dp, axis=1, keepdims=True)
        return dp / np.maximum(distances, 1e-6)
    
    @property
    def dp(self):
        return np.array([
            plane.dp for plane in self.planes
        ], dtype=np.float32)

    """
    rows: start
    columns: end
    """
    @property
    def relative_positions(self):
        positions = np.array(
            [plane.position for plane in self.planes],
            dtype=np.float32
        )

        return positions[None,:,:] - positions[:,None,:]

    @property
    def distance_matrix(self):
        return np.linalg.norm(self.relative_positions, axis=2)
    
    @property
    def planes_not_landed(self):
        return [plane for plane in self.planes if not plane.landed]

    @property
    def no_diagonal_distances(self):
        dist = self.distance_matrix.copy()
        np.fill_diagonal(dist, np.inf)
        return dist

    @property
    def name_to_index(self):
        return {plane.name: index for index, plane in enumerate(self.planes)}

    @property
    def name_to_plane(self):
        return {plane.name: plane for plane in self.planes}

    

    def find_closest_plane(self, plane):
        index = self.name_to_index[plane.name]
        min_index = np.argmin(self.no_diagonal_distances[index])
        return self.planes[min_index]

    def reset_planes(self):
        not_landed = self.planes_not_landed
        self.planes = []
        for plane in not_landed:
            self.planes.append()
        

    def to_scale(self):
        return self.width * self.scale, self.height * self.scale

    def step(self):
        for plane in self.planes:
            plane.move(self.dt)
    
    def act(self, plane: Plane, action):
        plane.autopilot_input(plane.speed, action[0], 0)

    def angle_between_planes(self, plane1, plane2):
        point1 = plane1.position
        point2 = plane2.position
        return angle_between_points(point1 - point2)

    def add(self, plane, airport):
        plane.set_airport(airport)
        self.planes.append(plane)
        self.airports.append(airport)

    
