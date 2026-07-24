from Plane import Plane, Airport
import numpy as np

# Calculate angle in degrees between two points
def angle_between_points(dp):
    return (np.degrees(np.arctan2(dp[1], dp[0])) + 180) % 360 - 180

# Class representing the simulation environment for planes and airports
class PlaneSim():
    def __init__(self, scale, dt, width=800, height=600):
        self.width = width
        self.height = height
        self.scale = scale
        self.x, self.y = self.to_scale()
        self.dt = dt

        self.planes: list[Plane] = []
        self.airports: list[Airport] = []

    # Get primary plane 
    @property
    def plane(self):
        return self.planes[0]

    # Get primary airport
    @property
    def airport(self):
        return self.airports[0]
    
    # Distances from each plane to its corresponding airport
    @property
    def distance_to_airport(self):
        return np.linalg.norm(self.d_pos, axis=1)
    
    # Returns the angles from each plane to its corresponding airport
    @property
    def angle_to_airport(self):
        return np.array([
            angle_between_points(dp) for dp in self.d_pos
        ], dtype=np.float32)
    
    # Returns the normalized direction vectors from each plane to its airport
    @property
    def direction_to_airport(self):
        d_pos = self.d_pos
        distances = np.linalg.norm(d_pos, axis=1, keepdims=True)
        return d_pos / np.maximum(distances, 1e-6)
    
    # Returns the relative position vectors from each plane to its corresponding airport
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

    # Returns the relative angles from each plane
    @property
    def distance_matrix(self):
        positions = np.array(
            [plane.position for plane in self.planes],
            dtype=np.float32
        )

        diff = positions[:, None, :] - positions[None, :, :]

        return np.linalg.norm(diff, axis=2)
    
    # List of planes that have not landed yet
    @property
    def planes_not_landed(self):
        return [plane for plane in self.planes if not plane.landed]

    @property
    def no_diagonal_distances(self):
        dist = self.distance_matrix.copy()
        np.fill_diagonal(dist, np.inf)
        return dist
    
    # Convert pixel dimensions to scaled simulation dimensions
    def to_scale(self):
        return self.width * self.scale, self.height * self.scale

    # Step simulation forward by dt for all planes
    def step(self):
        for plane in self.planes:
            plane.move(self.dt)
    
    # Apply heading action inputs to all plane autopilots
    def act(self, actions):
        for plane, action in zip(self.planes, actions):
            plane.autopilot_input(plane.speed, action, 0)

    # Get observation state array combining plane and airport data
    def observe(self):
        maksym = list(self.planes.observe().values())
        maksym.extend(self.aiport)
        return list(maksym)

    # Angle from plane at index2 to plane at index1
    def angle_between_planes(self, index1, index2):
        point1 = self.planes[index1].position
        point2 = self.planes[index2].position
        return angle_between_points(point1 - point2)

    # Add plane and paired airport to simulation
    def add(self, plane, airport):
        self.planes.append(plane)
        self.airports.append(airport)