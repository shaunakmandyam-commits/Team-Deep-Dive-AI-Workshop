from simple_pid import PID
import numpy as np

# Represents an aircraft with PID-controlled movement and navigation state
class Plane:
    def __init__(self, name, model, capacity, x=0, y=0, altitude=0, heading=0, speed=0.2,
                  max_speed_change=5, max_heading_change=3, max_altitude_change=100):
        # Basic aircraft information
        self.name : str = name
        self.model : str = model
        self.capacity : int = capacity
        self.position : np.ndarray = np.array([x, y])
        self.altitude : float = altitude
        self.heading : float = heading
        self.speed : float = speed
        self.landed : bool = False

        self.x : float = x
        self.y : float = y

        # Target parameters for autopilot
        self.target_speed : float = speed
        self.target_heading : float = heading
        self.target_altitude : float = altitude

        # Rate limit constraints
        self.max_speed_change : float = max_speed_change
        self.max_heading_change : float = max_heading_change
        self.max_altitude_change : float = max_altitude_change

        # PID controllers for flight control
        self.speed_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_speed)
        self.heading_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_heading)
        self.altitude_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_altitude)
        self.set_limits()

    # Get X coordinate
    @property
    def x(self):
        return self.position[0]

    # Set X coordinate
    @x.setter
    def x(self, value):
        self.position[0] = value

    # Get Y coordinate
    @property
    def y(self):
        return self.position[1]

    # Set Y coordinate
    @y.setter
    def y(self, value):
        self.position[1] = value

    # Return formatted string of aircraft state
    def get_info(self):
        return (
            f"Plane Name: {self.name}, Model: {self.model}, Capacity: {self.capacity}, "
            f"Position: ({self.x}, {self.y}), Altitude: {self.altitude}, "
            f"Heading: {self.heading}, Speed: {self.speed}"
        )

    # Set 2D position vector
    def set_position(self, x, y):
        self.position = np.array([x, y])

    # Set current altitude
    def set_altitude(self, altitude):
        self.altitude = altitude

    # Set current heading angle
    def set_heading(self, heading):
        self.heading = heading

    # Set current speed
    def set_speed(self, speed):
        self.speed = speed

    # Calculate 2D direction vector from heading angle
    @property
    def direction(self):
        rad = np.radians(self.heading)
        return np.array([np.cos(rad), np.sin(rad)])
    
    # Advance plane position for time step dt using autopilot outputs
    def move(self, dt=1.0):
        self.autopilot_update(dt=dt)
        self.position += self.direction * self.speed * dt

    # Adjust speed, heading, and altitude using PID control loops
    def autopilot_update(self, dt=1.0):
        self.speed += self.speed_pid(self.speed, dt=dt) * dt
        self.heading += self.heading_pid(self.heading, dt=dt) * dt 
        self.altitude += self.altitude_pid(self.altitude, dt=dt) * dt

    # Wrap heading angle within 0 to 360 degrees
    def normalize_angle(self):
        self.heading = self.heading % 360

    # Set new flight target variables for autopilot
    def autopilot_input(self, target_speed, target_heading, target_altitude):
        self.target_speed = target_speed
        self.target_heading = self.heading + self.angle_error(target_heading)
        self.target_altitude = target_altitude
        self.update_pid()

    # Calculate shortest angle delta (-180 to 180 degrees) to target heading
    def angle_error(self, target):
        self.normalize_angle()
        return (target - self.heading + 180) % 360 - 180

    # Update setpoint targets in PID controllers
    def update_pid(self):
        self.speed_pid.setpoint = self.target_speed
        self.heading_pid.setpoint = self.target_heading
        self.altitude_pid.setpoint = self.target_altitude
    
    # Configure PID output boundaries based on max rates of change
    def set_limits(self):
        self.speed_pid.output_limits = (-self.max_speed_change, self.max_speed_change)
        self.heading_pid.output_limits = (-self.max_heading_change, self.max_heading_change)
        self.altitude_pid.output_limits = (-self.max_altitude_change, self.max_altitude_change)
    
    # Calculate distance to specific 2D coordinates
    def distance(self, x, y):
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    # Return dictionary with current flight status and targets
    def info(self):
        self.heading = self.heading % 360
        return {
            "position": (self.x, self.y),
            "altitude": self.altitude,
            "heading": self.heading,
            "speed": self.speed,
            "target_speed": self.target_speed,
            "target_heading": self.target_heading,
            "target_altitude": self.target_altitude
        }

    # Return essential telemetry dict for observations
    def observe(self):
        return {
            "x": self.x,
            "y": self.y,
            "altitude": self.altitude,
            "heading": self.heading,
            "speed": self.speed
        }

    # Concise string output of Plane instance
    def __str__(self):
        self.normalize_angle()
        return f"Plane(name={self.name}, model={self.model}, capacity={self.capacity}, x={self.x}, y={self.y}, altitude={self.altitude}, heading={self.heading}, speed={self.speed})"

    

# Represents an airport location grid point
class Airport():

    def __init__(self, x, y):
        self.position = np.array([x,y])

    # Get X coordinate
    @property
    def x(self):
        return self.position[0]

    # Set X coordinate
    @x.setter
    def x(self, value):
        self.position[0] = value

    # Get Y coordinate
    @property
    def y(self):
        return self.position[1]

    # Set Y coordinate
    @y.setter
    def y(self, value):
        self.position[1] = value