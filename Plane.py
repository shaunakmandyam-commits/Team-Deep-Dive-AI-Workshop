from simple_pid import PID
import numpy as np

class Plane:
    def __init__(self, name, model, capacity, x=0, y=0, altitude=0, heading=0, speed=0.2,
                  max_speed_change=5, max_heading_change=3, max_altitude_change=100):
        self.name = name
        self.model = model
        self.capacity = capacity
        self.position = np.array([x, y])
        self.altitude = altitude
        self.heading = heading
        self.speed = speed
        self.landed = False
        

        self.x = x
        self.y = y

        self.target_speed = speed
        self.target_heading = heading
        self.target_altitude = altitude

        self.max_speed_change = max_speed_change
        self.max_heading_change = max_heading_change
        self.max_altitude_change = max_altitude_change

        self.speed_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_speed)
        self.heading_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_heading)
        self.altitude_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_altitude)
        self.set_limits()

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value):
        self.position[0] = value

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, value):
        self.position[1] = value

    def get_info(self):
        return (
            f"Plane Name: {self.name}, Model: {self.model}, Capacity: {self.capacity}, "
            f"Position: ({self.x}, {self.y}), Altitude: {self.altitude}, "
            f"Heading: {self.heading}, Speed: {self.speed}"
        )

    def set_position(self, x, y):
        self.position = np.array([x, y])

    def set_altitude(self, altitude):
        self.altitude = altitude

    def set_heading(self, heading):
        self.heading = heading

    def set_speed(self, speed):
        self.speed = speed

    @property
    def direction(self):
        rad = np.radians(self.heading)
        return np.array([np.cos(rad), np.sin(rad)])
    
    def move(self, dt=1.0):
        self.autopilot_update(dt=dt)
        self.position += self.direction * self.speed * dt


    def autopilot_update(self, dt=1.0):
        
        self.speed += self.speed_pid(self.speed, dt=dt) * dt
        self.heading += self.heading_pid(self.heading, dt=dt) * dt 
        self.altitude += self.altitude_pid(self.altitude, dt=dt) * dt

    def normalize_angle(self):
        self.heading = self.heading % 360

    def autopilot_input(self, target_speed, target_heading, target_altitude):

        self.target_speed = target_speed
        
        self.target_heading = self.heading + self.angle_error(target_heading)
        self.target_altitude = target_altitude
        self.update_pid()

    def angle_error(self, target):
        self.normalize_angle()
        return (target - self.heading + 180) % 360 - 180

    def update_pid(self):
        self.speed_pid.setpoint = self.target_speed
        self.heading_pid.setpoint = self.target_heading
        self.altitude_pid.setpoint = self.target_altitude
    
    def set_limits(self):
        self.speed_pid.output_limits = (-self.max_speed_change, self.max_speed_change)
        self.heading_pid.output_limits = (-self.max_heading_change, self.max_heading_change)
        self.altitude_pid.output_limits = (-self.max_altitude_change, self.max_altitude_change)
    
    def distance(self, x, y):
        return np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

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

    def observe(self):
        return {
            "x": self.x,
            "y": self.y,
            "altitude": self.altitude,
            "heading": self.heading,
            "speed": self.speed
        }

    def __str__(self):
        self.normalize_angle()
        return f"Plane(name={self.name}, model={self.model}, capacity={self.capacity}, x={self.x}, y={self.y}, altitude={self.altitude}, heading={self.heading}, speed={self.speed})"

    

class Airport():

    def __init__(self, x, y):
        self.position = np.array([x,y])

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, value):
        self.position[0] = value

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, value):
        self.position[1] = value
