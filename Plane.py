from simple_pid import PID
import math

class Plane:
    def __init__(self, name, model, capacity, x=0, y=0, altitude=0, heading=0, speed=0,
                  max_speed_change=5, max_heading_change=0.1, max_altitude_change=100):
        self.name = name
        self.model = model
        self.capacity = capacity
        self.x = x
        self.y = y
        self.altitude = altitude
        self.heading = heading
        self.speed = speed

        self.target_speed = speed
        self.target_heading = heading
        self.target_altitude = altitude

        self.max_speed_change = max_speed_change
        self.max_heading_change = max_heading_change
        self.max_altitude_change = max_altitude_change

        self.speed_pid, self.heading_pid, self.altitude_pid = self.make_pid_controllers()


    def get_info(self):
        return (
            f"Plane Name: {self.name}, Model: {self.model}, Capacity: {self.capacity}, "
            f"Position: ({self.x}, {self.y}), Altitude: {self.altitude}, "
            f"Heading: {self.heading}, Speed: {self.speed}"
        )

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_altitude(self, altitude):
        self.altitude = altitude

    def set_heading(self, heading):
        self.heading = heading

    def set_speed(self, speed):
        self.speed = speed

    def move(self, dt=1.0):  # Default dt is 1/60 seconds for 60 FPS
        # Update autopilot every frame before moving the plane
        self.update_autopilot(dt=dt)

        # Update position based on speed, heading, and time
        self.x += self.speed * math.cos(math.radians(self.heading)) * dt
        self.y += self.speed * math.sin(math.radians(self.heading)) * dt

    def update_autopilot(self, dt=1.0):
        # Continuously correct the plane toward the target values each frame
        
        self.speed += self.speed_pid(self.speed, dt=dt) * dt
        self.heading += self.heading_pid(self.heading, dt=dt) * dt
        self.altitude += self.altitude_pid(self.altitude, dt=dt) * dt

    def autopilot_update(self, dt=1.0):
        self.update_autopilot(dt=dt)

    def autopilot_input(self, target_speed, target_heading, target_altitude):
        self.target_speed = target_speed
        self.target_heading = target_heading
        self.target_altitude = target_altitude
        self.speed_pid, self.heading_pid, self.altitude_pid = self.make_pid_controllers()

    def make_pid_controllers(self):
        speed_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_speed)
        speed_pid.output_limits = (-self.max_speed_change, self.max_speed_change)

        heading_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_heading)
        heading_pid.output_limits = (-self.max_heading_change, self.max_heading_change)

        altitude_pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=self.target_altitude)
        altitude_pid.output_limits = (-self.max_altitude_change, self.max_altitude_change)

        return speed_pid, heading_pid, altitude_pid
