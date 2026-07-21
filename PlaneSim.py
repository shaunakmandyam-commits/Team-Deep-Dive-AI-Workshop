
from Plane import Plane, Airport

class PlaneSim():
    def __init__(self, scale, dt, width=800, height=600):
        self.width = width
        self.height = height
        self.x, self.y = self.to_scale(scale)
        self.dt = dt

        self.plane : Plane = None
        self.airport : Airport = None
        self.npcs = Plane = None
        self.airports = Airport = None

    def to_scale(self, scale):
        return 800 * scale, 600 * scale

    def step(self):
    
        self.plane.move(self.dt)
        for npc in self.npcs:
            npc.move(self.dt)
    
    def act(self, action):
        self.plane.autopilot_input(self.plane.speed, action, 0)

    def observe(self):
        maksym = list(self.plane.observe().values())
        maksym.extend(self.aiport)
        return list(maksym)

    def reset(self, plane, airport):
        self.plane = plane
        self.airport = airport

    
