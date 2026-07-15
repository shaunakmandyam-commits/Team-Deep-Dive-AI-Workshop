import math
import sys
import pygame
from Plane import Plane

pygame.init()

WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Plane Turn Demo")
CLOCK = pygame.time.Clock()

WORLD_WIDTH_MILES = 40
WORLD_HEIGHT_MILES = 30


def world_to_screen(x, y):
    screen_x = (x / WORLD_WIDTH_MILES) * WIDTH
    screen_y = HEIGHT - (y / WORLD_HEIGHT_MILES) * HEIGHT
    return screen_x, screen_y

def draw_plane(plane, size=10):
    screen_x, screen_y = world_to_screen(plane.x, plane.y)
    angle = math.radians(plane.heading)

    tip_x = screen_x + size * math.cos(angle)
    tip_y = screen_y - size * math.sin(angle)
    left_x = screen_x - (size / 2) * math.cos(angle + math.radians(140))
    left_y = screen_y + (size / 2) * math.sin(angle + math.radians(140))
    right_x = screen_x - (size / 2) * math.cos(angle - math.radians(140))
    right_y = screen_y + (size / 2) * math.sin(angle - math.radians(140))

    pygame.draw.polygon(
        SCREEN,
        (255, 255, 255),
        [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)],
    )

    #line representing heading from center to beyond the tip of the plane
    tip_x = screen_x + (size * 1.5) * math.cos(angle)
    tip_y = screen_y - (size * 1.5) * math.sin(angle)
    pygame.draw.line(
        SCREEN,
        (255, 0, 0),
        (screen_x, screen_y),
        (tip_x, tip_y),
        1
    )

plane = Plane(
    name="Demo Plane",
    model="Model X",
    capacity=120,
    x=0,
    y=0,
    altitude=1000,
    heading=45,
    speed=0.16,
    max_speed_change=0,
    max_heading_change=3,
    max_altitude_change=100,
)

# Start by moving toward the bottom-left, then switch to due north halfway through.
fps = 1
speed = 1
turn_frame = 10 * fps
frame_count = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not running:
        break

    frame_count += 1
    if frame_count >= turn_frame:
        plane.autopilot_input(target_speed=plane.speed, target_heading=90, target_altitude=1000)

    dt = CLOCK.tick(fps*speed) / 1000.0 * speed # Convert milliseconds to seconds
    plane.move(dt=dt)

    SCREEN.fill((20, 20, 30))

    screen_x, screen_y = world_to_screen(plane.x, plane.y)

    draw_plane(plane, size=25)

    
    #print time elapsed on top left corner of screen
    font = pygame.font.Font(None, 36)
    elapsed_time = frame_count / fps
    time_text = font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))

    SCREEN.blit(time_text, (10, 10))

    pygame.display.flip()

pygame.quit()
sys.exit()
