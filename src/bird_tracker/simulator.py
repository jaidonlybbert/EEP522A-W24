import numpy as np
from numpy.fft import fft, ifft
from parse_audio import read_mp3
import math
import pygame


# Spatial constants
WIDTH_PIXELS = 640
HEIGHT_PIXELS = 480
WIDTH_METERS = 10  # meters
HEIGHT_METERS = 10  # meters
PIXELS_PER_METER = WIDTH_PIXELS / WIDTH_METERS
SOUND_SPEED = 330
FPS = 30

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

lon, lat = (149.960695, 71.511299)  # GPS coordinates of scene
sound_speed = 330  # meters/second


# listening node locations
sensor_locations_x_y_meters = np.array([(2, 2), (7, 7), (2, 7)])
print(sensor_locations_x_y_meters.shape)
print(sensor_locations_x_y_meters)
bird_location_x_y_meters = (5, 5)


def distance(pos1, pos2) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# true time delays
sensor_time_delays = [distance(point, bird_location_x_y_meters) / SOUND_SPEED
                      for point in sensor_locations_x_y_meters]

# load sample audio
samples, sample_rate, resolution = read_mp3("../../data/calidris_falcinellus.mp3")
signal_length = samples.shape[0]
normalized_samples = samples / np.linalg.norm(samples)

# sample shift based on time delay and sample rate
sensor_sample_shift = [int(np.floor(sample_rate * delta_t)) for delta_t in sensor_time_delays]
largest_sample_delay = max(sensor_sample_shift)
# print(largest_sample_delay)

# shifted signals based on time shift
recieved_signals = np.zeros((signal_length + largest_sample_delay,
                             len(sensor_time_delays)))

# Create random array of environment noise from normal distribution
mean = 0
sigma = 0.1
sample_size = recieved_signals.shape[1]
env_noise = np.random.normal(mean, sigma, sample_size)

# print(recieved_signals.shape)
# print(recieved_signals[0].shape)
for idx, shift in enumerate(sensor_sample_shift):
    print("sensor idx: ", idx)
    print("sensor shift: ", shift)
    recieved_signals[shift:shift+signal_length, idx] = samples

# make TDE with simple correlation algorithm
tdes = []

for i in range(recieved_signals.shape[1]):
    fft1 = fft(recieved_signals[:, 0])
    fft2 = fft(recieved_signals[:, i])
    # PHAT - Phase Transform Cross-Power Spectrum
    phat = fft1 * np.conjugate(fft2) / np.linalg.norm(fft1 * np.conjugate(fft2))
    # GCC-PHAT - Generalized Cross-Correlation Phase Transform
    gcc_phat = ifft(phat)
    print("gcc_phat length: ", gcc_phat.shape)
    print("max: ", np.max(gcc_phat))
    print("argmax: ", np.argmax(gcc_phat))
    tdes.append(np.argmax(gcc_phat) / sample_rate)

print("TDEs: ", tdes)

# Chan-ho algorithm - https://doi.org/10.1109/78.301830
# Notation consistent with the linked reference
# Signal power density - https://doi.org/10.1109/TIT.1973.1055077
k = sensor_locations_x_y_meters[:, 0] ** 2 + \
        sensor_locations_x_y_meters[:, 1] ** 2
print("K: ", k)
# [x**2 + y**2 for (x, y) in sensor_locations_x_y_meters]
d = tdes
print("d: ", d)
r = np.array(tdes) / SOUND_SPEED
print("r: ", r)
# xi,1 & yi,1 stand for xi - x1 and yi - y1 respectively
xy = sensor_locations_x_y_meters - sensor_locations_x_y_meters[0]
print("x: ", xy)
Ga = np.column_stack((xy, r))
print("Ga: ", Ga)
h = 0.5 * np.array(r ** 2 - k + k[0]).transpose()
print("h: ", h)

# initialize pygame and create window
pygame.init()
pygame.mixer.init()  # For sound
screen = pygame.display.set_mode((WIDTH_PIXELS+100, HEIGHT_PIXELS+100))
pygame.display.set_caption("Bird Localization Simulator")
clock = pygame.time.Clock()     # For syncing the FPS

# images representing sensors and source
sensor_image = pygame.image.load("img/cat.png").convert_alpha()
bird_image = pygame.image.load("img/bird.png").convert_alpha()

# Define dimension scale properties
scale_color = (0, 0, 255)  # Blue color (RGB format)
scale_position = (50, 50)
scale_height = 15
scale_font_color = (255, 255, 255)
scale_font_size = 18
scale_font_obj = pygame.font.Font(None, scale_font_size)
scale_text = scale_font_obj.render("10 meters", True, scale_font_color, None)
scale_width = PIXELS_PER_METER * 10

# Game loop
running = True
i = 0
while running:

    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    (xm, ym) = bird_location_x_y_meters
    (xpix, ypix) = (xm * PIXELS_PER_METER, ym * PIXELS_PER_METER)
    screen.blit(bird_image,
                (xpix, ypix))

    for (x, y) in sensor_locations_x_y_meters:
        (xpix, ypix) = (x * PIXELS_PER_METER, y * PIXELS_PER_METER)
        screen.blit(sensor_image, (xpix, ypix))


    # Draw the measurement scale
    pygame.draw.rect(screen, scale_color,
                     pygame.Rect(scale_position, (scale_width, scale_height)))
    screen.blit(scale_text, (scale_position[0] + 5, scale_position[1] + 2))

    pygame.display.flip()

pygame.quit()
