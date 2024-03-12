import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from typing import Dict

execution_times_ms = [295.003007, 281.53983, 338.305615, 325.040549, 276.625318, 275.907317, 276.118964, 276.593908, 277.049705, 279.82499, 277.82354, 295.58265, 277.58015, 278.061133, 277.494759, 276.666868, 277.371296, 277.491129, 274.572659, 276.148292, 274.78777, 274.143323, 275.323992, 275.466955, 274.718158, 275.258583, 275.261694, 275.607361, 274.560803, 276.589603, 274.760173, 275.297285, 274.633875, 276.336304, 275.136598, 275.856007]


def gps_conv_factors_deg_to_meters(degrees_latitude: float) -> Dict:
	# Compute lengths of degrees at specific latitude

	# Convert latitude to radians
	lat = math.radians(degrees_latitude)

	# Set up "Constants"
	m1 = 111132.92		# latitude calculation term 1
	m2 = -559.82		# latitude calculation term 2
	m3 = 1.175			# latitude calculation term 3
	m4 = -0.0023		# latitude calculation term 4
	p1 = 111412.84		# longitude calculation term 1
	p2 = -93.5			# longitude calculation term 2
	p3 = 0.118			# longitude calculation term 3

	# Calculate the length of a degree of latitude and longitude in meters
	latlen = m1 + (m2 * math.cos(2 * lat)) + (m3 * math.cos(4 * lat)) +\
		(m4 * math.cos(6 * lat))

	longlen = (p1 * math.cos(lat)) + (p2 * math.cos(3 * lat)) +\
		(p3 * math.cos(5 * lat))

	return {"meters_per_degree": {"latitude": latlen, "longitude": longlen}}


def plot_inference():
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(execution_times_ms)), execution_times_ms, marker='o', linestyle='-', color='b')
    plt.title("TensorFlow-Lite Inference Times for 16K Parameter Model on 144kS Audio Input (Pi 4B)")
    plt.xlabel("Audio Frame")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_gps():
    gps_readings_long_lat = []
    filepath = "./gps_readings.csv"
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            gps_readings_long_lat.append((float(row[0]), float(row[1])))
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    longitudes = [i[0] for i in gps_readings_long_lat]
    latitudes = [i[1] for i in gps_readings_long_lat]
    gps_conversion = gps_conv_factors_deg_to_meters(47.0)
    longitude_mean = np.mean(longitudes)
    latitude_mean = np.mean(latitudes)

    count, bins, patches = axs[0].hist(longitudes, bins=30)
    ticks = [(patch.get_x() + (patch.get_x() + patch.get_width())) / 2 for patch in patches]
    ticklabels = (bins[1:] + bins[:-1]) / 2
    ticklabels = np.round((ticklabels - longitude_mean) *
			   gps_conversion["meters_per_degree"]["longitude"], 2)
    axs[0].set_xticks(ticks, ticklabels, rotation=90)
    axs[0].set_xlabel('Deviation from Mean (meters)')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Distribution of Longitude Readings')

    count, bins, patches = axs[1].hist(latitudes, bins=30)
    ticks = [(patch.get_x() + (patch.get_x() + patch.get_width())) / 2 for patch in patches]
    ticklabels = (bins[1:] + bins[:-1]) / 2
    ticklabels = np.round((ticklabels - latitude_mean) *
			   gps_conversion["meters_per_degree"]["latitude"], 2)
    axs[1].set_xticks(ticks, ticklabels, rotation=90)
    axs[1].set_xlabel('Deviation from Mean (meters)')
    axs[1].set_ylabel('Count')
    axs[1].set_title('Distribution of Latitude Readings')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot_gps()