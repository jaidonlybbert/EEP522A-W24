import matplotlib.pyplot as plt
import numpy as np
import csv
from gps_conversion import gps_conv_factors_deg_to_meters

execution_times_ms = [295.003007, 281.53983, 338.305615, 325.040549, 276.625318, 275.907317, 276.118964, 276.593908, 277.049705, 279.82499, 277.82354, 295.58265, 277.58015, 278.061133, 277.494759, 276.666868, 277.371296, 277.491129, 274.572659, 276.148292, 274.78777, 274.143323, 275.323992, 275.466955, 274.718158, 275.258583, 275.261694, 275.607361, 274.560803, 276.589603, 274.760173, 275.297285, 274.633875, 276.336304, 275.136598, 275.856007]


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


def plot_microphone():
    mic_readings = []
    filepath = "./microphone_reading.csv"
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            for reading in row:
                mic_readings.append(int(reading))
    print(mic_readings)
    print("Length: ", len(mic_readings))
    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    mean = int(np.floor(np.mean(mic_readings)))

    rng = np.max(mic_readings) - np.min(mic_readings)

    count, bins, patches = axs.hist(mic_readings, bins=rng)
    ticks = [(patch.get_x() + (patch.get_x() + patch.get_width())) / 2 for patch in patches]
    # ticklabels = (bins[1:] + bins[:-1]) / 2
#    ticklabels = np.round((ticklabels - mean), 2)
    ticklabels = np.floor(bins[:-1] - mean)
    axs.set_xticks(ticks, ticklabels, rotation=90)
    axs.set_xlabel('Deviation from Mean (bits)')
    axs.set_ylabel('Count')
    axs.set_title('Distribution of Microphone Samples')

    # Adjust spacing between subplots
    plt.tight_layout()

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
    plot_microphone()
