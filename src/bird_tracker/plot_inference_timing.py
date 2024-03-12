import matplotlib.pyplot as plt

execution_times_ms = [295.003007, 281.53983, 338.305615, 325.040549, 276.625318, 275.907317, 276.118964, 276.593908, 277.049705, 279.82499, 277.82354, 295.58265, 277.58015, 278.061133, 277.494759, 276.666868, 277.371296, 277.491129, 274.572659, 276.148292, 274.78777, 274.143323, 275.323992, 275.466955, 274.718158, 275.258583, 275.261694, 275.607361, 274.560803, 276.589603, 274.760173, 275.297285, 274.633875, 276.336304, 275.136598, 275.856007]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(execution_times_ms)), execution_times_ms, marker='o', linestyle='-', color='b')
plt.title("TensorFlow-Lite Inference Times for 16K Parameter Model on 144kS Audio Input (Pi 4B)")
plt.xlabel("Audio Frame")
plt.ylabel("Execution Time (ms)")
plt.grid(True)

# Show the plot
plt.show()
