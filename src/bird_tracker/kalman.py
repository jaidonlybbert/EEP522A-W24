import pydub
import math
import numpy as np
import operator
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
from numpy.fft import fft, ifft
import os


userDir = os.path.expanduser('~')
model = "BirdNET_6K_GLOBAL_MODEL"
sf_thresh = 0.03


def splitSignal(sig, rate, overlap, seconds=6.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break

        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp

        sig_splits.append(split)

    return sig_splits


def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))


def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])


def predict(sample, sensitivity):
    print(sample[0].shape)
    print(sample[1].shape)
    global INTERPRETER
    # Make a prediction
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    if model == "BirdNET_6K_GLOBAL_MODEL":
        INTERPRETER.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    INTERPRETER.invoke()
    prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # Apply custom sigmoid
    p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # Get label and scores for pooled predictions
    p_labels = dict(zip(CLASSES, p_sigmoid))

    # Sort by score
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    return p_sorted


def inference():
    global INTERPRETER, INCLUDE_LIST, EXCLUDE_LIST
    INTERPRETER = loadModel()
    lon, lat = (122.30096333, 47.67300667)  # gps_parse.getPositionData()
    week = 10
    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)
    sensitivity = 1.0  # range 0.5 to 1.5

    samples, sample_rate, resolution = read_mp3("../../data/birds.mp3")

    chunks = splitSignal(samples, sample_rate, 0.0)

    for c in chunks:
        sig = np.expand_dims(c, 0)

        p = predict([sig, mdata], sensitivity)
        print("Prediction: ", p)



def loadModel():
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX
    global MDATA_INPUT_INDEX
    global CLASSES

    print('LOADING TF LITE MODEL...', end=' ')

    # Load TFLite model and allocate tensors.
    # model will either be BirdNET_GLOBAL_6K_V2.4_Model_FP16 (new) or BirdNET_6K_GLOBAL_MODEL (old)
    modelpath = userDir + '/BirdNET-Pi/model/'+model+'.tflite'
    myinterpreter = tflite.Interpreter(model_path=modelpath, num_threads=2)
    myinterpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = myinterpreter.get_input_details()
    output_details = myinterpreter.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    if model == "BirdNET_6K_GLOBAL_MODEL":
        MDATA_INPUT_INDEX = input_details[1]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

    # Load labels
    CLASSES = []
    labelspath = userDir + '/BirdNET-Pi/model/labels_en.txt'
    with open(labelspath, 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.replace('\n', ''))

    print('DONE!')
    return myinterpreter


def read_mp3(file_path) -> np.array:
    audio = pydub.AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate  # Hertz
    resolution = audio.sample_width  # resolution in bytes
    print("Samples: ", samples)
    print("Fs: ", sample_rate)
    print("Resolution: ", resolution)
    return samples, sample_rate, resolution


def tacoma():
    # torque = 282  # ftlbs
    # torque_nm = torque * 1.35  # nm
    # power = 118.5  # kW
    # tire size = 265/70R16
    outer_tire_radius = ((265 * 0.7) / 25.4 + 8) * 25.4 / 1000  # meters
    print(outer_tire_radius)
    weight = 5000  # lbs
    weight_kg = 0.45 * weight  # kg
    print(weight_kg)
    # max_acceleration = 4 * (torque_nm / outer_tire_radius) / weight_kg
    # print("Tacoma max acceleration: ", max_acceleration, " m/s^2")
    # print("1/4 mile time: ", math.sqrt(2 * 400 / max_acceleration))
    quarter_mile_time = 17  # seconds
    quarter_mile_meters = 400  # meters
    avg_acceleration = 2 * quarter_mile_meters / (quarter_mile_time**2)
    return avg_acceleration


def gaussian_plot():
    mean = 0
    sigma = 0.1

    sample_size = 1000
    samples = np.random.normal(mean, sigma, sample_size)

    mean_difference = abs(mean - np.mean(samples))
    std_difference = abs(sigma - np.std(samples, ddof=1))

    print(f"Mean difference: {mean_difference:.2f}")
    print(f"Standard deviation: {std_difference:.2f}")

    # Plot
    count, bins, _ = plt.hist(samples, bins=100, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(-(bins - mean) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Normal Distribution")
    plt.show()


def state_update(kalman_gain, predicted_state, measurement):
    state_estimate = predicted_state + kalman_gain * (measurement -
                                                      predicted_state)
    return state_estimate


def state_extrapolation(current_state):
    # predicted_location = vel * delta_t + location
    # predicted_velocity = vel + delta_t * acceleration
    pass


def constant_velocity_tacoma():
    timesteps = 1000
    xvelocity, yvelocity = 15, 0  # mph
    velocity = np.array([xvelocity, yvelocity]) * 0.44  # m/s
    location = np.fromfunction(lambda t, _: t * velocity, (timesteps, 2),
                               dtype=float)
    mean = 0
    sigma = 0.1
    measurement_noise = np.random.normal(mean, sigma, timesteps)

    measurements = location + np.array([measurement_noise, measurement_noise]).transpose()

    kalman_gain = 0
    initial_location = np.array([0, 0])

    print(location)
    print(velocity.shape, location.shape)


def plot_mp3():
    samples, sample_rate, resolution = read_mp3("../../data/birds.mp3")
    plot_fft_mp3(samples, sample_rate)
    plt.plot([i for i in range(len(samples))], samples)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Birds mp3")
    plt.show()


def plot_samples(samples, sample_rate, resolution):

    N = len(samples)
    X = fft(samples)
    n = np.arange(N)
    T = N / sample_rate
    freq = n / T

    # Trim FFT result to positive freq spectrum
    X_positive = X[:(N//2-1)] * 2
    freq_positive = freq[:(N//2-1)]

    print("Sample dtype: ", type(samples[0]))
    print("FFT dtype: ", type(X[0]))
    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Plot the FFT amplitude spectrum
    axs[0].stem(freq_positive, np.abs(X_positive), 'b', markerfmt=" ",
                basefmt="-b", label="FFT Amplitude")
    axs[0].set_xlabel('Frequency k (Hz)')
    axs[0].set_ylabel('FFT Power |Arms^2_k|')
    axs[0].set_title('Single-sided FFT Power Spectrum')

    # Plot the amplitude of the audio signal
    axs[1].plot([i for i in range(len(samples))], samples, label="Amplitude")
    axs[1].set_xlabel('Sample (m)')
    axs[1].set_ylabel('Amplitude (A_m)')
    axs[1].set_title('Audio Signal')

    # Add a legend to the second subplot
    # axs[1].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()


def plot_fft_mp3(samples, sample_rate):
    N = len(samples)
    # N = 128
    X = fft(samples)
    n = np.arange(N)
    T = N / sample_rate
    freq = n / T

    # Plot the FFT amplitude spectrum
    plt.figure(figsize=(12, 6))
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()


def peak_detect(samples):
    threshold = np.int16(0.8 * samples.max())
    print(threshold)
    return np.where(samples > threshold)[0]


def split_windows(window_width, peak_samples_idx):
    current_window_filter = np.where(peak_samples_idx < peak_samples_idx[0] +
                                     window_width)
    print(current_window_filter)
    print(peak_samples_idx[current_window_filter])


def bird_sound_generate():
    # Simulates a bird sound recieved at a number of microphones
    # With correlated noise
    length = 2**14
    sample_rate = 20E3
    samples = np.zeros(length)
    sample_idx = np.arange(length)
    X = fft(samples)
    freq = sample_idx * sample_rate / length
    idx_of_1k = int(1E3 * length // sample_rate)
    X[idx_of_1k] = 1E3

    samples = ifft(X)

    plot_samples(samples, sample_rate, 2)

    # Plot the FFT amplitude spectrum
    # plt.figure(figsize=(12, 6))
    # plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('FFT Amplitude |X(freq)|')
    # plt.show()


if __name__ == "__main__":
    constant_velocity_tacoma()
    print("Tacoma acceleration: ", tacoma())
    # plot_mp3()
    samples, sample_rate, resolution = read_mp3("../../data/birds.mp3")
    # plot_samples(samples, sample_rate, resolution)
    peak_samples_idx = peak_detect(samples)
    split_windows(2**14, peak_samples_idx)
    bird_sound_generate()
    inference()
