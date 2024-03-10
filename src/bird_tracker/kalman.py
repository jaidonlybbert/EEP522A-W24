import pydub
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


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
