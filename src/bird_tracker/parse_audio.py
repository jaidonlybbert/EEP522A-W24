import pydub
import numpy as np


def read_mp3(file_path) -> np.array:
    audio = pydub.AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate  # Hertz
    resolution = audio.sample_width  # resolution in bytes
    print("Samples: ", samples)
    print("Fs: ", sample_rate)
    print("Resolution: ", resolution)
    return samples, sample_rate, resolution


if __name__ == "__main__":
    read_mp3("../../data/calidris_falcinellus.mp3")
