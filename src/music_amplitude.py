import math

import numpy as np
from scipy.io import wavfile


def get_music_amplitude(audio_path, boundaries):
    rate, data = wavfile.read(audio_path)

    data = np.abs(data)[:, 0]

    amplitudes = []

    for i, boundary in enumerate(boundaries):
        cut = math.floor(boundary * rate)
        previous_cut = 0 if (i == 0) else math.floor(boundaries[i - 1] * rate)

        amplitudes.append(np.mean(data[previous_cut:cut]))

    return amplitudes
