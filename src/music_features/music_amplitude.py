import math

import numpy as np
from scipy.io import wavfile


def get_music_amplitude(audio_path, boundaries):
    """get the amplitudes of the audio file

    Args:
        audio_path (string): path of the audio file
        boundaries (array): boundaries of the audio file

    Returns:
        array: amplitudes of the audio file
    """

    rate, data = wavfile.read(audio_path)

    data = np.abs(data)[:, 0]

    amplitudes = []

    for i, boundary in enumerate(boundaries):
        cut = math.floor(boundary * rate)
        previous_cut = 0 if (i == 0) else math.floor(boundaries[i - 1] * rate)

        amplitudes.append(np.mean(data[previous_cut:cut]))

    return amplitudes
