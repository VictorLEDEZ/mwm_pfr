import IPython
import librosa
import librosa.display
import numpy as np

from config.main import AUDIO_PATH


def get_beats(audio_path):
    x, sr = librosa.core.load(audio_path)
    _, beat_times = librosa.beat.beat_track(x, sr=sr, units='time')

    return beat_times
