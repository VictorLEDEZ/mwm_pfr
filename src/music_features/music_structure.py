import math

import librosa
from sf_segmenter.feature import audio_extract_pcp
from sf_segmenter.segmenter import Segmenter


def get_structure(audio_path, sampling_rate):
    y, sr = librosa.load(audio_path, sr=sampling_rate)
    pcp = audio_extract_pcp(y, sr)
    segmenter = Segmenter()
    boundaries, labels = segmenter.process(pcp, is_label=True)
    boundaries = [x / (boundaries[-1] /
                       (len(y) / sampling_rate)) for x in boundaries][1:]
    return boundaries, labels


def get_shots_range(boundaries, audio_length):
    mean_shots = len(boundaries) * (audio_length / boundaries[-1]) * 3
    min_shots = math.floor(mean_shots)
    max_shots = math.ceil(mean_shots)
    return min_shots, max_shots
