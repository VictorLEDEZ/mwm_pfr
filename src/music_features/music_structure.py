import math

import librosa
from sf_segmenter.feature import audio_extract_pcp
from sf_segmenter.segmenter import Segmenter


def get_structure(audio_path, sampling_rate):
    """Get the boundaries and the labels from an audio file

    Args:
        audio_path (string): path from the audio file
        sampling_rate (int): sampling rate for analyzing the audio

    Returns:
        tuple: the boundaries and the labels
    """

    y, sr = librosa.load(audio_path, sr=sampling_rate)
    pcp = audio_extract_pcp(y, sr)
    segmenter = Segmenter()
    boundaries, labels = segmenter.process(pcp, is_label=True)
    boundaries = [x / (boundaries[-1] /
                       (len(y) / sampling_rate)) for x in boundaries][1:]
    return boundaries, labels
