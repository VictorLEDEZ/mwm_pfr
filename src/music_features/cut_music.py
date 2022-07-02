from pydub import AudioSegment
import pathlib

import numpy as np
import librosa
import soundfile


def apply_fadeout(audio, sr, duration=3.0):
    """
    Function that creates a fade at the end of audio
    Args:
        audio (ndarray): librosa audio ndarray
        sr (int): sample rate
        duration (float): fade duration en seconds
    """
    length = int(duration * sr)
    end = len(audio)
    start = end - length

    # linear fade
    fade_curve = np.linspace(1.0, 0.0, length)

    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve


def cut_music(t1, t2, audio_path):
    """cuts the audio file on the sequence

    Args:
        t1 (float): starting time for cutting
        t2 (float): end of cut
        audio_path (string): path of the audio file
    """
    new_audio, sr = librosa.load(audio_path)
    new_audio = new_audio[int(t1*sr):int(t2*sr)]

    apply_fadeout(new_audio, sr, duration=3.0)

    print(pathlib.Path(__file__).parent.parent.joinpath("_app_data/output/audio_sequence.wav"))
    soundfile.write(pathlib.Path(__file__).parent.parent.joinpath("_app_data/output/audio_sequence.wav"),
                    new_audio,
                    samplerate=sr)
