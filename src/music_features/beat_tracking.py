import librosa
import librosa.display


def get_beats(audio_path):
    """get the beats of the audio

    Args:
        audio_path (string): path of the audio file

    Returns:
        array: array of times for each beats
    """

    x, sr = librosa.core.load(audio_path)
    _, beat_times = librosa.beat.beat_track(x, sr=sr, units='time')

    return beat_times

# import madmom

# proc = madmom.features.downbeats.RNNDownBeatProcessor()
# out = proc('src/music_features/audio_samples/PharrellWilliams_Happy.wav')

# print(out)
