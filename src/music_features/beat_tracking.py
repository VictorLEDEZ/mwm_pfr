import librosa
import librosa.display


def get_beats(audio_path):
    x, sr = librosa.core.load(audio_path)
    _, beat_times = librosa.beat.beat_track(x, sr=sr, units='time')

    return beat_times