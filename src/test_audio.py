from music_features.config.main import AUDIO_PATH, DURATION, SAMPLING_RATE, T_BEFORE_START
from music_features.main import music_features


all_segments, picked_segments, beat_start, t_peak, beat_end, offset_start, offset_end = music_features(
    AUDIO_PATH, DURATION, SAMPLING_RATE, T_BEFORE_START, printing=True, plotting=True)