from config.main import AUDIO_PATH, DURATION, SAMPLING_RATE, T_BEFORE_START
from main import music_features

all_segments, picked_segments, beat_start, beat_peak, beat_end, offset_start, offset_end = music_features(
    AUDIO_PATH, DURATION, SAMPLING_RATE, T_BEFORE_START, printing=True, plotting=True)
