from music_features.config.main import (AUDIO_PATH, DURATION, SAMPLING_RATE,
                                        T_BEFORE_START)
from music_features.main import music_features
from music_features.beat_tracking import get_downbeats

AUDIO_PATH = str(AUDIO_PATH)

downbeats_frequency, downbeat_times = get_downbeats(AUDIO_PATH)

all_segments, t_start_sequence, t_downbeat_max, t_end_sequence = music_features(
    AUDIO_PATH, DURATION, SAMPLING_RATE, T_BEFORE_START, downbeat_times, printing=True, plotting=True)
