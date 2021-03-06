import pathlib
import sys
import warnings

from music_features.beat_tracking import get_downbeats
from music_features.config.main import AUDIO_PATH, SAMPLING_RATE
from music_features.main import music_features
from video_features.generate_summary import (create_summary,
                                             summary_frames_selection)
from video_features.main import (create_clip, ordering_videos,
                                 read_and_save_frames, summary_param)
from video_features.score import score
from video_features.shot_detection import define_shots

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("""
        Four arguments are required:
            - The first argument should be the path to the videos directory
            - The second argument should be the name of the audio file (e.g. PharrellWilliams_Happy)
            - The third argument is the name of the clip
            - The fourth argument is an integer for the summary duration time in seconds
            - The fifth argument is an integer for the percentage of one shot summary
            """)
        sys.exit(0)

    videos_path = sys.argv[1]
    music = str(sys.argv[2]) + '.wav'
    clip_filename = sys.argv[3]
    summary_duration = int(sys.argv[4])
    shot_percentage = int(sys.argv[5])

    summary_path = pathlib.Path(__file__).parent.joinpath(
        'video_features/summary.mp4')
    audio_sequence_path = pathlib.Path(__file__).parent.joinpath(
        'music_features/audio_sequence.wav')

    AUDIO_PATH = pathlib.Path(__file__).parent.joinpath(
        'music_features/audio_samples').joinpath(music)
    AUDIO_PATH = str(AUDIO_PATH)  # convert pathlib to string

    videos_order = ordering_videos(videos_path)

    frames_list, summary_fps, summary_resolution = read_and_save_frames(
        videos_order)

    downbeats_frequency, downbeat_times = get_downbeats(AUDIO_PATH)

    nb_shots = 10
    shots = define_shots(frames_list, summary_fps, nb_shots,
                         shot_percentage, downbeats_frequency, show_viz=True)

    dict_shots_order = score(frames_list, shots, sampling_rate=10)

    min_shot_nb = len(videos_order)

    summary_frames_index, time_before_drop, summary_duration = summary_frames_selection(
        summary_duration, summary_fps, shot_percentage, dict_shots_order, min_shot_nb, downbeats_frequency)

    print('time before drop:', time_before_drop)
    print("total duration summary", summary_duration)

    all_segments, t_start_sequence, t_downbeat_max, t_end_sequence = music_features(
        AUDIO_PATH, summary_duration, SAMPLING_RATE, time_before_drop, downbeat_times, printing=False, plotting=False)

    create_summary(frames_list, summary_frames_index,
                   summary_path, summary_resolution, summary_fps)

    create_clip(summary_video_path=summary_path,
                audio_path=audio_sequence_path, clip_filename=clip_filename)
