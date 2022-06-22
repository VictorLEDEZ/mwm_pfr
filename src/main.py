import sys

import score

from music_features.config.main import AUDIO_PATH, SAMPLING_RATE
from music_features.main import music_features
from video_features.generate_summary import (create_summary,
                                             summary_frames_selection)
from video_features.main import (ordering_videos, read_and_save_frames,
                                 summary_param, create_clip)
from video_features.shot_detection import define_shots


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("""
        Three arguments are required:
            - The first argument should be the path to the videos directory
            - The second argument is the name of the summary output file
            - The third argument is an integer for the summary duration time in seconds
            - The fourth argument is an integer for the percentage of one shot summary
            """)
        sys.exit(0)

    dir_path = sys.argv[1]
    clip_filename = sys.argv[2]
    summary_duration = int(sys.argv[3])
    shot_percentage = int(sys.argv[4])
    summary_path = 'src/video_features/summary'
    audio_path = 'src/music_features/audio_sequence.wav'

    videos_order = ordering_videos(dir_path)

    frames_list, videos_param = read_and_save_frames(videos_order)

    nb_shots = 16
    shots = define_shots(frames_list, videos_param,
                         nb_shots, shot_percentage, show_viz=True)

    summary_fps, summary_resolution = summary_param(videos_param)

    dict_shots_order = score(frames_list, shots, sampling_rate=10)

    min_shot_nb = len(videos_param)

    summary_frames_index, time_before_drop, summary_duration = summary_frames_selection(
        summary_duration, summary_fps, shot_percentage, dict_shots_order, min_shot_nb)

    print('time before drop:', time_before_drop)
    print("total duration summary", summary_duration)

    all_segments, picked_segments, beat_start, t_peak, beat_end, offset_start, offset_end = music_features(
        AUDIO_PATH, summary_duration, SAMPLING_RATE, time_before_drop, printing=False, plotting=False)

    summary_frames_index = summary_frames_index[summary_fps*offset_start:len(
        summary_frames_index)-(summary_fps*offset_end)]

    create_summary(frames_list, summary_frames_index,
                   summary_path, summary_resolution, summary_fps)

    create_clip(summary_video_path=summary_path,audio_path=audio_path, clip_filename=clip_filename)
