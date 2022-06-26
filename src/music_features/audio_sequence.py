import numpy as np


def get_audio_sequence(boundaries, labels, amplitudes, beat_times, downbeat_times, duration, t_before_peak=15):
    """get the segments and the peak of the audio file

    Args:
        boundaries (array): boundaries in seconds of the audio file
        labels (array): labels of the segments
        amplitudes (array): amplitudes of the audio file
        beat_times (array): beats within the audio file
        duration (float): duration of the picked sequence
        t_before_peak (int, optional): time before the peak. Defaults to 15.

    Returns:
        tuple: the segments and the peak time
    """

    boundaries = np.around(boundaries, decimals=1)
    amplitudes = np.around(amplitudes, decimals=1)
    beat_times = np.around(beat_times, decimals=1)
    downbeat_times = np.around(downbeat_times, decimals=1)
    duration = np.around(duration, decimals=1)
    t_before_peak = np.around(t_before_peak, decimals=1)

    # I] CREATING THE DICTIONARY
    all_segments = []
    for i, boundary in enumerate(boundaries):
        t_start = boundaries[i - 1] if (i > 0) else 0

        segment = {
            "mean_amplitude": amplitudes[i],
            "t_start": t_start,
            "t_end": boundary,
            "beats": list(filter(lambda beat_time: (beat_time >= t_start) & (beat_time < boundary), beat_times)),
            "downbeats": list(filter(lambda downbeat_time: (downbeat_time >= t_start) & (downbeat_time < boundary), downbeat_times)),
        }

        all_segments.append(segment)

    # II] PICKING THE RIGHT SEQUENCE WITH THE CORRESPONDING SEGMENTS
    max_amplitude_index = np.argmax(amplitudes)
    max_amplitude_label = labels[max_amplitude_index]

    while(max_amplitude_label == labels[max_amplitude_index - 1]):
        max_amplitude_index = max_amplitude_index - 1

    t_start_max = boundaries[max_amplitude_index - 1]

    downbeat_peak_after = all_segments[max_amplitude_index]["downbeats"][0]
    downbeat_peak_before = all_segments[max_amplitude_index -
                                        1]["downbeats"][-1]

    t_downbeat_max = min([downbeat_peak_before, downbeat_peak_after],
                         key=lambda x: abs(x-t_start_max))

    t_start_sequence = t_downbeat_max - t_before_peak
    t_end_sequence = t_start_sequence + duration

    # picked_segments = []
    # for i, segment in enumerate(all_segments):

    #     if ((t_start_sequence <= segment['t_start'] <= t_end_sequence) | (t_start_sequence <= segment['t_end'] <= t_end_sequence)):

    #         t_start = t_start_sequence if (
    #             t_start_sequence >= segment['t_start']) else segment['t_start']
    #         t_end = t_end_sequence if (
    #             t_end_sequence <= segment['t_end']) else segment['t_end']

    #         picked_segment = {
    #             "mean_amplitude": amplitudes[i],
    #             "t_start": t_start,
    #             "t_end": t_end,
    #             "beats": list(filter(lambda beat_time: (beat_time >= t_start) & (beat_time < t_end), beat_times)),
    #             "downbeats": list(filter(lambda downbeat_time: (downbeat_time >= t_start) & (downbeat_time < t_end), downbeat_times)),
    #         }

    #         picked_segments.append(picked_segment)

    return all_segments, t_start_sequence, t_downbeat_max, t_end_sequence
