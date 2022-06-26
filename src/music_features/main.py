import numpy as np
from matplotlib import pyplot as plt

from music_features.aggregation_structure_amplitude import \
    aggregate_structure_amplitude
from music_features.audio_sequence import get_audio_sequence
from music_features.beat_tracking import get_beats, get_downbeats
from music_features.cut_music import cut_music
from music_features.music_amplitude import get_music_amplitude
from music_features.music_structure import get_structure


def music_features(audio_path, duration, sampling_rate, t_before_start, downbeat_times, printing=True, plotting=True):
    """Get the audio features and cuts the audio

    Args:
        audio_path (string): path of the audio file
        duration (int): duration of the editing
        sampling_rate (int): sampling rate for analyzing the audio
        t_before_start (float): time before the peak of the score
        printing (bool, optional): prints the results. Defaults to True.
        plotting (bool, optional): prints the curve. Defaults to True.

    Returns:
        tuple: all the audio descriptors
    """

    boundaries, labels = get_structure(audio_path, sampling_rate)

    amplitudes = get_music_amplitude(audio_path, boundaries)
    beat_times = get_beats(audio_path)

    all_segments, t_start_sequence, t_downbeat_max, t_end_sequence = get_audio_sequence(
        boundaries, labels, amplitudes, beat_times, downbeat_times, duration, t_before_start)

    cut_music(t_start_sequence, t_end_sequence, audio_path)

    # print the results
    if (printing == True):
        print('----------------------------------------------------------------------')
        print('ALL SEGMENTS:')
        print(all_segments)
        print('----------------------------------------------------------------------')

        print('----------------------------------------------------------------------')
        print('DOWNBEAT START:')
        print('-> ' + str(t_start_sequence))
        print('DOWNBEAT PEAK:')
        print('-> ' + str(t_downbeat_max))
        print('DOWNBEAT END:')
        print('-> ' + str(t_end_sequence))
        print('----------------------------------------------------------------------')

    # plot the cut
    time, aggregation = aggregate_structure_amplitude(boundaries, amplitudes)

    if (plotting == True):
        plt.plot(time, aggregation)

        plt.vlines(t_start_sequence, 0, np.max(aggregation),
                   linestyles="dashed", colors="red")

        plt.vlines(t_downbeat_max, 0, np.max(aggregation),
                   linestyles="dashed", colors="green")

        plt.vlines(t_end_sequence, 0, np.max(aggregation),
                   linestyles="dashed", colors="red")

        plt.title(f'Music Structure of: {audio_path}')
        plt.xlabel('Time in seconds')
        plt.ylabel('Amplitude')
        plt.show()

    return all_segments, t_start_sequence, t_downbeat_max, t_end_sequence
