import numpy as np
from matplotlib import pyplot as plt

from aggregation_structure_amplitude import aggregate_structure_amplitude
from audio_sequence import get_audio_sequence
from beat_tracking import get_beats
from cut_music import cut_music
from music_amplitude import get_music_amplitude
from music_structure import get_structure


def music_features(audio_path, duration, sampling_rate, t_before_start, printing=True, plotting=True):

    boundaries, labels = get_structure(audio_path, sampling_rate)

    amplitudes = get_music_amplitude(audio_path, boundaries)
    beat_times = get_beats(audio_path)

    all_segments, picked_segments, beat_peak = get_audio_sequence(
        boundaries, labels, amplitudes, beat_times, duration, t_before_start)

    beat_start = picked_segments[0]['beats'][0]
    beat_end = picked_segments[-1]['beats'][-1]

    offset_start = round(abs(beat_start - picked_segments[0]['t_start']), 1)
    offset_end = round(abs(picked_segments[-1]['t_end'] - beat_end), 1)

    cut_music(beat_start, beat_end, audio_path)

    # print the results
    if (printing == True):
        print('----------------------------------------------------------------------')
        print('ALL SEGMENTS:')
        print(all_segments)
        print('----------------------------------------------------------------------')

        print('----------------------------------------------------------------------')
        print('PICKED SEGMENTS:')
        print(picked_segments)
        print('----------------------------------------------------------------------')

        print('----------------------------------------------------------------------')
        print('BEAT START:')
        print('-> ' + str(beat_start))
        print('BEAT PEAK:')
        print('-> ' + str(beat_peak))
        print('BEAT END:')
        print('-> ' + str(beat_end))
        print('----------------------------------------------------------------------')

        print('----------------------------------------------------------------------')
        print('OFFSET START:')
        print('-> ' + str(offset_start))
        print('OFFSET END:')
        print('-> ' + str(offset_end))
        print('----------------------------------------------------------------------')

    # plot the cut
    time, aggregation = aggregate_structure_amplitude(boundaries, amplitudes)

    if (plotting == True):
        plt.plot(time, aggregation)

        plt.vlines(beat_start, 0, np.max(aggregation),
                   linestyles="dashed", colors="red")

        plt.vlines(beat_peak, 0, np.max(aggregation),
                   linestyles="dashed", colors="green")

        plt.vlines(beat_end, 0, np.max(aggregation),
                   linestyles="dashed", colors="red")

        plt.title(f'Music Structure of: {audio_path}')
        plt.xlabel('Time in seconds')
        plt.ylabel('Amplitude')
        plt.show()

    return all_segments, picked_segments, beat_start, beat_peak, beat_end, offset_start, offset_end
