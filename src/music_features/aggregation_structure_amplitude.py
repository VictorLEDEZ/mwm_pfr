import numpy as np


def aggregate_structure_amplitude(boundaries, amplitudes):
    """make the aggregation to get a score and the time array

    Args:
        boundaries (array): boundaries of the audio file
        amplitudes (array): amplitudes of the audio file

    Returns:
        tuple: the aggregation as a function of the time
    """

    amplitudes = np.around(amplitudes, decimals=0)
    boundaries = np.around(boundaries, decimals=1)

    time = list(range(0, int(boundaries[-1]) * 10))
    aggregation = []

    for i, boundary in enumerate(boundaries):
        aggregation = aggregation + [amplitudes[i]] * \
            int((boundary - (0 if (i == 0) else boundaries[i - 1])) * 10)

    shift = len(time) - len(aggregation)
    if (shift < 0):
        aggregation = aggregation[abs(shift):]
    else:
        time = time[abs(shift):]

    time = [x / 10 for x in time]

    return time, aggregation
