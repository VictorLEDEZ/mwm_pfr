import numpy as np


def aggregate_structure_amplitude(boundaries, amplitudes):
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
