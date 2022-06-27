import librosa
import librosa.display
import madmom
from matplotlib import pyplot as plt


def count_dups(list):
    answer = []
    if not list:
        return answer
    running_count = 1
    for i in range(len(list)-1):
        if list[i] == list[i+1]:
            running_count += 1
        else:
            answer.append(running_count)
            running_count = 1
    answer.append(running_count)

    answer = [i for i in answer if i > 20]

    return answer


def most_frequent(list):
    counter = 0
    number = 0

    for i in list:
        curr_frequency = list.count(i)
        if(curr_frequency > counter):
            counter = curr_frequency
            number = i

    return number


def get_downbeats(audio_path):
    """get the downbeats of the audio

    Args:
        audio_path (string): path of the audio file

    Returns:
        array: array of times for each downbeats
    """

    sampling_per_seconds = 100

    proc = madmom.features.downbeats.RNNDownBeatProcessor()
    beat_probabilities = proc(audio_path)

    downbeats = [row[1] if row[1] >= row[0]
                 else 0 for row in beat_probabilities]
    downbeats = [1 if beat > 0.1 else 0 for beat in downbeats]

    len_full_audio = len(downbeats)

    duplicates = count_dups(downbeats)

    downbeats_frequency = most_frequent(duplicates)
    index_first_downbeat = duplicates[0]
    index_last_downbeat = duplicates[-1]

    len_audio = len_full_audio - (index_first_downbeat + index_last_downbeat)
    amount_downbeats = int(len_audio / (downbeats_frequency + 1))

    downbeats = ([0] * index_first_downbeat) + \
        (([1] + ([0] * downbeats_frequency)) *
         amount_downbeats) + ([0] * index_last_downbeat)

    downbeats_index = [i if downbeat == 1 else 0 for i,
                       downbeat in enumerate(downbeats)]

    plt.plot(downbeats_index)
    plt.show()

    downbeats_index = list(filter(lambda a: a != 0, downbeats_index))

    downbeat_times = [x / sampling_per_seconds for x in downbeats_index]

    downbeats_frequency = downbeats_frequency / sampling_per_seconds

    return downbeats_frequency, downbeat_times


def get_beats(audio_path):
    """get the beats of the audio

    Args:
        audio_path (string): path of the audio file

    Returns:
        array: array of times for each beats
    """

    x, sr = librosa.core.load(audio_path)
    _, beat_times = librosa.beat.beat_track(x, sr=sr, units='time')

    return beat_times
