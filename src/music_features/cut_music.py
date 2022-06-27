from pydub import AudioSegment
import pathlib


def cut_music(t1, t2, audio_path):
    """cuts the audio file on the sequence

    Args:
        t1 (float): starting time for cutting
        t2 (float): end of cut
        audio_path (string): path of the audio file
    """

    t1 = t1 * 1000  # Works in milliseconds
    t2 = t2 * 1000

    newAudio = AudioSegment.from_wav(audio_path)
    newAudio = newAudio[t1:t2]

    newAudio.export(pathlib.Path(__file__).parent.joinpath(
    "audio_sequence.wav"), format="wav")