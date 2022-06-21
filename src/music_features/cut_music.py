from pydub import AudioSegment


def cut_music(t1, t2, audio_path):
    t1 = t1 * 1000  # Works in milliseconds
    t2 = t2 * 1000

    newAudio = AudioSegment.from_wav(audio_path)
    newAudio = newAudio[t1:t2]

    newAudio.export('cut.wav', format="wav")
