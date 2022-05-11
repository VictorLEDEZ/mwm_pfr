from music_structure import get_structure
import matplotlib.pyplot as plt

SAMPLING_RATE = 11025
AUDIO_PATH = 'Pharrell_Williams-Happy.wav'

boundaries, labels = get_structure(AUDIO_PATH, SAMPLING_RATE)

plt.plot(boundaries, labels)
plt.show()

for label, boundary in zip(labels, boundaries):
    print(f'label {label:.0f} at t = {boundary:.1f} s')
