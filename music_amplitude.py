from scipy.io import wavfile
import numpy as np

rate, data = wavfile.read('Pharrell_Williams-Happy.wav')

data2 = np.abs(data[::rate//1000])
