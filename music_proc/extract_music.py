from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os

PATH = os.path.dirname(os.path.realpath(__file__))
# extract volume
data, samplerate = sf.read(PATH+'/summer_light_extract.wav')
print(data.shape)

sample = data[:,0]
mask = sample < 0.1
sample[mask] = 0
peak_indices = argrelextrema(sample,np.greater,order=2000)
print(peak_indices[0])
print(len(data[:,0]))
plt.plot(peak_indices[0], sample[peak_indices[0]], linestyle='', marker='x')
plt.plot(sample)
plt.show()

np.savetxt('summer_light_extract.txt', peak_indices[0]/48000, delimiter=',')