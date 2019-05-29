from obspy.core import Trace, UTCDateTime
import numpy as np
from scipy import  signal
from scipy import fftpack
import pandas as pd
import matplotlib.pyplot as plt

def create_random_trace(sampling_rate=40.0, duration = 5.0):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime()
            }
    data  = np.random.uniform(-1,1,(int(duration*sampling_rate)))
    return Trace(data=data,header=header)

def create_triangle_trace(sampling_rate=40.0, duration = 5.0):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime()
            }
    x = np.linspace(0, duration, num=int(duration * sampling_rate))
    data  = signal.triang(int(duration*sampling_rate))
    return Trace(data=data,header=header)


def create_sinsoidal_trace(sampling_rate=40.0, period=0.5, duration = 5.0):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime()
            }
    x     = np.linspace(0, duration,num=int(duration*sampling_rate))
    data  = np.sin( x * 2 * np.pi / period )

    return Trace(data=data,header=header)

def plot_spectrum(trace):
    sample_rate = trace.stats['sampling_rate']
    amplitudes = fftpack.fft(trace.data)
    amplitudes = np.abs(amplitudes * np.conjugate(amplitudes))

    padded_data = np.pad(amplitudes, pad_width=15, mode='constant')
    amplitudes  = pd.Series(padded_data).rolling(window=30).mean().iloc[30 :].values
    amplitudes /= np.max(amplitudes)
    freqs = fftpack.fftfreq(len(trace.data)) * sample_rate

    plt.figure()
    plt.plot(freqs, amplitudes)

    plt.xlim([-sample_rate / 2, sample_rate / 2])
    plt.show()
