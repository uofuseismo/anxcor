from obspy.core import Trace, UTCDateTime
import numpy as np
from scipy import  signal
from scipy import fftpack
import pandas as pd



def linear_ramp_trend(sampling_rate=40.0, duration = 5.0,height=1.0,mean=0.5):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime(0),
        'channel': 'Z',
        'station': 'test'
            }
    data  = np.random.uniform(-1,1,(int(duration*sampling_rate)))
    data += np.linspace(0,height,num=int(duration*sampling_rate))
    data += mean*np.ones(int(duration*sampling_rate))
    trace = Trace(data=data,header=header)
    trace.stats.data_type='test'
    return trace


def create_random_trace(sampling_rate=40.0, duration = 5.0, **header_kwargs):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime(0),
        'channel': 'Z',
        'station': 'test'
    }
    header = {**header, **header_kwargs}
    data  = np.random.uniform(-1,1,(int(duration*sampling_rate)))
    trace = Trace(data=data, header=header)
    trace.stats.data_type = 'test'
    return trace

def create_triangle_trace(sampling_rate=40.0, duration = 5.0,**header_kwargs):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime(0),
        'channel': 'Z',
        'station': 'test'
            }
    header = {**header,**header_kwargs}
    x = np.linspace(0, duration, num=int(duration * sampling_rate))
    data  = signal.triang(int(duration*sampling_rate))
    trace = Trace(data=data, header=header)
    trace.stats.data_type = 'test'
    return trace


def create_sinsoidal_trace(sampling_rate=40.0, period=0.5, duration = 5.0,**header_kwargs):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime(0),
        'channel'       : 'Z',
        'station'       : 'test'
            }
    header = {**header, **header_kwargs}
    x     = np.linspace(0, duration,num=int(duration*sampling_rate))
    data  = np.sin( x * 2 * np.pi / period )

    trace = Trace(data=data, header=header)
    trace.stats.data_type = 'test'
    return trace

def create_sinsoidal_trace_w_decay(sampling_rate=40.0,decay=0.01, period=0.5, duration = 5.0,**header_kwargs):
    header={
        'sampling_rate' : sampling_rate,
        'starttime'     : UTCDateTime(0),
        'channel': 'Z',
        'station': 'test'
            }
    header = {**header, **header_kwargs}
    x     = np.linspace(0, duration,num=int(duration*sampling_rate))
    data  = np.sin( x * 2 * np.pi / period ) * np.exp( -decay*(x-duration/2)**2)

    trace = Trace(data=data, header=header)
    trace.stats.data_type = 'test'
    return trace

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
