from typing import List
from obspy.core import Trace
import numpy as np
import pandas as pd

# TODO: add an option for each filter to do a taper before each step

class Taper:

    def __init__(self,type='hann',percent=0.05):
        self.type = type
        self.percent = percent

    def __call__(self, trace_list: List[Trace],*args):
        new_list = []
        for trace in trace_list:
            trace.taper(type=self.type,max_percentage=self.percent)
            new_list.append(trace)
        return new_list


class Response:

    def __init__(self,response_file):
        self.response_file=response_file


    def __call__(self, taper_list: List[Trace],*args):
        new_traces=[]
        for trace in taper_list:
            trace.remove_response()
            new_traces.append(new_traces)
        return  new_traces

class OneBit:


    def __init__(self):
        pass


    def __call__(self,trace_list: List[Trace],*args):
        new_list = []
        for trace in trace_list:
            trace.data = np.sign(trace.data)
            new_list.append(trace)

        return new_list


class RemoveMeanTrend:

    def __init__(self):
        pass


    def __call__(self,trace_list: List[Trace],*args):
        new_list = []
        for trace in trace_list:
            trace.detrend(type='demean')
            trace.detrend(type='linear')
            new_list.append(trace)

        return new_list


class BandPass:

    def __init__(self,freqmin, freqmax, zerophase=True, corners=2):
        self.freqmin=freqmin
        self.freqmax=freqmax
        self.zerophase=zerophase
        self.corners = corners

    def __call__(self, trace_list: List[Trace],*args):
        new_list = []
        for trace in trace_list:
            trace.filter('bandpass',freqmax=self.freqmax,
                                    freqmin=self.freqmax,
                                    zerophase=self.zerophase,
                                    corners=self.corners)
            new_list.append(trace)
        return new_list


class RunningAbsoluteMeanNorm:


    def __init__(self,time_window):
        self.time_window=time_window


    def __call__(self, trace_list,*args):
        new_list = []
        samples  = self._calculate_samples(trace_list[0])
        for trace in trace_list:
            abs_mean    = self._rolling_mean(trace.data,samples)
            trace.data /= abs_mean
            new_list.append(trace)
        return new_list


    def _rolling_mean(self,data,samples):
        rma = np.abs(pd.Series(data).rolling(window=samples).mean().iloc[samples - 1:].values)
        return rma

    def _calculate_samples(self,trace):
        delta = trace.stats.delta
        return self.time_window/delta


class SpectralWhiten(object):
    #TODO: Implement spectral whitening
    def __init__(self):
       pass


class Resample:
    def __init__(self,target, zerophase=True, corners=2):
        self.target = target
        self.zerophase = zerophase
        self.corners = corners

    def __call__(self, trace_list: List[Trace],*args):
        new_list = []
        nyquist  = self._calculate_nyquist(trace_list[0])
        for trace in trace_list:
            trace.filter('lowpass', freq=nyquist,df=nyquist*2,
                         zerophase=self.zerophase,
                         corners=self.corners)

            trace.interpolate(sampling_rate=self.target,method='linear')
            new_list.append(trace)
        return new_list

    def _calculate_nyquist(self, trace):
        return trace.stats.sampling_rate/2


class BergNorm:

    def __init__(self,  freqmin=1 / 50.0,
                        freqmax=1 / 15.0,
                        time_window=128,
                        corners=2,
                        zerophase=True):
        self.freqmin        = freqmin
        self.freqmax        = freqmax
        self.time_window    = time_window
        self.zerophase = zerophase
        self.corners = corners


    def __call__(self, trace_list,*args):
        new_traces     = self._initial_bp(trace_list)
        abs_inv_means  = self._rolling_mean(new_traces)
        normed_traces  = self._normalize(trace_list,abs_inv_means)
        return normed_traces

    def _initial_bp(self,trace_list):

        new_traces = []
        for trace in trace_list:
            trace.filter('bandpass', freqmax=self.freqmax,
                         freqmin=self.freqmax,
                         zerophase=self.zerophase,
                         corners=self.corners)
            new_traces.append(new_traces)

        return new_traces

    def _rolling_mean(self,trace_list,samples):
        averaging_samples = self._calculate_samples(trace_list[0])

        components    = len(trace_list)
        total_samples = max(trace_list[0].data.size)

        rma_matrix = np.zeros((components,total_samples))
        for index,trace in enumerate(trace_list):
            rma_matrix[index,:] = np.abs(pd.Series(trace.data).rolling(window=averaging_samples).mean().iloc[samples - 1:].values)

        return np.amax(rma_matrix,axis=0)

    def _normalize(self,trace_list,norm_array):
        new_list = []
        for trace in trace_list:
            trace.data /= norm_array
            new_list.append(trace)
        return new_list

    def _calculate_samples(self,trace):
        delta = trace.stats.delta
        return self.time_window/delta