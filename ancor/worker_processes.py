from typing import List
from obspy.core import Trace
import numpy as np
import pandas as pd
import scipy.fftpack as fft

class AncorProcessorBase:

    def __init__(self):
        pass

    def __call__(self, trace_list: List[Trace],*args):
        trace_list = self._all_component_ops(trace_list)
        trace_list = self._per_component_ops(trace_list)
        return trace_list

    def _all_component_ops(self,trace_list: List[Trace])-> List[Trace]:
        return trace_list

    def _per_component_ops(self,trace_list: List[Trace])-> List[Trace]:
        for trace in trace_list:
            self._operate_per_trace(trace)
        return trace_list

    def _operate_per_trace(self, trace: Trace) -> Trace:
        return trace




class Taper(AncorProcessorBase):

    def __init__(self,type='hann',percent=0.05):
        """

        Parameters
        ----------
        type: str
            filter type. defaults to "hann" for hanning
        percent: float
            percent off each end to taper
        """
        super().__init__()
        self.type = type
        self.percent = percent

    def _operate_per_trace(self, trace: Trace) ->Trace:
        trace.taper(type=self.type,max_percentage=self.percent)
        return trace


class Response(AncorProcessorBase):


    def __init__(self, **kwargs):
        super().__init__()
        self.resp_kwargs = kwargs

    def _operate_per_trace(self, trace: Trace) ->Trace:
        trace.remove_response(**self.resp_kwargs)
        return trace


class OneBit(AncorProcessorBase):


    def __init__(self):
        super().__init__()

    def _operate_per_trace(self, trace: Trace) ->Trace:
        trace.data = np.sign(trace.data)
        return trace


class RemoveMeanTrend(AncorProcessorBase):

    def __init__(self):
        super().__init__()

    def _operate_per_trace(self, trace: Trace) ->Trace:
        trace.detrend(type='demean')
        trace.detrend(type='linear')
        return trace


class BandPass(AncorProcessorBase):

    def __init__(self, freqmin: float, freqmax: float, zerophase: bool = True, corners: int = 2):
        super().__init__()
        self.freqmin  =freqmin
        self.freqmax  =freqmax
        self.zerophase=zerophase
        self.corners  = corners


    def _operate_per_trace(self, trace: Trace) -> Trace:
        trace.filter('bandpass',
                     freqmax=self.freqmax,
                     freqmin=self.freqmin,
                     zerophase=self.zerophase,
                     corners=self.corners)
        return trace


class RunningAbsoluteMeanNorm(AncorProcessorBase):


    def __init__(self,time_window):
        super().__init__()
        self.time_window=time_window


    def _operate_per_trace(self, trace: Trace) -> Trace:
        samples     = self._calculate_samples(trace)
        abs_mean    = self._rolling_mean(trace.data, samples)
        trace.data /= abs_mean

        return trace

    def _rolling_mean(self,data,samples):
        padded_data = np.pad(data,pad_width=samples/2,mode='constant')
        rma = np.abs(pd.Series(padded_data).rolling(window=samples).mean().iloc[samples:].values)
        return rma

    def _calculate_samples(self,trace):
        delta = trace.stats.delta
        return self.time_window/delta


class SpectralWhiten(AncorProcessorBase):

    def __init__(self,frequency_smoothing_interval,taper_percent):
        """
            Whitens the frequency response by dividing the original spectrum by
            a running mean of the absolute value of the original spectrum.
        Parameters
        ----------
        frequency_smoothing_interval: float
            smoothing interval (in hz)
        taper_percent:
            percent taper to perform prior to whitening
        """
        super().__init__()
        self.smoothing_interval=frequency_smoothing_interval
        self.percent = taper_percent


    def _operate_per_trace(self, trace: Trace) -> Trace:
        trace.taper(self.percent)
        array_len            = trace.stats.npts
        time_domain_original = trace.data
        delta                = trace.stats.delta

        target_width = fft.next_fast_len(array_len)

        spectrum    = fft.fftshift(fft.fft(time_domain_original.data,target_width))
        frequencies = fft.fftfreq(array_len, d=delta)

        smoothing_pnts = int(-(-self.smoothing_interval//frequencies[1]))
        convolve_ones  = np.ones((smoothing_pnts,))/smoothing_pnts
        running_spec   = np.convolve(np.abs(spectrum), convolve_ones, mode='same')

        spectrum      /= running_spec
        spectrum       = fft.ifftshift(spectrum)

        trace.data     = np.real(fft.ifft(spectrum))[:array_len]

        return trace


class Downsample(AncorProcessorBase):
    def __init__(self, target_rate, zerophase=True, corners=2, interpolation='weighted_average_slopes'):
        """
            Resamples traces. order is:
            - lowpass filter below the nyquist
            - linearly interpolate
        Parameters
        ----------
        target_rate: float
            target sampling rate
        zerophase: bool
            if true, the resulting lowpass filter will not result in a phase shift
        corners: int
            number of corner frequencies in the filter
        interpolation: str
            interpolation type to perform after filtering. defaults to 'linear'
        """
        super().__init__()
        self.target = target_rate
        self.zerophase = zerophase
        self.corners = corners
        self.interpolation = interpolation

    def _operate_per_trace(self, trace: Trace) -> Trace:
        nyquist = self._calculate_nyquist(trace)
        trace.filter('lowpass',
                     freq=nyquist,
                     zerophase      =self.zerophase,
                     corners        =self.corners)
        trace.interpolate(sampling_rate=self.target, method=self.interpolation)

        return trace

    def _calculate_nyquist(self, trace):
        return trace.stats['sampling_rate']/2.0001


class MaxMeanComponentNorm(AncorProcessorBase):

    def __init__(self,  freqmin=1 / 50.0,
                        freqmax=1 / 15.0,
                        time_window=128,
                        corners=2,
                        zerophase=True):
        super().__init__()
        self.freqmin        = freqmin
        self.freqmax        = freqmax
        self.time_window    = time_window
        self.zerophase = zerophase
        self.corners = corners

    def _per_component_ops(self,trace_list: List[Trace])->List[Trace]:
        return trace_list

    def _all_component_ops(self,trace_list: List[Trace]):
        new_traces     = self._initial_bp(trace_list)
        abs_inv_means  = self._rolling_mean(new_traces)
        normed_traces  = self._normalize(trace_list,abs_inv_means)
        return normed_traces

    def _initial_bp(self,trace_list):
        for trace in trace_list:
            trace.filter('bandpass', freqmax=self.freqmax,
                         freqmin=self.freqmax,
                         zerophase=self.zerophase,
                         corners=self.corners)

        return trace_list

    def _rolling_mean(self,trace_list,samples):
        averaging_samples = self._calculate_samples(trace_list[0])

        components    = len(trace_list)
        total_samples = max(trace_list[0].data.size)

        rma_matrix = np.zeros((components,total_samples))
        for index,trace in enumerate(trace_list):
            padded_data = np.pad(trace.data, pad_width=samples / 2, mode='constant')
            rma_matrix[index,:] = np.abs(pd.Series(padded_data).rolling(window=averaging_samples).mean().iloc[samples:].values)

        return np.amax(rma_matrix,axis=0)

    def _normalize(self,trace_list,norm_array):
        for trace in trace_list:
            trace.data /= norm_array
        return trace_list

    def _calculate_samples(self,trace):
        delta = trace.stats.delta
        return self.time_window/delta