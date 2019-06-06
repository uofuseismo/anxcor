from typing import List, Dict
from obspy.core import Trace, Stats
import numpy as np
from numpy.fft import fft, ifft, fftshift

class BaseSingleThreadCorrelate:


    def __init__(self,tau_shift):
        self._max_t = tau_shift


    def __call__(self, trace_list: List[Trace]):
        corr_list = []
        for source in trace_list:
            for receiver in trace_list:
                if self._valid_corr_pair(source,receiver):
                    header      = self.combine_headers(source.stats, receiver.stats)
                    correlation = self.correlate(source.data, receiver.data, source.stats.sampling_rate)
                    corr_trace = Trace(data=correlation, header=header)
                    corr_list.append(corr_trace)

        return corr_list


    def correlate(self,source, receiver, sampling_rate)-> np.ndarray:
        #TODO: pad with magic numbers!
        f1 = fft(source)
        f2 = fft(np.flip(receiver))
        cc = np.real(ifft(f1 * f2))
        cc = fftshift(cc)
        zero_index = int(len(source) / 2) -1
        tau_index  = int(sampling_rate*self._max_t)
        preserved_correlation = cc[zero_index-tau_index:zero_index+tau_index+1]
        return preserved_correlation

    def _valid_corr_pair(self,source: Trace, receiver: Trace)-> bool:
        station1 = source.stats.station
        station2 = receiver.stats.station
        network1 = source.stats.network
        network2 = receiver.stats.network
        if station1==station2 and network1==network2:
            return False
        return True

    def combine_headers(self, source_stats: Stats, receiver_stats: Stats) -> Dict:

        station1 = source_stats.station
        station2 = receiver_stats.station

        network1 = source_stats.network
        network2 = receiver_stats.network

        channel1 = source_stats.channel
        channel2 = receiver_stats.channel

        delta    = source_stats.delta

        station = station1 + '.' + station2
        network = network1 + '.' + network2
        channel = channel1 + '.' + channel2

        return {'station' : station,
                'network': network,
                'channel' : channel,
                'delta':delta }
