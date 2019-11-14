import unittest
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime, Stream, Trace
from anxcor.xarray_routines import XArrayXCorrelate, XArrayConverter, XArrayResample
import numpy as np
from anxcor.core import Anxcor
from anxcor.xarray_routines import XArrayProcessor, XArrayRemoveMeanTrend, XArrayComponentNormalizer
from anxcor.containers import AnxcorDatabase

class IRISWrapper(AnxcorDatabase):

    def __init__(self):
        super().__init__()
        self.client    =  Client("IRIS")
        self.station_list = ['S01','S02']#,'S03','S04','S05']
        self.pre_filter = (0.003, 0.005, 40.0, 45.0)

    def get_waveforms(self,starttime=0,endtime=0,station=0,network=0, **kwargs):
        traces = []
        stream =  self.client.get_waveforms(network, station, "*", "H*", starttime,endtime,attach_response=True)
        stream.remove_response(output='DISP', pre_filt=self.pre_filter)
        for trace in stream:
            data = trace.data[:-1]
            header = {'delta':   trace.stats.delta,
                      'station': trace.stats.station,
                      'starttime':trace.stats.starttime,
                      'channel': trace.stats.channel,
                      'network': trace.stats.network}
            traces.append(Trace(data,header=header))

        return Stream(traces=traces)

    def get_stations(self):
        station_list = []
        for station in self.station_list:
            station_list.append('YB.{}'.format(station))
        return station_list

class XArrayOneBit(XArrayProcessor):
    """
    one bit normalization
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def execute(self, xarray, *args, **kwargs):
        return np.sign(xarray)

    def _add_operation_string(self):
        return 'one-bit-norm'

    def _get_process(self):
        return 'one-bit-norm'

converter = XArrayConverter()
class TestNotebookUsageExample(unittest.TestCase):
    def test_expected_pair_amount(self):
        anxcor_main = Anxcor(60 * 10.0)
        anxcor_main.add_dataset(IRISWrapper(), 'IMUSH_ST_HELLENS_DATA')
        resample_kwargs  = dict(taper=0.05, target_rate=20.0)
        correlate_kwargs = dict(taper=0.1, max_tau_shift=50.0)
        anxcor_main.add_process(XArrayResample(**resample_kwargs))
        anxcor_main.set_task_kwargs('crosscorrelate', correlate_kwargs)
        anxcor_main.add_process(XArrayRemoveMeanTrend())
        anxcor_main.add_process(XArrayOneBit())
        starttime = UTCDateTime("2005-6-22 12:00:00").timestamp
        starttimes = []
        for window_number in range(0, 4):
            starttimes.append(starttime + 60 * 10.0 * window_number)
        xarray_dataset = anxcor_main.process(starttimes)
        assert len(list(xarray_dataset.coords['rec'].values))==2


