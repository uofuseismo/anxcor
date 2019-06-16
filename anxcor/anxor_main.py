from anxcor_containers import _CombinedDataLoaderTest
from xarray_routines import XArrayConverter, XResample, XArrayXCorrelate, \
    XArrayStack, XArrayCombine, XArrayIO
import itertools
import os_utils as os_utils
from obspy.core import UTCDateTime


def gather(bool_list):
    return True

class Anxcor:
    time_format = '%d-%m-%Y %H:%M:%S'

    def __init__(self, window_length, overlap, target_downsample_rate=10.0):
        """

        Parameters
        ----------
        window_length: float
            length of window to correlate
        overlap: float
            overlap % of windows. must be between 0 - 1, with 1 == 100%.
        """
        self._overlap       = overlap
        self._window_length = window_length
        self._tasks  = {
            'data': _CombinedDataLoaderTest(window_length),
            'xconvert' : XArrayConverter(),
            'resample' : XResample(target_downsample_rate),
            'process'  : [],
            'correlate': XArrayXCorrelate(),
            'stack'    : XArrayStack(),
            'write_stack':   XArrayIO(),
            'combine'    :   XArrayCombine(),
            'write_combine': XArrayIO()
        }

    def write_result_to_file(self,folder, type='resample', order=None,**kwargs):
        """

        Parameters
        ----------
        folder
        type:
            one of:
            'data'
            'resample'
            'xconvert'
            'process'
            'correlate'
            'write_stack'
            'combine'
            'write_combine'
            an integer representing the order of extra station ops

        order: int
            the order of the processing stack to assign a write function

        Returns
        -------

        """
        if isinstance(type,str):
            if type in self._tasks.keys():
                if order is None:
                    self._tasks[type].write_to_file(folder,**kwargs)
                elif isinstance(order,int) and order < len(self._tasks[type]):
                    self._tasks[type][order].write_to_file(folder,**kwargs)
                else:
                    raise Exception('Write assign call error. please refer to documentation')
            else:
                raise Exception('Not a valid key')



    def add_dataset(self, dataset, name, trace_prep=None, **kwargs):
        self._tasks['data'].add_dataset(dataset, name, trace_prep=trace_prep, **kwargs)

    def add_process(self, process):
        self._tasks['process'].append(process)


    def _station_window_operations(self, channels,dask_client=None):
        xarray      = self._tasks['xconvert'](channels, dask_client=dask_client )
        downsampled = self._tasks['resample'](xarray, dask_client=dask_client )
        for process in self._tasks['process']:
            downsampled = process(downsampled, dask_client=dask_client )
        return downsampled


    def process(self,starttime, endtime, dask_client=None,**kwargs):

        starttimes    = self._get_starttimes(starttime, endtime)
        station_pairs = self._get_station_pairs()
        futures = []
        for pair in station_pairs:
            source        = pair[0]
            receiver      = pair[1]
            correlation_stack   = []
            for starttime in starttimes:
                if pair==('FG.21','FG.22'):
                    print('trying again')
                source_channels   = self._tasks['data'](starttime, source,   dask_client=dask_client )
                receiver_channels = self._tasks['data'](starttime, receiver, dask_client=dask_client )

                source_ch_ops     = self._station_window_operations(source_channels, dask_client=dask_client)
                receiver_ch_ops   = self._station_window_operations(receiver_channels, dask_client=dask_client)

                correlation       = self._tasks['correlate'](source_ch_ops, receiver_ch_ops, dask_client=dask_client)

                correlation_stack.append(correlation)

            correlation_stack = self._stack_correlations(correlation_stack,  dask_client=dask_client)
            extension = self._make_write_stack_path('correlation_stack', pair)
            self._tasks['write_stack'](correlation_stack,extension,'xstack',dask_client=dask_client)
            futures.append(correlation_stack)

        combined_crosscorrelations= self._concatenate_stacks(futures, dask_client=dask_client)

        self._tasks['write_combine'](combined_crosscorrelations,'combined_correlations','correlations',dask_client=dask_client)
        return combined_crosscorrelations

    def _concatenate_stacks(self, futures, dask_client=None):
        if len(futures) % 2 != 0:
            # small process and remap
            first_future = futures.pop()
            second_future = futures.pop()

            result = self._tasks['combine'](first_future, second_future, dask_client=dask_client)
            futures.append(result)
        while len(futures) > 1:
            new_futures = []
            for i in range(0, len(futures), 2):
                future = self._tasks['combine'](futures[i], futures[i + 1], dask_client=dask_client)
                new_futures.append(future)

            futures = new_futures
        return futures[0]

    def _stack_correlations(self, correlation_stack, dask_client=None):
        if len(correlation_stack) % 2 != 0:
            # small process and remap
            first_future = correlation_stack.pop()
            second_future = correlation_stack.pop()

            result = self._tasks['stack'](first_future, second_future, dask_client=dask_client)
            correlation_stack.append(result)
        while len(correlation_stack) > 1:
            new_correlation_stack = []
            for i in range(0, len(correlation_stack), 2):
                future = self._tasks['stack'](correlation_stack[i], correlation_stack[i + 1], dask_client=dask_client)
                new_correlation_stack.append(future)

            correlation_stack = new_correlation_stack

        return correlation_stack[0]

    def _process_key(self,pair,starttime):
        time_key = self._time_key(starttime)
        pair_key = self._pair_key(pair)
        key = '{} : {}'.format(pair_key, time_key)
        return key

    def _pair_key(self,pair):
        key = 's {}>r {}'.format(pair[0],pair[1])
        return key

    def _time_key(self,starttime):
        fmttime = UTCDateTime(starttime).strftime(self.time_format)
        return fmttime

    def create_processing_arguments(self, endtime, starttime):
        """
        create a list of arguments
        Parameters
        ----------
        endtime: float
        starttime: float

        Returns
        -------
            iterable list of [ (starttimes, (source,receiver)) ]
        """
        starttimes = self._get_starttimes(starttime, endtime)
        station_pairs = self._get_station_pairs()
        process_args = self._create_process_args(starttimes, station_pairs)
        return process_args

    def _get_starttimes(self,starttime,endtime):
        starttimes = []
        delta      = self._window_length * (1 - self._overlap)
        time = starttime
        while time < endtime:
            starttimes.append(time)
            time+=delta
        return starttimes

    def _get_station_pairs(self):
        stations = self._tasks['data'].get_stations()
        station_pairs = list(itertools.combinations_with_replacement(stations, 2))
        return station_pairs

    def _create_process_args(self, starttimes, station_pairs):
        args = []
        for time in starttimes:
            for pair in station_pairs:
                args.append( ( time, pair ))

        return args

    def _make_write_stack_path(self, folder, pair):
        path = os_utils.os.path.join(*[folder, 'src:' + pair[0] , 'rec:' + pair[1]])
        return path


