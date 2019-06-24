from  anxcor.anxcor_containers import DataLoader, XArrayCombine, XArrayStack
from  anxcor.xarray_routines import XArrayConverter, XResample, XArrayXCorrelate
import numpy as np
import itertools
import anxcor.os_utils as os_utils
from obspy.core import UTCDateTime, Stream, Trace

class Anxcor:
    time_format = '%d-%m-%Y %H:%M:%S'
    TASKS = ['data','xconvert','resample','process','correlate','stack','combine']
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
            'data': DataLoader(window_length),
            'xconvert' : XArrayConverter(),
            'resample' : XResample(target_downsample_rate),
            'process'  : [],
            'correlate': XArrayXCorrelate(),
            'combine'  : XArrayCombine(),
            'stack'    : XArrayStack()
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
                    self._tasks[type].set_folder(folder,'write', **kwargs)
                elif isinstance(order,int) and order < len(self._tasks[type]):
                    self._tasks[type][order].set_folder(folder,'write', **kwargs)
                else:
                    raise Exception('Write assign call error. please refer to documentation')
            else:
                raise Exception('Not a valid key')


    def read_from_file_at_step(self,folder, type='resample', order=None,**kwargs):
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
                    self._tasks[type].set_folder(folder,'read', **kwargs)
                    self._disable_tasks(type)
                elif isinstance(order,int) and order < len(self._tasks[type]):
                    self._tasks[type][order].set_folder(folder,'read', **kwargs)
                    self._disable_process(order)
                    self._disable_tasks('process')
                else:
                    raise Exception('Write assign call error. please refer to documentation')
            else:
                raise Exception('Not a valid key')

    def _disable_process(self,position):
        for index in range(0,position+1):
            self._tasks['process'][index].disable()

    def _disable_tasks(self,key):
        index = self.TASKS.index(key)
        for disable in range(0,index+1):
            task = self.TASKS[disable]
            if task is not 'process':
                self._tasks[task].disable()
            else:
                self._disable_process(len(self._tasks['process'])-1)

    def add_dataset(self, dataset, name, trace_prep=None, **kwargs):
        self._tasks['data'].add_dataset(dataset, name, trace_prep=trace_prep, **kwargs)

    def add_process(self, process):
        self._tasks['process'].append(process)

    def set_parameters(self, process, key_dict, *args):
        if process=='process':
            self._tasks[process][args[0]].set_param(key_dict)
        else:
            self._tasks[process].set_param(key_dict)

    def _station_window_operations(self, channels, dask_client=None, starttime=None, station=None):
        xarray      = self._tasks['xconvert'](channels, starttime=starttime, station=station, dask_client=dask_client )
        downsampled = self._tasks['resample'](xarray, starttime=starttime, station=station, dask_client=dask_client )
        tasks = [downsampled]
        for process in self._tasks['process']:
            task        = tasks.pop()
            result = process(task,starttime=starttime, station=station, dask_client=dask_client )
            tasks.append(result)
        return tasks[0]


    def process(self,starttime, endtime, dask_client=None,autocorrelate=True,**kwargs):

        starttimes    = self._get_starttimes(starttime, endtime)
        station_pairs = self._get_station_pairs()
        futures = []
        for pair in station_pairs:

            if not pair[0]==pair[1] or autocorrelate:
                correlation_list  = self._iterate_starttimes(pair,  starttimes,dask_client=dask_client)
                correlation_stack = self._stack_correlations(correlation_list, pair,  dask_client=dask_client)
                futures.append(correlation_stack)

        combined_crosscorrelations = self._combine_stack_pairs(futures, dask_client=dask_client)

        return combined_crosscorrelations


    def _iterate_starttimes(self, pair, starttimes, dask_client=None):
        source   = pair[0]
        receiver = pair[1]
        correlation_stack = []
        for starttime in starttimes:
            source_channels = self._tasks['data'](starttime, source,
                                                  starttime=starttime,
                                                  station=source,
                                                  dask_client=dask_client)
            source_ch_ops = self._station_window_operations(source_channels,
                                                            starttime=starttime,
                                                            station=source,
                                                            dask_client=dask_client)

            receiver_channels = self._tasks['data'](starttime, receiver,
                                                      starttime=starttime,
                                                      station=receiver,
                                                      dask_client=dask_client)
            receiver_ch_ops   = self._station_window_operations(receiver_channels,
                                                                starttime=starttime,
                                                                station=receiver,
                                                                dask_client=dask_client)

            correlation = self._tasks['correlate'](source_ch_ops,
                                                   receiver_ch_ops,
                                                   station='src:{}rec:{}'.format(source,receiver),
                                                   starttime=starttime,
                                                   dask_client=dask_client)

            correlation_stack.append(correlation)

        return correlation_stack

    def _combine_stack_pairs(self, futures, dask_client=None):
        tree_depth = 1
        while len(futures) > 1:
            new_list = []

            while len(futures) > 1:
                branch_index = len(futures)
                first  = futures.pop()
                second = futures.pop()
                starttime = 'depth:{}branch:{}'.format(tree_depth,branch_index)
                result=self._tasks['combine'](first,second,station='combined',
                                                           starttime=starttime,
                                                           dask_client=dask_client)
                new_list.append(result)

            if len(futures) == 1:
                branch_index = len(futures)
                first = futures.pop()
                second = new_list.pop()
                starttime = 'depth:{}branch:{}'.format(tree_depth, branch_index)
                result = self._tasks['combine'](first, second,
                                                station='combined',
                                                starttime=starttime,
                                                dask_client=dask_client)
                new_list.append(result)
            tree_depth+=1
            futures = new_list
        return futures[0]

    def _stack_correlations(self, correlation_stack,station_pairs, dask_client=None):
        tree_depth = 1
        while len(correlation_stack) > 1:
            new_list = []

            while len(correlation_stack) > 1:
                branch_index = len(correlation_stack)
                first  = correlation_stack.pop()
                second = correlation_stack.pop()
                starttime = 'depth:{}branch:{}'.format(tree_depth,branch_index)
                result = self._tasks['stack'](first, second, station=str(station_pairs),
                                                             starttime=starttime,
                                                             dask_client=dask_client)
                new_list.append(result)

            if len(correlation_stack) == 1:
                branch_index = len(correlation_stack)
                first  = correlation_stack.pop()
                second = new_list.pop()
                starttime =  'depth:{}branch:{}'.format(tree_depth,branch_index)
                result = self._tasks['stack'](first, second, station=str(station_pairs),
                                                             starttime=starttime,
                                                             dask_client=dask_client)
                new_list.append(result)
            tree_depth+=1
            correlation_stack=new_list
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

    def xarray_to_obspy(self,xdataset):
        attrs = xdataset.attrs
        traces = []
        starttime = list(xdataset.coords['time'].values)[0]
        starttime = self._extract_timestamp(starttime)
        for name in xdataset.data_vars:
            xarray = xdataset[name]
            for pair in list(xdataset.coords['pair'].values):
                for src_chan in list(xdataset.coords['src_chan'].values):
                    for rec_chan in list(xdataset.coords['rec_chan'].values):

                        record = xarray.loc[dict(pair=pair,src_chan=src_chan,rec_chan=rec_chan)]

                        station_1, station_2, network_1, network_2 = self._extract_station_network_info(pair)
                        header_dict = {
                        'delta'    : record.attrs['delta'],
                        'npts'     : record.data.shape[-1],
                        'starttime': starttime,
                        'station'  : '{}.{}'.format(station_1,station_2),
                        'channel'  : '{}.{}'.format(src_chan,rec_chan),
                        'network'  : '{}.{}'.format(network_1,network_2)
                        }
                        trace = Trace(data=record.data,header=header_dict)
                        traces.append(trace)

        return Stream(traces=traces)

    def _extract_timestamp(self,starttimestamp):
        starttimestamp = starttimestamp.astype(np.float64)/1e9
        return UTCDateTime(starttimestamp)

    def _extract_station_network_info(self,pair):
        stations = [0, 0]
        pairs = pair.split('rec:')
        stations[0] = pairs[0].split(':')[1]
        stations[1] = pairs[1]

        station_1 = stations[0].split('.')[1]
        station_2 = stations[1].split('.')[1]
        network_1 = stations[0].split('.')[0]
        network_2 = stations[1].split('.')[0]
        return station_1, station_2, network_1, network_2

    def align_station_pairs(self,xdataset):
        attrs = xdataset.attrs.copy()
        del attrs['delta']
        del attrs['operations']
        print(attrs.keys())
        for name in xdataset.data_vars:
            xarray = xdataset[name]




