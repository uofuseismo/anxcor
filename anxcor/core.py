

from  anxcor.containers import DataLoader, XArrayCombine, XArrayStack, AnxcorDatabase
from  anxcor.xarray_routines import XArrayConverter, XResample, XArrayXCorrelate
from anxcor.abstractions import XArrayProcessor
from typing import List, Callable
from xarray import Dataset
import numpy as np
import itertools
from obspy.core import UTCDateTime, Stream, Trace

class Anxcor:



    def __init__(self, window_length: float,**kwargs):
        """

        Parameters
        ----------
        window_length: float
            the window length in seconds
        """
        self._data      = _AnxcorData(window_length=window_length, **kwargs)
        self._processor = _AnxcorProcessor(self._data)
        self._converter = _AnxcorConverter(self._data)


    def add_dataset(self, database: AnxcorDatabase, name: str, **kwargs) -> None:
        """
        Adds a new dataset implementing the AnxorDatabase Interface to the crosscorrelation.
        If desired, you may provide an additional callable function used to remove the instrument
        response at every step.

        Parameters
        ----------
        database : AnxcorDatabase
            a database to add which implements the AnxorDatabase interface
        name : str
            a name describing the type of the resultant data
        trace_prep : Callable[[Trace],Trace], Optional
            an optional callable function to remove the instrument response.


        """
        self._data.add_dataset(database, name, **kwargs)


    def add_process(self, process: XArrayProcessor) -> None:
        """
        Add an XArrayProcessor to the processing steps. Functions are performed in the order they are added

        Parameters
        ----------
        process : XArrayProcessor
            an XArrayProcessor object

        """
        self._data.add_process(process)

    def set_parameters(self, process_key: str, key_dict: dict, *args) -> None:
        """
        Set parameters on a processing step not defined prior to adding them

        Parameters
        ----------
        process_key : str
            the key of the process to add the keyword-arguments to. If it is a process added through the
        'add_process()' method, the key should be 'process'

        key_dict : dict
            a dictionary of keyword arguments to pass on to the XArray Processor object

        """
        self._data.set_parameters(process_key,key_dict,*args)

    def get_task(self,key: str) -> XArrayProcessor:
        """
        Returns an XArrayProcessor task based on its key

        Parameters
        ----------
        key : str
            the key to the processor object

        Returns
        -------
        XArrayProcessor, List[XArrayProcessor]
            either the processor or the list of processes defined by the task key

        """
        return self._data.get_task(key)

    def get_starttimes(self,starttime: float, endtime: float, overlap: float) -> List[float]:
        """
        Get a list of starttimes based on a given starttime, endtime, and percent overlap of windows

        Parameters
        ----------
        starttime : float
            the starttime (as a UTCDateTime timestamp)
        endtime : float
            the endtime (as a UTCDateTime timestamp)
        overlap : float
            percent overlap of windows as a decimal (100% = 1.0). Must be between 0 - 1.

        Returns
        -------
        List[float]
            a list of UTCDateTime timestamps representing window starttimes

        """
        return self._converter.get_starttimes(starttime, endtime, overlap)

    def save_at_step(self,folder: str,type: str,order: int = None,**kwargs) -> None:
        """
        Save result of method to file at a given step type

        Parameters
        ----------
        folder : str
            the folder to save the step result to.
        type : str
            one of:
            'data'
            'resample'
            'xconvert'
            'process'
            'correlate'
            'write_stack'
            'combine'
            'write_combine'
        representing the process key at which to save
        order: int, Optional
            the order of the processing stack to write at. Only necessary if type=='process'

        """
        self._data.save_at_step(folder,type=type,order=order,**kwargs)

    def load_at_step(self,folder: str,type: str,order: int=None,**kwargs)-> None:
        """
        Start processing at the given step, skipping all prior steps

        Parameters
        ----------
        folder : str
            the folder to save the step result to.
        type : str
            one of:
            'data'
            'resample'
            'xconvert'
            'process'
            'correlate'
            'write_stack'
            'combine'
            'write_combine'
        representing the process key at which to save
        order: int, Optional
            the order of the processing stack to write at. Only necessary if type=='process'

        """
        self._data.load_at_step(folder,type=type,order=order,**kwargs)

    def process(self, starttimes: List[float], dask_client=None) -> Dataset:
        """
        Perform the crosscorrelation routines. By default this processes, stacks, and combines
        all crosscorrelations during the given starttimes. See documentation for finer control

        Parameters
        ----------
        starttimes : List[float]
            a list of UTCDateTime timestamps representing window starttimes
        dask_client : Optional
            an optional dask_client instance for parallel processing using dask

        Returns
        -------
        Dataset, Future,

            if single threaded, will return a single XArray Dataset object containing the
        stacked crosscorrelations. If a dask client is provided it will return a future instance.
        See Dask Documentation for further details
        """
        return self._processor.process(starttimes, dask_client=dask_client)

    def xarray_to_obspy(self, xdataset: Dataset):
        """
        convert the output of a anxcor correlation into an obspy stream

        Parameters
        ----------
        xdataset : Dataset
            an xarray Dataset object produced by AnXcor

        Returns
        -------
        Stream
            obspy trace stream of cross correlations

        """
        return self._converter.xarray_to_obspy(xdataset)



class _AnxcorProcessor:

    time_format = '%d-%m-%Y %H:%M:%S'
    def __init__(self,data):
        self.data =data

    def _station_window_operations(self, channels, dask_client=None, starttime=None, station=None):
        xarray      = self.data.get_task('xconvert')(channels, starttime=starttime, station=station, dask_client=dask_client )
        downsampled = self.data.get_task('resample')(xarray, starttime=starttime, station=station, dask_client=dask_client )
        tasks = [downsampled]
        process_list = self.data.get_task('process')
        for process in process_list:
            task   = tasks.pop()
            result = process(task,starttime=starttime, station=station, dask_client=dask_client )
            tasks.append(result)
        return tasks[0]


    def process(self,starttimes, dask_client=None,**kwargs):

        station_pairs = self.data.get_station_combinations()
        futures = []
        for pair in station_pairs:

            correlation_list  = self._iterate_starttimes(pair,  starttimes,dask_client=dask_client)
            correlation_stack = self._reduce(correlation_list,
                                             station=str(pair),
                                             reducing_func=self.data.get_task('stack'),
                                             dask_client=dask_client)

            futures.append(correlation_stack)

        combined_crosscorrelations = self._reduce(futures,
                                                  station='combined',
                                                  reducing_func=self.data.get_task('combine'),
                                                  dask_client=dask_client)

        return combined_crosscorrelations


    def _iterate_starttimes(self, pair, starttimes, dask_client=None):
        source   = pair[0]
        receiver = pair[1]
        correlation_stack = []
        for starttime in starttimes:
            source_channels = self.data.get_task('data')(
                                                  starttime=starttime,
                                                  station=source,
                                                  dask_client=dask_client)
            source_ch_ops = self._station_window_operations(source_channels,
                                                            starttime=starttime,
                                                            station=source,
                                                            dask_client=dask_client)
            if source==receiver:
                receiver_ch_ops = source_ch_ops
            else:
                receiver_channels = self.data.get_task('data')(
                                                      starttime=starttime,
                                                      station=receiver,
                                                      dask_client=dask_client)
                receiver_ch_ops   = self._station_window_operations(receiver_channels,
                                                                starttime=starttime,
                                                                station=receiver,
                                                                dask_client=dask_client)

            correlation = self.data.get_task('correlate')(source_ch_ops,
                                                   receiver_ch_ops,
                                                   station='src:{}rec:{}'.format(source,receiver),
                                                   starttime=starttime,
                                                   dask_client=dask_client)

            correlation_stack.append(correlation)

        return correlation_stack

    def _reduce(self,
                future_stack,
                station = None,
                reducing_func = None,
                dask_client=None):

        tree_depth = 1
        while len(future_stack) > 1:
            new_future_list = []

            while len(future_stack) > 1:
                branch_index = len(future_stack)
                first  = future_stack.pop()
                second = future_stack.pop()
                result = reducing_func(first, second,
                                       station=station,
                                       starttime=reducing_func.starttime_parser(tree_depth, branch_index),
                                       dask_client=dask_client)
                new_future_list.append(result)

            if len(future_stack) == 1:
                branch_index = len(future_stack)
                first  = future_stack.pop()
                second = new_future_list.pop()
                result = reducing_func(first, second,
                                       station=station,
                                       starttime=reducing_func.starttime_parser(tree_depth, branch_index),
                                       dask_client=dask_client)
                new_future_list.append(result)
            tree_depth +=1
            future_stack=new_future_list
        return future_stack[0]



class _AnxcorData:
    TASKS = ['data', 'xconvert', 'resample', 'process', 'correlate', 'stack', 'combine']
    def __init__(self,window_length: float =3600.0, include_pairs=None,exclude_pairs=None, **kwargs):
        self._window_length = window_length
        self._include_pairs=include_pairs
        self._exclude_pairs=exclude_pairs
        self._tasks = {
            'data': DataLoader(window_length),
            'xconvert': XArrayConverter(),
            'resample': XResample(),
            'process': [],
            'correlate': XArrayXCorrelate(),
            'combine': XArrayCombine(),
            'stack': XArrayStack()
            }

    def get_window_length(self):
        return self._window_length

    def get_station_combinations(self):
        stations = self._tasks['data'].get_stations()
        station_pairs = list(itertools.combinations_with_replacement(stations, 2))
        return station_pairs

    def save_at_step(self, folder, type='resample', order=None, **kwargs):
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
        if isinstance(type, str):
            if type in self._tasks.keys():
                if order is None:
                    self._tasks[type].set_folder(folder, 'write', **kwargs)
                elif isinstance(order, int) and order < len(self._tasks[type]):
                    self._tasks[type][order].set_folder(folder, 'write', **kwargs)
                else:
                    raise Exception('Write assign call error. please refer to documentation')
            else:
                raise Exception('Not a valid key')

    def load_at_step(self, folder, type='resample', order=None, **kwargs):
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
        if isinstance(type, str):
            if type in self._tasks.keys():
                if order is None:
                    self._tasks[type].set_folder(folder, 'read', **kwargs)
                    self._disable_tasks(type)
                elif isinstance(order, int) and order < len(self._tasks[type]):
                    self._tasks[type][order].set_folder(folder, 'read', **kwargs)
                    self._disable_process(order)
                    self._disable_tasks('process')
                else:
                    raise Exception('Write assign call error. please refer to documentation')
            else:
                raise Exception('Not a valid key')


    def _disable_process(self, position):
        for index in range(0, position + 1):
            self._tasks['process'][index].disable()


    def _disable_tasks(self, key):
        index = self.TASKS.index(key)
        for disable in range(0, index + 1):
            task = self.TASKS[disable]
            if task is not 'process':
                self._tasks[task].disable()
            else:
                self._disable_process(len(self._tasks['process']) - 1)


    def add_dataset(self, dataset, name, trace_prep=None, **kwargs):
        self._tasks['data'].add_dataset(dataset, name, trace_prep=trace_prep, **kwargs)


    def add_process(self, process):
        self._tasks['process'].append(process)


    def set_parameters(self, process, key_dict, *args):
        if process == 'process':
            self._tasks[process][args[0]].set_param(key_dict)
        else:
            self._tasks[process].set_param(key_dict)


    def get_task(self,key):
        if key in self._tasks.keys():
            return self._tasks[key]
        else:
            raise KeyError('key does not exist in tasks')


class _AnxcorConverter:


    def __init__(self,data):
        self.data = data

    def get_starttimes(self, starttime, endtime, overlap):
        starttimes = []
        delta      = self.data.get_window_length() * (1 - overlap)
        time = starttime
        while time < endtime:
            starttimes.append(time)
            time+=delta
        return starttimes

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