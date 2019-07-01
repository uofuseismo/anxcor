from  anxcor.containers import DataLoader, XArrayCombine, XArrayStack, AnxcorDatabase
from  anxcor.xarray_routines import XArrayConverter, XResample, XArrayXCorrelate
from anxcor.abstractions import XArrayProcessor
from typing import List
from xarray import Dataset
import numpy as np
import itertools
from obspy.core import UTCDateTime, Stream, Trace

import json

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

    def set_task_kwargs(self, task: str, kwargs: dict):
        """
        sets keyword-arguments into the given task.

        Parameters
        ----------
        task : str
            the task key
        kwargs : dict
            the keyword-arguments to set

        """
        self._data.set_task_kwargs(task,kwargs)

    def set_process_kwargs(self, process: str, kwargs: dict):
        """
        sets keyword-arguments into the given process

        Parameters
        ----------
        process : str
            the process key
        kwargs : dict
            the keyword-arguments to set

        """
        self._data.set_process_kwargs(process,kwargs)

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

    def save_at_task(self, folder: str,task,**kwargs) -> None:
        """
        Save result of method to file at a given task step

        Parameters
        ----------
        folder : str
            the folder to save the step result to.
        task : str
            one of:
            'data'
            'resample'
            'xconvert'
            'correlate'
            'stack'
            'combine'
        representing the process key at which to save.

        Note
        ----
        see save_at_process() for saving after a specific processing step

        """
        self._data.save_at_task(folder,task)

    def load_at_task(self, folder: str,task,**kwargs)-> None:
        """
        load result of method from file at a given task step.
        will disable all routines behind it in the compute graph

        Parameters
        ----------
        folder : str
            the folder to save the step result to.
        task : str
            one of:
            'data'
            'resample'
            'xconvert'
            'correlate'
            'stack'
            'combine'
        representing the process key at which to save.

        Note
        ----
        see load_at_process() for saving after a specific processing step

        """
        self._data.load_at_task(folder, task)

    def save_at_process(self, folder: str,process : str,**kwargs) -> None:
        """
        save result of method to file at a given task step

        Parameters
        ----------
        folder : str
            the folder to save the step result to.
        process : str
            a string returned by the get_name()/get_process() method of an XArrayProcessor
            object assigned to an Anxcor object

        Note
        ----
        the object must have already been added via the add_process() routine

        """
        self._data.save_at_process(folder,process)

    def load_at_process(self, folder: str,process: str,**kwargs)-> None:
        """
        load result of method from file at a given task step
        will disable all routines behind it in the compute graph

        Parameters
        ----------
        folder : str
            the folder to save the step result to.
        process : str
            a string returned by the get_name()/get_process() method of an XArrayProcessor
            object assigned to an Anxcor object

        Note
        ----
        the object must have already been added via the add_process() routine

        """
        self._data.load_at_process(folder, process)

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

    def save_config(self,file):
        """
        saves the parameters for each processing step as a .ini file
        Parameters
        ----------
        file : str
            the config file to save
        """
        self._data.save_config(file)

    def load_config(self,file):
        """
        loads a previously built .ini file, assigning the results to the current process stack
        will present a warning if config file parameters are not present
        Parameters
        ----------
        file : str
            the config file to load


        """
        self._data.load_config(file)



class _AnxcorProcessor:

    time_format = '%d-%m-%Y %H:%M:%S'
    def __init__(self,data):
        self.data =data

    def _station_window_operations(self, channels, dask_client=None, starttime=None, station=None):
        xarray      = self.data.get_task('xconvert')(channels, starttime=starttime, station=station, dask_client=dask_client )
        downsampled = self.data.get_task('resample')(xarray, starttime=starttime, station=station, dask_client=dask_client )
        tasks = [downsampled]
        process_list = self.data.get_process_order()
        for process_key in process_list:
            process = self.data.get_process(process_key)
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
                                                  station='combine',
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

            correlation = self.data.get_task('crosscorrelate')(source_ch_ops,
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
    TASKS = ['data', 'xconvert', 'resample', 'process', 'crosscorrelate', 'stack', 'combine']
    def __init__(self,window_length: float =3600.0):
        self._window_length=window_length
        self._tasks = {
            'data': DataLoader(window_length),
            'xconvert': XArrayConverter(),
            'resample': XResample(),
            'process' : {},
            'crosscorrelate': XArrayXCorrelate(),
            'combine': XArrayCombine(),
            'stack': XArrayStack()
            }
        self._process_order = []

    def get_process_order(self):
        return self._process_order.copy()

    def get_process(self,process_key):
        return self._tasks['process'][process_key]

    def get_window_length(self):
        return self._window_length

    def get_station_combinations(self):
        stations = self._tasks['data'].get_stations()
        station_pairs = list(itertools.combinations_with_replacement(stations, 2))
        return station_pairs

    def save_at_task(self, folder, task : str='resample'):
        if task in self._tasks.keys() and task!='process':
             self._tasks[task].set_io_task(folder, 'save')
        else:
            print('{} is Not a valid task'.format(task))

    def save_at_process(self, folder, process : str='whiten'):
        if process in self._process_order:
             self._tasks['process'][process].set_io_task(folder, 'save')
        else:
            print('{} is Not a valid process'.format(process))


    def load_at_process(self, folder, process='resample'):
        if process in self._process_order:
            self._tasks['process'][process].set_io_task(folder, 'load')
            self._disable_tasks('process')
            self._disable_process(process)
        else:
            print('{} is Not a valid process'.format(process))

    def load_at_task(self, folder, task='resample'):
        if task in self._tasks.keys() and task!='process':
             self._tasks[task].set_io_task(folder, 'load')
             self._disable_tasks(task)
        else:
            print('{} is Not a valid task'.format(task))


    def _disable_process(self, process):
        index = self._process_order.index(process)
        for disable in range(0, index + 1):
            pr = self._process_order[disable]
            self._tasks['process'][pr].disable()


    def _disable_tasks(self, key):
        index = self.TASKS.index(key)
        for disable in range(0, index + 1):
            task = self.TASKS[disable]
            if task is not 'process':
                self._tasks[task].disable()
            else:
                if self._process_order:
                    self._disable_process(self._process_order[-1])


    def add_dataset(self, dataset, name, trace_prep=None, **kwargs):
        self._tasks['data'].add_dataset(dataset, name, trace_prep=trace_prep, **kwargs)


    def add_process(self, process):
        key = process.get_name()
        self._tasks['process'][key]=process
        self._process_order.append(key)

    def set_task_kwargs(self,task: str,kwargs: dict):
        if task in self._tasks.keys():
            self._tasks[task].set_kwargs(kwargs)
        else:
            print('{}: is not a valid task. ignoring'.format(task))

    def set_process_kwargs(self,process: str, kwargs: dict):
        if process in self._tasks['process'].keys():
            self._tasks['process'][process].set_kwargs(kwargs)
        else:
            print('{}: is not a valid process. ignoring'.format(process))

    def get_task(self,key):
        if key in self._tasks.keys():
            return self._tasks[key]
        else:
            raise KeyError('key does not exist in tasks')

    def save_config(self, file):
        config = {}
        for task in self.TASKS:
            if task!='process':
                config[task] = self._tasks[task].get_kwargs()
            else:
                config[task] = {
                    'processing order' : self._process_order,
                }
                for key, process in self._tasks[task].items():
                    config[task][key]=process.get_kwargs()
        with open(file, 'w') as configfile:
            json.dump(config,configfile, sort_keys=True, indent=4)


    def load_config(self, file):
        with open(file, 'r') as p_file:
            config = json.load(p_file)
        for task in self.TASKS:
            if task!='process':
                self._tasks[task].set_kwargs(config[task])
            else:
                self._process_order = config[task]['processing order']
                for key, process_kwargs in config[task].items():
                    if key!='processing order':
                        if key not in self._tasks[task].keys():
                            print('{} does not exist inside current Anxcor Object\n'.format(key)+\
                                  'Skipping for now..')
                        else:
                            self._tasks[task][key].set_kwargs(process_kwargs)


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