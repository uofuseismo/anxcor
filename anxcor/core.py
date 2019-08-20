from  anxcor.containers import DataLoader, XArrayCombine, XArrayStack
from  anxcor.xarray_routines import XArrayConverter, XArrayResample, XArrayXCorrelate
import xarray as xr
import numpy as np
import itertools
from obspy.core import UTCDateTime, Stream, Trace
import json
import anxcor.utils as utils

class _AnxcorProcessor:

    time_format = '%d-%m-%Y %H:%M:%S'
    CORR_FORMAT = 'src:{} rec:{}'
    def __init__(self):
        pass

    def _station_window_operations(self, channels, dask_client=None, starttime=None, station=None):
        xarray      = self._get_task('xconvert')(channels, starttime=starttime, station=station, dask_client=dask_client )
        downsampled = self._get_task('resample')(xarray, starttime=starttime, station=station, dask_client=dask_client )
        tasks = [downsampled]
        process_list = self._get_process_order()
        for process_key in process_list:
            process = self._get_process(process_key)
            task   = tasks.pop()
            result = process(task,starttime=starttime, station=station, dask_client=dask_client )
            tasks.append(result)

        return tasks[0]


    def process(self,starttimes, dask_client=None,**kwargs):

        station_pairs = self.get_station_combinations()
        futures = []
        for pair in station_pairs:

            correlation_list  = self._iterate_starttimes(pair,  starttimes,dask_client=dask_client)
            correlation_stack = self._reduce(correlation_list,
                                             station=str(pair),
                                             reducing_func=self._get_task('stack'),
                                             dask_client=dask_client)

            futures.append(correlation_stack)

        combined_crosscorrelations = self._reduce(futures,
                                                  station='combine',
                                                  reducing_func=self._get_task('combine'),
                                                  dask_client=dask_client)

        return combined_crosscorrelations


    def _iterate_starttimes(self, pair, starttimes, dask_client=None):
        source   = pair[0]
        receiver = pair[1]
        correlation_stack = []
        for starttime in starttimes:
            source_channels = self._get_task('data')(
                                                  starttime=starttime,
                                                  station=source,
                                                  dask_client=dask_client)
            source_ch_ops = self._station_window_operations(source_channels,
                                                            starttime=starttime,
                                                            station=source,
                                                            dask_client=dask_client)
            if source==receiver:
                receiver_ch_ops   = source_ch_ops
            else:
                receiver_channels = self._get_task('data')(
                                                      starttime=starttime,
                                                      station=receiver,
                                                      dask_client=dask_client)
                receiver_ch_ops   = self._station_window_operations(receiver_channels,
                                                                starttime=starttime,
                                                                station=receiver,
                                                                dask_client=dask_client)

            correlation = self._get_task('crosscorrelate')(source_ch_ops,
                                                   receiver_ch_ops,
                                                   station=self.CORR_FORMAT.format(source,receiver),
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

        result = reducing_func(future_stack[0], None,
                      station=station,
                      starttime=reducing_func.starttime_parser('Final', 0),
                      dask_client=dask_client)
        return result



class _AnxcorData:
    TASKS = ['data', 'xconvert', 'resample', 'process', 'crosscorrelate', 'stack', 'combine']
    def __init__(self):
        self._window_length=60*60.0
        self._tasks = {
            'data': DataLoader(),
            'xconvert': XArrayConverter(),
            'resample': XArrayResample(),
            'process' : {},
            'crosscorrelate': XArrayXCorrelate(),
            'combine': XArrayCombine(),
            'stack': XArrayStack()
            }
        self._process_order = []

    def _get_task_keys(self):
        return self._tasks.keys()

    def _get_process_keys(self):
        return self._tasks['process'].keys()

    def set_window_length(self,window_length: float):
        self._window_length=window_length
        self._tasks['data'].set_kwargs(dict(window_length=window_length))

    def _get_process_order(self):
        return self._process_order.copy()

    def _get_process(self,process_key):
        return self._tasks['process'][process_key]

    def get_window_length(self):
        return self._tasks['data'].get_kwargs()['window_length']

    def get_station_combinations(self):
        stations = self._tasks['data'].get_stations()
        station_pairs = list(itertools.combinations_with_replacement(stations, 2))
        return station_pairs

    def has_data(self):
        return self._tasks['data'].has_data()

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
            print('{}: is not a valid task. must be one of:'.format(task))
            for acceptable_task in self._tasks.keys():
                print(acceptable_task)
            print('ignoring')

    def set_process_kwargs(self,process: str, kwargs: dict):
        if process in self._tasks['process'].keys():
            self._tasks['process'][process].set_kwargs(kwargs)
        else:
            print('{}: is not a valid process. ignoring'.format(process))

    def _get_task(self,key):
        if key in self._tasks.keys():
            return self._tasks[key]
        else:
            raise KeyError('key does not exist in tasks')


class _AnxcorConfig:

    def __init__(self):
        pass

    def save_config(self, file):
        config = {}
        for task in self.TASKS:
            if task!='process':
                config[task] = self._get_task(task).get_kwargs()
            else:
                config[task] = {
                    'processing order' : self._process_order,
                }
                for key, process in self._get_task(task).items():
                    config[task][key]=process.get_kwargs()

        folder = utils.get_folderpath(file)
        if not utils.folder_exists(folder):
            print('given folder to save file does not exist. ignoring save call')
            return None
        with open(file, 'w') as configfile:
            json.dump(config,configfile, sort_keys=True, indent=4)


    def load_config(self, file):
        if not utils.file_exists(file):
            print('given .ini file to load file does not exist. Inoring load call')
            return None

        with open(file, 'r') as p_file:
            config = json.load(p_file)
        for task in self.TASKS:
            if task!='process':
                self._get_task(task).set_kwargs(config[task])
            else:
                self._process_order = config[task]['processing order']
                for key, process_kwargs in config[task].items():
                    if key!='processing order':
                        if key not in self._get_process_keys():
                            print('{} does not exist inside current Anxcor Object\n'.format(key)+\
                                  'Skipping for now..')
                        else:
                            self._get_task(task)[key].set_kwargs(process_kwargs)

class _AnxcorConverter:


    def __init__(self):
        pass

    def get_starttimes(self, starttime, endtime, overlap):
        starttimes = []
        delta      = self.get_window_length() * (1 - overlap)
        time = starttime
        while time < endtime:
            starttimes.append(time)
            time+=delta
        return starttimes

    def xarray_3D_to_2D(self,xdataset: xr.Dataset):
        # get vars
        new_ds = xdataset.assign_coords(component='{}{}{}'.format(xdataset.src_chan,':',xdataset.rec_chan))
        new_ds = new_ds.drop_dims(['src_chan','rec_chan'])
        return new_ds

    def xarray_2D_to_3D(self,xdataset: xr.Dataset):
        new_ds = xdataset.assign_coords(src_chan=xdataset.component.split(':')[0])
        new_ds = new_ds.assign_coords(rec_chan=xdataset.component.split(':')[1])
        new_ds = new_ds.drop_dims(['component'])
        return new_ds

    def xarray_to_obspy(self,xdataset: xr.Dataset):
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

    def align_station_pairs(self,xdataset: xr.Dataset):
        attrs = xdataset.attrs.copy()
        del attrs['delta']
        del attrs['operations']
        print(attrs.keys())
        for name in xdataset.data_vars:
            xarray = xdataset[name]




class Anxcor(_AnxcorData, _AnxcorProcessor, _AnxcorConverter, _AnxcorConfig):



    def __init__(self,*args,**kwargs):
        super().__init__()