from  anxcor.containers import DataLoader, XArrayCombine, XArrayStack
from  anxcor.xarray_routines import XArrayConverter, XArrayResample, XArrayXCorrelate
from anxcor.abstractions import NullTask, NullDualTask
import xarray as xr
import numpy as np
import itertools
from obspy.core import UTCDateTime, Stream, Trace
from obspy.geodetics.base import gps2dist_azimuth
import json
import pandas as pd
import anxcor.utils as utils
import copy

class _AnxcorProcessor:

    time_format = '%d-%m-%Y_T%H:%M:%S'
    CORR_FORMAT = 'src:{} rec:{}'
    def __init__(self,**kwargs):
        pass

    def _station_window_operations(self, channels, dask_client=None, starttime=None, station=None):
        xarray       = self._get_task('xconvert')(channels, starttime=starttime, station=station, dask_client=dask_client )
        tasks        = [xarray]
        process_list = self._get_process_order()
        for process_key in process_list:
            process = self._get_process(process_key)
            task   = tasks.pop()
            result = process(task,starttime=starttime, station=station, dask_client=dask_client )
            tasks.append(result)

        return tasks[0]


    def process(self,starttimes, dask_client=None,stack_immediately=False,**kwargs):

        station_pairs = self.get_station_combinations()
        if station_pairs.empty:
            print('no possible station pairs detected. Exiting')
            return None
        else:
            print('correlating {} station-pairs'.format(len(station_pairs.index)))
        futures = []
        for starttime in starttimes:
            print('processing window {}'.format(UTCDateTime(starttime)))
            correlation_dataset  = self._iterate_over_pairs(starttime, station_pairs, dask_client=dask_client)
            futures.append(correlation_dataset)
            if stack_immediately:
                stacked_result = self._prepare_results(dask_client, futures)
                if dask_client is not None and \
                    stacked_result is not None \
                    and not isinstance(stacked_result,xr.Dataset):
                    stacked_result = stacked_result.result()
                futures=[stacked_result]

        combined_crosscorrelations = self._prepare_results(dask_client, futures)

        return combined_crosscorrelations

    def _prepare_results(self, dask_client, futures):
        if len(futures) >= 2:
            if dask_client is not None:
                dask_client.scatter(futures)
            combined_crosscorrelations = self._reduce(futures,
                                                      station='stack',
                                                      reducing_func=self._get_task('stack'),
                                                      dask_client=dask_client)
        else:
            combined_crosscorrelations = futures[0]
        return combined_crosscorrelations

    def _iterate_over_pairs(self, starttime, station_pairs, dask_client=None):
        correlation_stack = []
        for index, row in station_pairs.iterrows():
            source   = row['source']
            receiver = row['receiver']
            source_channels = self._get_task('data',dask_client=dask_client)(
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
                receiver_channels = self._get_task('data',dask_client=dask_client)(
                                                      starttime=starttime,
                                                      station=receiver,
                                                      dask_client=dask_client)
                receiver_ch_ops   = self._station_window_operations(receiver_channels,
                                                                starttime=starttime,
                                                                station=receiver,
                                                                dask_client=dask_client)
            corr_station = self.CORR_FORMAT.format(source,receiver)
            correlation = self._get_task('crosscorrelate')(source_ch_ops,
                                                   receiver_ch_ops,
                                                   station=corr_station,
                                                   starttime=starttime,
                                                   dask_client=dask_client)

            correlation_stack.append(correlation)

            correlation = self._get_task('post-correlate')(correlation,
                                                          station=corr_station,
                                                          starttime=starttime,
                                                          dask_client=dask_client)


        combined_crosscorrelations = self._reduce(correlation_stack,
                                                  station=UTCDateTime(starttime).strftime(self.time_format),
                                                  reducing_func=self._get_task('combine'),
                                                  dask_client=dask_client)
        combined_crosscorrelations = self._get_task('post-combine')(combined_crosscorrelations,
                                                  station='all',
                                                  starttime=starttime,
                                                  dask_client=dask_client)

        if dask_client is not None:
            combined_crosscorrelations = combined_crosscorrelations.result()

        return combined_crosscorrelations


    def _reduce(self,
                future_stack,
                station = None,
                reducing_func = None,
                dask_client=None):

        tree_depth = 1
        if dask_client is not None:
            dask_client.scatter(future_stack)
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
    TASKS = ['data', 'xconvert', 'process',
             'crosscorrelate','post-correlate',
             'combine','post-combine',
             'pre-stack','stack','post-stack']
    def __init__(self,window_length=None,**kwargs):
        self._single_station_include = []
        self._station_pair_include   = []
        self._station_pair_exclude   = []
        self._single_station_exclude = []
        self._window_length=window_length
        self._tasks = {
            'data': DataLoader(**kwargs),
            'xconvert': XArrayConverter(),
            'process' : {},
            'crosscorrelate': XArrayXCorrelate(),

            'post-correlate': NullTask('post-correlate'),
            'combine':      XArrayCombine(),
            'post-combine': NullTask('post-combine'),

            'pre-stack':    NullDualTask('pre-stack'),
            'stack':        XArrayStack(),
            'post-stack':   NullTask('post-stack')
            }
        self._process_order = []

    def _get_anxcor_config_dict(self):
        return {'source_stations': self._single_station_include,
                'receiver_stations': self._single_station_exclude,
                'window_length': self._window_length }
    def _set_anxcor_config_dict(self,dict):
        self._single_station_include=dict['source_stations']
        self._single_station_exclude= dict['receiver_stations']
        self._window_length = dict['window_length']

    def set_must_include_single_stations(self, include_stations):
        if isinstance(include_stations,str):
            self._single_station_include.append(include_stations)
        elif isinstance(include_stations,tuple) or isinstance(include_stations,list):
            self._single_station_include = self._single_station_include + include_stations
        else:
            print('include station pairs are neither string, tuple, or list. Ignoring')

    def set_must_only_include_station_pairs(self, include_stations):
        if isinstance(include_stations,tuple) or isinstance(include_stations,list):
            self._station_pair_include = self._station_pair_include + include_stations
        else:
            print('include station pairs are neither string, tuple, or list. Ignoring')

    def set_must_exclude_single_stations(self, exclude_stations):
        if isinstance(exclude_stations, str):
            self._single_station_exclude.append(exclude_stations)
        elif isinstance(exclude_stations, tuple) or isinstance(exclude_stations, list):
            self._single_station_exclude = self._single_station_exclude + exclude_stations
        else:
            print('source stations are neither tuple, or list. Ignoring')

    def set_must_exclude_station_pairs(self, exclude_stations):
        if isinstance(exclude_stations,tuple) or isinstance(exclude_stations,list):
            self._station_pair_exclude = self._station_pair_exclude + exclude_stations
        else:
            print('exclude station pairs are neither tuple, or list. Ignoring')

    def print_parameters(self):
        for task in self._tasks.keys():
            if task!='process':
                print('task: {} with parameters:\n{}'.format(task,self._tasks[task].get_kwargs()))
            else:
                for process in self._tasks[task].keys():
                    print('process: {} with parameters:\n{}'.format(process, self._tasks[task][process].get_kwargs()))

    def set_task(self,key,obj):
        if key not in self._tasks.keys():
            print('Anxcor does not use task:{}\n please use one of the following tasks'.format(key))
            for key in self._tasks.keys():
                print(key)
        else:
            self._tasks[key]=obj

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
        df = pd.DataFrame(station_pairs, columns=['source', 'receiver'])
        # ok first must include
        if self._station_pair_include:
            df = df[df['source'].isin(self._station_pair_include) & df['receiver'].isin(self._station_pair_include)]

        elif  self._single_station_include:
            df=df[df['source'].isin(self._single_station_include) | df['receiver'].isin(self._single_station_include)]

        # then exclude

        # then must exclude

        if self._single_station_exclude:
            df = df[(~df['receiver'].isin(self._single_station_exclude)) &
                    (~df['source'].isin(self._single_station_exclude))]
        return df

    def has_data(self):
        return self._tasks['data'].has_data()

    def save_at_task(self, folder, task : str='resample'):
        if task in self._tasks.keys() and task!='process':
             self._tasks[task].set_io_task(folder, 'save')
        else:
            print('{} is Not a valid task to save from'.format(task))

    def save_at_process(self, folder, process : str='whiten'):
        if process in self._process_order:
             self._tasks['process'][process].set_io_task(folder, 'save')
        else:
            print('{} is not a valid process to save from'.format(process))


    def load_at_process(self, folder, process='resample'):
        if process in self._process_order:
            self._tasks['process'][process].set_io_task(folder, 'load')
            self._disable_tasks('process')
            self._disable_process(process)
        else:
            print('{} is not a valid process to load from'.format(process))

    def load_at_task(self, folder, task='resample'):
        if task in self._tasks.keys() and task!='process':
             self._tasks[task].set_io_task(folder, 'load')
             self._disable_tasks(task)
        else:
            print('{} is not a valid task to load from'.format(task))


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
        key = process._get_process()
        while key in self._tasks['process'].keys():
            process.increment_process_number()
            key = process._get_process()
        print('adding process {}'.format(process._get_process()))
        self._tasks['process'][key]=process
        self._process_order.append(key)

    def set_task_kwargs(self,task: str,kwargs: dict):
        if task in self._tasks.keys():
            self._tasks[task].set_kwargs(kwargs)
        else:
            print('{}: is not a valid task to assign kwargs. must be one of:'.format(task))
            for acceptable_task in self._tasks.keys():
                print(acceptable_task)
            print('ignoring')

    def set_process_kwargs(self,process: str, kwargs: dict):
        if process in self._tasks['process'].keys():
            self._tasks['process'][process].set_kwargs(kwargs)
        else:
            print(self._tasks['process'])
            print('{}: is not a valid process to assign kwargs. ignoring'.format(process))

    def _get_task(self,key,dask_client=None):
        if key in self._tasks.keys():
            task = self._tasks[key]
            if dask_client is not None:
                return copy.deepcopy(task)
            else:
                return task
        else:
            raise KeyError('key {} does not exist in tasks'.format(key))


class _AnxcorConfig:

    def __init__(self,**kwargs):
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

        config['anxcor_configs'] = self._get_anxcor_config_dict()
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

        self._set_anxcor_config_dict(config['anxcor_configs'])

class _AnxcorConverter:


    def __init__(self,**kwargs):
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
        df = xdataset.attrs['df']
        traces = []
        starttime = list(xdataset.coords['time'].values)[0]
        starttime = self._extract_timestamp(starttime)
        for name in xdataset.data_vars:
            xarray = xdataset[name]

            srcs = xarray.coords['src'].values
            recs = xarray.coords['rec'].values
            src_chans = xarray.coords['src_chan'].values
            rec_chans = xarray.coords['rec_chan'].values
            unique_stations = set(list(srcs)+list(recs))
            unique_channels = set(list(src_chans)+list(rec_chans))
            unique_pairs    = itertools.combinations(unique_stations,2)
            arg_list = itertools.product(unique_pairs, unique_channels, unique_channels)
            for parameter in arg_list:
                src = parameter[0][0]
                rec = parameter[0][1]
                src_chan = parameter[1]
                rec_chan = parameter[2]
                arg_combos= [dict(src=src, rec=rec, src_chan=src_chan, rec_chan=rec_chan),
                 dict(src=src, rec=rec, src_chan=rec_chan, rec_chan=src_chan),
                 dict(src=rec, rec=src, src_chan=src_chan, rec_chan=rec_chan),
                 dict(src=rec, rec=src, src_chan=rec_chan, rec_chan=src_chan)]

                arg_dict_to_use=None
                for subdict in arg_combos:
                    meta_record = df.loc[(df['src']     == subdict['src'])      & (df['rec']         ==subdict['rec'])  &
                                     (df['src channel'] == subdict['src_chan']) & (df['rec channel'] ==subdict['rec_chan'])]
                    arg_dict_to_use = subdict
                    if not meta_record.empty:
                        break
                record = xarray.loc[arg_dict_to_use]

                if not meta_record.empty:
                    station_1,network_1  = self._extract_station_network_info(src)
                    station_2,network_2  = self._extract_station_network_info(rec)
                    header_dict = {
                            'delta'    : meta_record['delta'].values[0],
                            'npts'     : record.data.shape[-1],
                            'starttime': starttime,
                            'station'  : '{}.{}'.format(station_1,station_2),
                            'channel'  : '{}.{}'.format(src_chan,rec_chan),
                            'network'  : '{}.{}'.format(network_1,network_2)
                            }
                    trace = Trace(data=record.data,header=header_dict)
                    if 'rec_latitude' in meta_record.columns:
                            trace.stats.coordinates = {
                                'src_latitude' :meta_record['src_latitude'].values[0],
                                'src_longitude':meta_record['src_longitude'].values[0],
                                'rec_latitude':meta_record['rec_latitude'].values[0],
                                'rec_longitude':meta_record['rec_longitude'].values[0]
                                }
                    traces.append(trace)

        return Stream(traces=traces)

    def _extract_timestamp(self,starttimestamp):
        starttimestamp = starttimestamp.astype(np.float64)/1e9
        return UTCDateTime(starttimestamp)

    def _extract_station_network_info(self,station):
        pairs = station.split('.')
        return pairs[1], pairs[0]

    def align_station_pairs(self,xdataset: xr.Dataset,dask_client = None):
        xdataset = xdataset.where(xdataset!=0)
        df = xdataset.attrs['df']
        merged_arrays = []
        for name in xdataset.data_vars:
            xarray = xdataset[name]
            arguments = itertools.product(xarray.coords['rec'].values, xarray.coords['src'].values)
            arrays = []
            for arg in arguments:
                if dask_client is not None:
                    arrays.append(dask_client.submit(self._align_pair,arg,xarray,df))
                else:
                    arrays.append(self._align_pair(arg, xarray,df))

            if dask_client is not None:
                merged_arrays.append(dask_client.submit(xr.merge,arrays))
            else:
                merged_arrays.append(xr.merge(arrays))
        if dask_client is not None:
            new_dataset = dask_client.submit(xr.merge,merged_arrays)
            new_dataset = new_dataset.result()
        else:
            new_dataset = xr.merge(merged_arrays)
        new_dataset.attrs['df']=df
        return new_dataset

    def _align_pair(self,pair_arg,xarray,df):
        rec = pair_arg[0]
        src = pair_arg[1]
        time_length   = len(xarray.coords['time'])
        xarray = xarray.transpose('src','rec','time','src_chan','rec_chan',transpose_coords=True)
        subset = xarray.loc[dict(rec=rec,src=src)]
        if rec==src:
            return self._create_radial_transverse_xarray(np.reshape(subset.data,
                                                                    (1,1,time_length,3,3))
                                                         ,xarray,src,rec)

        data  = subset.data
        count = np.count_nonzero(~np.isnan(data))
        if count == 0:
            subset = xarray.loc[dict(rec=src,src=rec)]
            rec = pair_arg[1]
            src = pair_arg[0]

       # get coordinates
        local_frame = df.loc[(df['src']==src) & (df['rec']==rec)].iloc[0]
        src_lat = local_frame['src_latitude']
        src_lon = local_frame['src_longitude']
        rec_lat = local_frame['rec_latitude']
        rec_lon = local_frame['rec_longitude']
        dist, az, backaz = gps2dist_azimuth(src_lat,src_lon,rec_lat,rec_lon)
        az      = np.deg2rad(az)
        backaz  = np.deg2rad(backaz)
        rotation_matrix_a = np.asarray([[1,           0,          0],
                                        [0,  np.cos(az), np.sin(az)],
                                        [0, -np.sin(az), np.cos(az)]])
        rotation_matrix_b = np.asarray([[1,               0,               0],
                                        [0, -np.cos(backaz), -np.sin(backaz)],
                                        [0,  np.sin(backaz), -np.cos(backaz)]]).T
        subset_data   = subset.data
        subset_data   = np.reshape(rotation_matrix_a @ subset_data @ rotation_matrix_b,
                                   (1,1,time_length,3,3))

        return self._create_radial_transverse_xarray(subset_data,xarray,src,rec)

    def _create_radial_transverse_xarray(self,data, original_xarray, src, rec):
        xarray_rotated = xr.DataArray(data,
                                dims=['src','rec','time','src_chan','rec_chan'],
                                coords={
                                    'rec_chan':['z','r','t'],
                                    'src_chan':['z','r','t'],
                                    'src':[src],
                                    'rec':[rec],
                                    'time':original_xarray.coords['time'].values
                                    },
                                name=original_xarray.name)
        return xarray_rotated


            #ok. so for each source-receiver we need:
        # | ZZ ZR ZT |       | 1    0     0     |   | ZZ  ZN  ZE |   | 1       0       0  |
        # | RZ RR RT |   =   | 0  cos(a) sin(a) |   | NZ  NN  NE |   | 0  -cos(b) -sin(b) |
        # | TZ TR TT |       | 0 -sin(a) cos(a) |   | EZ  EN  EE |   | 0   sin(b) -cos(b) |
        # where a is source-receiver azimuth and b is receiver-source azimuth.
        # lets rename each tensor according to stuff
        # Aligned_S1R1  = ROT_S1R1 * Unaligned_S1R1 * ROTINV_S1R1
        # in this example, unaligned S1R1 has dimensions of (3 src, 3 rec, t time)
        # to add a second pair example, we can do:
        # Aligned_S2R2  = ROT_S2R2 * Unaligned_S2R2 * ROTINV_S2R2
        # this represents an operation where 

    def save_result(self,result: xr.Dataset,directory):
        result = result.copy()
        df = result.attrs['df']
        del result.attrs['df']
        utils.make_dir(directory)
        result.to_netcdf(path='{}{}{}.nc'.format(directory,utils.sep,'result'))
        df.to_csv('{}{}{}.csv'.format(directory,utils.sep, 'metadata'))

    def load_result(self,directory):
        df     = pd.read_csv('{}{}{}.csv'.format(directory,utils.sep, 'metadata'))

        for col in list(df.columns):
            if 'Unnamed: 0'== col:
                df     = df.drop(columns=['Unnamed: 0'])
                break
        result = xr.load_dataset('{}{}{}.nc'.format(directory,utils.sep,'result'))
        result.attrs['df'] = df
        return result



class Anxcor(_AnxcorData, _AnxcorProcessor, _AnxcorConverter, _AnxcorConfig):



    def __init__(self,*args,**kwargs):
        super().__init__(**kwargs)