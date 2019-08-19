import  anxcor.utils as utils
from obspy.core import UTCDateTime
import xarray as xr
import sys
import numpy as np
import json

def write(xarray, path, extension):
    array_path      = '{}{}{}{}'.format(path, utils.sep, extension, '.nc')
    attributes_path = '{}{}{}{}'.format(path, utils.sep, extension, '.metadata.json')
    data = xarray.copy()
    data.attrs = {}
    data.to_netcdf(array_path)
    with open(attributes_path, 'w') as p_file:
        json.dump(xarray.attrs, p_file, sort_keys=True, indent=4)


def read(path, extension):
    xarray_path     ='{}{}{}{}'.format(path, utils.sep, extension, '.nc')
    attributes_path ='{}{}{}{}'.format(path, utils.sep, extension, '.metadata.json')
    try:
        xarray = xr.open_dataset(xarray_path)
        with open(attributes_path, 'r') as p_file:
            attrs = json.load(p_file)
    except FileNotFoundError:
        print('File:\n {}\n not found. Ignoring window'.format(attributes_path))
        return None
    xarray.attrs = attrs
    return xarray


class _IO:

    def __init__(self, dir):
        self._file      = dir
        self._isenabled = False

    def enable(self):
        self._isenabled=True

    def is_enabled(self):
        return self._isenabled

    def get_folder_extension(self, xarray):
        if 'stacks' in xarray.attrs.keys():
            # then xarray represents stack data
            pair = list(xarray.coords['pair'].values)[0]
            stack_num = str(xarray.attrs['stacks'])
            return self._file + utils.sep + pair + utils.sep + stack_num
        elif 'operations' in xarray.attrs.keys():
            # then xarray represents single station data
            operation = xarray.attrs['operations'].split('\n')[-1]
            starttime = UTCDateTime(xarray.attrs['starttime']).isoformat()
            return self._file + utils.sep + operation + utils.sep + starttime
        else:
            # then xarray is a dataset
            pair_list = list(xarray.coords['pair'].values)
            strsum = pair_list[0] + '|' + str(len(pair_list))
            return self._file + utils.sep + strsum

    def get_filename(self, xarray):
        if 'stacks' in xarray.attrs.keys():
            # then xarray represents stack data
            starttime = UTCDateTime(xarray.attrs['starttime']).isoformat()
            return starttime
        elif 'operations' in xarray.attrs.keys():
            # then xarray represents single station data
            station = list(xarray.coords['station_id'].values)[0]
            return station
        else:
            return 'combined_data'


class _XArrayWrite(_IO):

    def __init__(self, directory=None):
        super().__init__(dir)

    def set_folder(self, file):
        self.enable()
        if not utils.folder_exists(file):
            utils.make_dir(file)
        self._file = file

    def _chkmkdir(self,dir):
        if not utils.folder_exists(dir):
            utils.make_dir(dir)

    def __call__(self, xarray, process, folder, file, dask_client=None, **kwargs):
        if self._file is not None and xarray is not None:
            folder    = '{}{}{}{}{}'.format(self._file, utils.sep, process, utils.sep, folder)
            self._chkmkdir(folder)
            write(xarray, folder, file)
        return None


class _XArrayRead(_IO):

    def __init__(self, directory=None):
        super().__init__(directory)
        self._file = directory

    def set_folder(self, directory):
        self.enable()
        if not utils.folder_exists(directory):
            utils.make_dir(directory)
        self._file = directory

    def __call__(self, xarray, process, folder, file, dask_client=None, **kwargs):
        result = None
        if self._file is not None:
            folder = self._file + utils.sep + process + utils.sep + folder
            result = read(folder, file)

        if result is not None:
            return result
        else:
            return xarray

class AnxcorTask:

    def __init__(self,dummy_task=False,**kwargs):
        self._kwargs = kwargs
        self._kwargs['dummy_task']=dummy_task
        self.read  = _XArrayRead(None)
        self.write = _XArrayWrite(None)
        self._enabled = True

    def disable(self):
        self._enabled=False

    def set_io_task(self, folder, action, **kwargs):
        if action=='save':
            self.write.set_folder(folder)
        else:
            self.read.set_folder(folder)


    def set_kwargs(self, kwarg):
        for key, value in kwarg.items():
            if key in self._kwargs.keys():
                self._kwargs[key]=value
            else:
                print('key [{}] is not a assignable parameter for {}\n'.format(key,self.get_name())+\
                    'skipping...')

    def get_kwargs(self):
        return {**self._kwargs}

    def get_name(self):
        return self._get_process()

    def __call__(self, *args, dask_client=None, **kwargs):
        key = self._get_operation_key(**kwargs)
        self._fire_and_forget = None

        result = None
        if self._enabled and not self.read.is_enabled() and self._should_process(*args):
                if dask_client is None:
                    result = self._execute(*args,**kwargs)
                else:
                    result = dask_client.submit(self._execute, *args, key=key,**kwargs)
        result = self._io_operations(*args, dask_client=dask_client, result=result,**kwargs)
        return result

    def _execute(self, *args, **kwargs):
        try:
            persist_name       = self.__get_name(*args)
            persisted_metadata = self.__metadata_to_persist(*args, **kwargs)
            if args is None or len(args)==1 and args[0] is None:
                result = None
            else:
                result = self._single_thread_execute(*args, **kwargs)
        except Exception as e:
            print(
            '********error********* \n\n {} \n\n from function \n \n {}' +\
                    '\n\n returning \'None\' and continuing'.format(str(e), self._get_process()))
            result = None
        if result is not None:
            self._assign_metadata(persist_name, persisted_metadata, result)
        return result

    def _assign_metadata(self, persist_name, persisted_metadata, result):
        if result is None:
            return None
        if persisted_metadata is not None:
            result.attrs = persisted_metadata
        if persist_name is not None:
            result.name  = persist_name

    def _io_operations(self, *args, result=None, **kwargs):
        key = self._get_operation_key(**kwargs)
        file, folder, process = self._get_io_string_vars(**kwargs)
        if self.read.is_enabled():
            result = self._dask_read_execute(*args, result=result, process=process,folder=folder,file=file,key=key,**kwargs)
        elif self.write.is_enabled():
            self._dask_write_execute(process=process, result=result, folder=folder,file=file,key=key,**kwargs)
        return result

    def _get_io_string_vars(self, starttime=0, station=0,**kwargs):
        process = self._get_process()
        folder  = self._window_key_convert(starttime=starttime)
        file    = station
        return file, folder, process

    def _read_execute(self, result, process, folder, file):
        result = self.read(result, process,folder,file)
        result = self._additional_read_processing(result)
        return result

    def _single_thread_execute(self,*args,**kwargs):
        pass

    def _dask_read_execute(self,*args,dask_client=None,result=None,process=None,folder=None,file=None,key=None,**kwargs):
        if dask_client is None:
            result = self._read_execute(result, process, folder, file)
        else:
            result = dask_client.submit(self._read_execute, *args, process, folder, file, key=key)
        return result

    def _dask_write_execute(self,dask_client=None,result=None,process=None,folder=None,file=None,key=None,**kwargs):
        if dask_client is None:
            self.write(result, process, folder, file)
        else:
            if self._fire_and_forget is None:
                from dask.distributed import fire_and_forget
                self._fire_and_forget = fire_and_forget
            end = dask_client.submit(self.write, result, process, folder, file, key='writing: ' + key)
            if self._fire_and_forget is not None:
                self._fire_and_forget(end)

    def __metadata_to_persist(self,*param,**kwargs):
        if param is None or (len(param)==1 and param[0] is None):
            return None
        else:
            return self._metadata_to_persist(*param,**kwargs)

    def __get_name(self,*param,**kwargs):
        if param is None or (len(param)==1 and param[0] is None):
            return None
        else:
            return self._get_name(*param,**kwargs)


    def _metadata_to_persist(self, *param, **kwargs):
        if len(param)==1:
            attrs = param[0].attrs.copy()
        else:
            attrs = {**param[0].attrs.copy(), **param[1].attrs.copy()}
        added_kv_metadata = self._add_metadata_key()
        add_operation     = self._add_operation_string()
        if added_kv_metadata is not None:
            if isinstance(added_kv_metadata[0],tuple) or isinstance(added_kv_metadata[0],list):
                for key, value in added_kv_metadata:
                    attrs[added_kv_metadata[0]] = added_kv_metadata[1]
            else:
                attrs[added_kv_metadata[0]] = added_kv_metadata[1]
        if add_operation is not None:
            attrs['operations']=attrs['operations'] + '\n' + add_operation
        return attrs

    def _get_name(self,*args,**kwargs):
        if len(args) == 1:
            return args[0].name
        elif args[0] is None and args[1] is not None:
            return args[1].name
        elif args[0] is not None and args[1] is None:
            return args[0].name
        else:
            return args[0].name + ':' + args[1].name

    def _add_operation_string(self):
        return None

    def _add_metadata_key(self):
        return None

    def _get_process(self):
        return 'process'

    def _additional_read_processing(self, result):
        return result

    def _window_key_convert(self,starttime=0):
        return starttime

    def _get_operation_key(self,starttime=0,station=0,**kwargs):
        window_key = self._window_key_convert(starttime=starttime)
        return '{}-{}-{}'.format(self._get_process(),station,window_key)

    def _should_process(self, *args):
        return True


class XArrayProcessor(AnxcorTask):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def _additional_read_processing(self, result):
        if result is not None:
            name   = list(result.data_vars)[0]
            xarray       = result[name].copy()
            xarray.attrs = result.attrs.copy()
            del result
            return xarray
        return result

    def  _should_process(self,xarray, *args):
        return xarray is not None

    def _window_key_convert(self,starttime=0):
        return UTCDateTime(int(starttime*100)/100).isoformat()


class AnxcorDataTask(AnxcorTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _additional_read_processing(self, result):
        return result


class XArrayRolling(XArrayProcessor):
    """
    whitens the frequency spectrum of a given xarray
    """
    def __init__(self, window=1.0,approach='rcc',center=True,
                 rolling_metric='mean',reduce_metric='mean',**kwargs):
        super().__init__(**kwargs)
        self._kwargs = {
            'window' : window,
            'approach' : approach,
            'center' : center,
            'reduce_metric' : reduce_metric,
            'rolling_metric': rolling_metric}

    def _single_thread_execute(self, xarray: xr.DataArray, *args, **kwargs):

        processed_array = self._preprocess(xarray)
        pre_rolling = self._pre_rolling_process(processed_array, xarray)
        rolling_array = self._apply_rolling_method(pre_rolling, xarray)
        dim = self._get_longest_dim_name(rolling_array)
        rolling_array = rolling_array.ffill(dim).bfill(dim)

        post_rolling = self._post_rolling_process(rolling_array, xarray)
        rolling_processed = self._reduce_by_channel(post_rolling)
        normalized_array = processed_array / rolling_processed
        final_processed = self._postprocess(normalized_array, xarray)

        return final_processed

    def _get_longest_dim_name(self,xarray):
        coords  = xarray.dims
        index = np.argmax(xarray.data.shape)
        return coords[index]

    def _preprocess(self, xarray : xr.DataArray) -> xr.DataArray:
        return xarray

    def _pre_rolling_process(self,processed_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return processed_array

    def _post_rolling_process(self,rolled_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return rolled_array

    def _postprocess(self,normed_array : xr.DataArray, xarray : xr.DataArray)-> xr.DataArray:
        return normed_array

    def _get_rolling_samples(self,processed_xarray : xr.DataArray, xarray: xr.DataArray)-> int:
        return int(self._kwargs['window'] / xarray.attrs['delta'])

    def _apply_rolling_method(self, processed_xarray, original_xarray):
        rolling_samples = self._get_rolling_samples(processed_xarray, original_xarray)
        rolling_procedure = self._kwargs['rolling_metric']
        dim = self._get_longest_dim_name(processed_xarray)
        rolling_dict = {dim: rolling_samples,
                        'center': self._kwargs['center'],
                        'min_periods':rolling_samples
                        }
        if rolling_procedure == 'mean':
            xarray = processed_xarray.rolling(**rolling_dict).mean()
        elif rolling_procedure == 'median':
            xarray = processed_xarray.rolling(**rolling_dict).median()
        elif rolling_procedure == 'min':
            xarray = processed_xarray.rolling(**rolling_dict).min()
        elif rolling_procedure == 'max':
            xarray = processed_xarray.rolling(**rolling_dict).max()
        else:
            xarray = processed_xarray

        xarray.attrs = processed_xarray.attrs
        return xarray

    def _reduce_by_channel(self, xarray):
        approach = self._kwargs['approach']
        if approach == 'src':
            reduction_procedure = self._kwargs['reduce_metric']
            if reduction_procedure == 'mean' or reduction_procedure is None:
                xarray = xarray.mean(dim='channel')
            elif reduction_procedure == 'median':
                xarray = xarray.median(dim='channel')
            elif reduction_procedure == 'min':
                xarray = xarray.min(dim='channel')
            elif reduction_procedure == 'max':
                xarray = xarray.max(dim='channel')
            elif 'z' in reduction_procedure.lower() or 'n' in reduction_procedure.lower() \
                    or 'e' in reduction_procedure.lower():
                for coordinate in list(xarray.coords['channel'].values):
                    if reduction_procedure in coordinate.lower():
                        return xarray[dict(channel=coordinate)]
        return xarray

    def _get_process(self):
        return 'rolling operation'
