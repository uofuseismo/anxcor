from typing import List
import  anxcor.abstractions as ab
import  anxcor.utils as os_utils
from  obspy.core import read, Stream, UTCDateTime
import xarray as xr
import pandas as pd
FLOAT_PRECISION = 1e-9
import numpy as np
#import sparse

def execute_if_ok_else_pass_through(method, one, two):
    if one is None and two is not None:
        return two
    elif one is not None and two is None:
        return one
    elif one is None and two is None:
        return None
    else:
        return method(one, two)

def method_per_op(method,opmethod, one, two):
    if one is None and two is not None:
        return opmethod(two)
    elif one is not None and two is None:
        return opmethod(one)
    elif one is None and two is None:
        return None
    else:
        return method(opmethod(one), opmethod(two))

class AnxcorDatabase:
    """ An interface for providing obspy.core.Stream objects to Anxcor objects

    get_stations() : List[str]
        returns a list of all stations in the dataset, formatted as a string with a network code
        e.g. 'Network.Station' ex: 'UU.1'

    get_waveforms(network : str =None, station : str =None,
                 starttime : float =None,endtime : =None,**kwargs) -> Stream:
        returns an obspy Stream of waveforms. Must take the kwargs:
        network and station as strings, and starttime, endtime as UTCDateTime timestamps
    """

    def __init__(self,**kwargs):
        pass

    def get_stations(self)-> List[str]:
        """
        returns a list of all stations in the dataset, formatted as a string with a network code
        e.g. 'Network.Station' ex: 'UU.1'

        Returns
        -------
        List[str]
            a list of strings representing stations. Must be formatted 'network_code.station_code'
        """
        raise NotImplementedError('Method: \'get_stations()\' method is not implemented')

    def get_waveforms(self,**kwarg_execute)->Stream:
        """
        returns an obspy Stream of waveforms

        Parameters
        ----------
        network : str,
            network to get waveform from. is called as a non-optional keyword-argument
        station : str,
            station to get waveform from. is called as a non-optional keyword-argument
        starttime : float,
            starttime of waveform in UTCDateTime timestamp. Non-optional keyword-argument
        endtime : float,
            endtime of waveform in UTCDateTime timestamp. Non-optional keyword-argument

        Returns
        -------
        Stream
            an obspy stream of traces

        Note
        ----
        Anxcor leaves both removing instrument response and ensuring continuity of data up to the user.
        If a given get_waveforms() query cannot deliver continuous data, it is perfectly fine to return a None object
        instead of an obspy Stream. Anxcor will handle the missing data accordingly.
        """
        raise NotImplementedError('Method: \'get_waveforms()\' is not implemented!')


class DataLoader(ab.AnxcorDataTask):

    def __init__(self,interp_method='nearest',**kwargs):
        super().__init__()
        self._kwargs['window_length'] =3600.0
        self._kwargs['interp_method']=interp_method
        self._seconds_buffer = 1.0
        self._datasets = {}

    def add_dataset(self, dataset: AnxcorDatabase, name: str, **kwargs):
        self._datasets[name]=dataset

    def get_stations(self) -> List[str]:
        """

        Returns
        -------
        station_list
            a list of all possible stations contained in the databases

        """
        station_list = []
        for key, value in self._datasets.items():
            one          = value
            seed_id_list = one.get_stations()
            for station in seed_id_list:
                if station not in station_list:
                    station_list.append(station)

        return station_list

    def has_data(self):
        return len(self._datasets.keys())>0

    def _combine(self, total_list, produced_list, key):
        for trace in produced_list:
            trace.stats.data_type = key
        total_list.extend(produced_list.traces)
        return total_list

    def _load_key(self,data_key,extension):
        load_key = '{}:{}@{}'.format('Trace IO',data_key,extension)
        return load_key

    def _response_mean_trend(self,data_key,extension):
        response_key = '{}:{}@{}'.format('RTrend, Mean, Resp',data_key,extension)
        return response_key

    def _stream_gather(self,data_key,extension):
        gather = '{}:{}@{}'.format('gather',data_key,extension)
        return gather

    def execute(self, *args, station=0, starttime=0, **kwargs):
        network, station = station.split('.')

        kwarg_execute    = {
            'network' : network,
            'station' : station,
            'starttime':starttime - self._seconds_buffer,
            'endtime':  starttime + self._seconds_buffer + self._kwargs['window_length']
            }

        traces = []
        for name, dataset in self._datasets.items():
            stream = dataset.get_waveforms(**kwarg_execute)
            for trace in stream:
                rate = trace.stats.sampling_rate
                # requires 2 points so as to match obspy's trim
                npts = int(rate*self._kwargs['window_length'])+1
                end_time = UTCDateTime(starttime+npts*trace.stats.delta)
                trace.stats.name=name
                if self._not_none_condition(trace,starttime,end_time):
                    trace.interpolate(rate,starttime=starttime,npts=npts,method=self._kwargs['interp_method'])
                else:
                    stream.remove(trace)
            traces = self._combine(traces, stream, name)
        return Stream(traces=traces)

    def _not_none_condition(self,trace,starttime,end_time):
        return starttime >= trace.stats.starttime.timestamp and \
        end_time < trace.stats.endtime and not np.isnan(trace.data).any()

    def _io_result(self, result, source, format='mseed', **kwargs):
        type_dict = {}
        path      = None
        for trace in result:
            trace_id = trace.get_id()
            time     = trace.stats.starttime.isoformat()
            path     = os_utils.make_path_from_list([self._file, source, time])

            trace.write(path + os_utils.sep + trace_id + '.' + format, format=format)
            type_dict[trace_id]=trace.stats.name

        new_result = read(path + os_utils.sep + '*.' + format)
        pass_on = []
        for trace in new_result:
            trace_id = trace.get_id()
            if trace_id in type_dict:
                trace.stats.name = type_dict[trace_id]
                pass_on.append(trace)

        return pass_on

    def _format_path(self, extension, format, trace_id):
        path = self._format_folder_path(extension)
        path = path + trace_id + '.' + format
        return path

    def _format_folder_path(self,extension):
        path = self._file + os_utils.sep + extension + os_utils.sep
        return path

    def _persist_metadata(self, *param, **kwargs):
        return None

    def _get_name(self,*args):
        return None

    def _get_process(self):
        return 'load'

    def _child_can_process(self, *args):
        return True

    def _window_key_convert(self,starttime=0):
        return UTCDateTime(int(starttime*100)/100).isoformat()

    def _additional_read_processing(self, result):
        name   = list(result.data_vars)[0]
        xarray       = result[name].copy()
        xarray.attrs = result.attrs.copy()
        del result
        return xarray



class XArrayCombine(ab.AnxcorDataTask):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def execute(self, first_data, second_data, **kwargs):
        if isinstance(first_data,xr.DataArray):
            first_data.attrs = {}
            first_data = first_data.to_dataset()
        if isinstance(second_data,xr.DataArray):
            second_data.attrs = {}
            second_data = second_data.to_dataset()
        result = execute_if_ok_else_pass_through(self._normal_combine,first_data,second_data)
        return result


    def _normal_combine(self, first_data, second_data):
        if isinstance(first_data, xr.DataArray) and isinstance(second_data, xr.DataArray):
            if first_data.name == second_data.name:
                result = xr.concat([first_data, second_data], dim='pair')
            else:
                result = xr.merge([first_data, second_data])

        elif isinstance(first_data, xr.Dataset) and isinstance(second_data, xr.DataArray):
            result = self._merge_DataArray_Dataset(first_data, second_data)
        elif isinstance(first_data, xr.DataArray) and isinstance(second_data, xr.Dataset):
            result = self._merge_DataArray_Dataset(second_data, first_data)
        else:
            result = xr.merge([first_data, second_data])
        if not isinstance(result, xr.Dataset):
            name   = result.name
            coords = result.coords
            result.attrs = {}
            result = xr.Dataset(data_vars={name: result}, coords=coords)
        return result

    def _merge_DataArray_Dataset(self, data_set, data_array):
        if data_array.name in data_set.data_vars.keys():
            result = xr.concat([data_set[data_array.name], data_array], dim='pair')
            result = xr.Dataset(data_vars={data_array.name: result},coords=result.coords)
            result = data_set.combine_first(result)
        else:
            result = xr.merge([data_set, data_array])
        return result

    def _persist_metadata(self, first_data, second_data, **kwargs):
        if first_data is None and second_data is None:
            return None
        elif first_data is None and second_data is not None:
            return second_data.attrs
        elif first_data is not None and second_data is None:
            return first_data.attrs
        else:
            attrs_1 = first_data.attrs
            attrs_2 = second_data.attrs
            assert 'df' in attrs_1.keys(), 'no dataframe in 1!!! attrs is {}'.format(attrs_1)
            assert 'df' in attrs_2.keys(), 'no dataframe in 2!!! attrs is {}'.format(attrs_1)
            df_1 = attrs_1['df']
            df_2 = attrs_2['df']
            attrs = {'df': pd.concat([df_1,df_2],ignore_index=True)}
            if attrs['df'].isnull().values.any():
                print('isnull')
            return attrs

    def _io_result(self, result, *args, **kwargs):
        return result

    def _get_process(self):
        return 'combine'

    def starttime_parser(self,first,second):
        return 'depth:{}branch:{}'.format(first, second)

    def _nonetype_returned_message(self, **kwargs):
        pass



class XArrayStack(ab.XArrayProcessor):

    def __init__(self,norm_procedure=0,**kwargs):
        super().__init__(**kwargs)
        self._kwargs['norm_procedure']=norm_procedure


    def execute(self, first: xr.Dataset, second: xr.Dataset, *args, **kwargs):
        if first is None and second is not None:
            return second
        elif first is not None and second is None:
            return first
        elif first is None and second is None:
            return None
        else:
            first_aligned, second_aligned = xr.align(first, second, join='outer')
            first_aligned_copy = first_aligned.copy()
            first_var_set = set(list(first.data_vars))
            second_var_set= set(list(second.data_vars))
            intersection  = first_var_set.intersection(second_var_set)
            for var in intersection:
                first_array  = first_aligned[var]
                second_array = second_aligned[var]
                first_array.data  = np.nan_to_num(first_array.data)
                second_array.data = np.nan_to_num(second_array.data)
                result = first_array + second_array
                first_aligned_copy[var]=result

            in_second_not_first = second_var_set - intersection
            for var in in_second_not_first:
                first_aligned_copy[var]=second_aligned[var]

        return first_aligned_copy


    def _persist_metadata(self, xarray_1, xarray_2, *args, **kwargs):
        return  method_per_op(self._combine_metadata,self._getattr,xarray_1,xarray_2)

    def _lambda_add(self,x_stacks,y_stacks):
        return int(x_stacks + y_stacks)

    def _combine_metadata(self,attrs1, attrs2):
        df_1 = attrs1['df']
        df_2 = attrs2['df']
        col_names= ['rec','src','src channel','rec channel','delta','operations']
        if 'rec_latitude' in df_1.columns:
            col_names = col_names + ['src_latitude','rec_latitude','src_longitude','rec_longitude']
        if 'rec_elevation' in df_2.columns:
            col_names = col_names + ['src_elevation','rec_elevation']
        df_joined = df_1.merge(df_2,on=col_names,how='outer')
        df_joined.fillna(0,inplace=True)
        df_joined['stacks']= df_joined.apply(lambda x:   self._lambda_add(x['stacks_x'],x['stacks_y']),axis=1)
        df_joined.drop(columns=['stacks_x','stacks_y'],inplace=True)
        return {'df':df_joined}

    def _getattr(self,array):
        return array.attrs

    def _get_lower(self,one,two,key):
        if one[key] < two[key]:
            return one[key]
        return two[key]

    def _get_upper(self,one,two,key):
        if one[key] > two[key]:
            return one[key]
        return two[key]

    def _get_process(self):
        return 'stack'

    def _child_can_process(self, xarray1, xarray2, *args):
        return xarray1 is not None or xarray2 is not None

    def starttime_parser(self,first,second):
        return 'depth:{}branch:{}'.format(first, second)

    def _window_key_convert(self,starttime=0):
        return starttime





