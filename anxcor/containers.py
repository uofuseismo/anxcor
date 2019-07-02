from typing import List
import  anxcor.abstractions as ab
import  anxcor.utils as os_utils
from  obspy.core import read, Stream, UTCDateTime
import xarray as xr


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

    def __init__(self):
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


class DataLoader(ab.XDatasetProcessor):

    def __init__(self, window_length):
        super().__init__()
        self._window_length = window_length
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

    def _single_thread_execute(self,*args, station=0, starttime=0, **kwargs):
        network, station = station.split('.')

        kwarg_execute    = {
            'network' : network,
            'station' : station,
            'starttime':starttime,
            'endtime':  starttime + self._window_length
            }

        traces = []
        for name, dataset in self._datasets.items():
            stream = dataset.get_waveforms(**kwarg_execute)
            stream = self._curate(stream,starttime)
            traces = self._combine(traces, stream, name)
        return Stream(traces=traces)

    def _io_result(self, result, starttime, source, format='mseed', **kwargs):
        type_dict = {}
        path      = None
        for trace in result:
            trace_id = trace.get_id()
            time     = trace.stats.starttime.isoformat()
            path     = os_utils.make_path_from_list([self._file, source, time])

            trace.write(path + os_utils.sep + trace_id + '.' + format, format=format)
            type_dict[trace_id]=trace.stats.data_type

        new_result = read(path + os_utils.sep + '*.' + format)
        pass_on = []
        for trace in new_result:
            trace_id = trace.get_id()
            if trace_id in type_dict:
                trace.stats.data_type = type_dict[trace_id]
                pass_on.append(trace)

        return pass_on

    def _format_path(self, extension, format, trace_id):
        path = self._format_folder_path(extension)
        path = path + trace_id + '.' + format
        return path

    def _format_folder_path(self,extension):
        path = self._file + os_utils.sep + extension + os_utils.sep
        return path

    def _metadata_to_persist(self, *param, **kwargs):
        return None

    def _get_name(self,*args):
        return None

    def _get_process(self):
        return 'load'

    def _curate(self, stream, starttime):
        endtime = starttime + self._window_length
        valid_traces = []
        for trace in stream:
            st = trace.stats.starttime.timestamp
            end= trace.stats.endtime.timestamp
            delta=trace.stats.delta

            cond1 = abs(st - starttime) < delta
            cond2 = abs(end - endtime) < delta
            if cond1 and cond2:
                valid_traces.append(trace)

        return Stream(traces=valid_traces)

    def _should_process(self, *args):
        return True

    def _window_key_convert(self,window):
        return UTCDateTime(int(window*100)/100).isoformat()

    def _addition_read_processing(self, result):
        name   = list(result.data_vars)[0]
        xarray       = result[name].copy()
        xarray.attrs = result.attrs.copy()
        del result
        return xarray



class XArrayCombine(ab.XDatasetProcessor):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def _single_thread_execute(self,first_data, second_data,**kwargs):
        if isinstance(first_data,xr.DataArray):
            first_data = first_data.to_dataset()
        if isinstance(second_data,xr.DataArray):
            second_data = second_data.to_dataset()
        return execute_if_ok_else_pass_through(self._normal_combine,first_data,second_data)

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
            name = result.name
            coords = result.coords
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

    def _metadata_to_persist(self, first_data, second_data, **kwargs):
        if first_data is None and second_data is None:
            return None
        elif first_data is None and second_data is not None:
            return {**self._extract_metadata_dict(second_data)}
        elif first_data is not None and second_data is None:
            return {**self._extract_metadata_dict(first_data)}
        else:
            persist1 = self._extract_metadata_dict(first_data)
            persist2 = self._extract_metadata_dict(second_data)
        result   = {**persist1,**persist2}
        return result

    def _extract_metadata_dict(self, data_array):
        if isinstance(data_array, xr.DataArray) and len(list(data_array.coords['pair'].values))==1:
            key = list(data_array.coords['pair'].values)[0]
            dict1 = {key: {'stacks': data_array.attrs['stacks'],
                           'starttime': data_array.attrs['starttime'],
                           'endtime': data_array.attrs['endtime']},
                           'delta': data_array.attrs['delta'],
                           'operations': data_array.attrs['operations']}

            if 'location' in data_array.attrs.keys():
                dict1[key]['location']=data_array.attrs['location']
        else:
            dict1 = data_array.attrs

        return dict1

    def _io_result(self, result, *args, **kwargs):
        return result

    def _get_name(self, *args):
        return None

    def _get_process(self):
        return 'combine'

    def _time_signature(self,time):
        return time

    def _window_key_convert(self,window):
        return window

    def starttime_parser(self,first,second):
        return 'depth:{}branch:{}'.format(first, second)



class XArrayStack(ab.XArrayProcessor):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)


    def _single_thread_execute(self,first: xr.DataArray, second: xr.DataArray,*args,**kwargs):
        if first is None and second is not None:
            return second
        elif first is not None and second is None:
            return first
        elif first is None and second is None:
            return None
        else:
            result       = first + second
        return result


    def _metadata_to_persist(self, xarray_1,xarray_2,*args, **kwargs):
        return  method_per_op(self._combine_metadata,self._getattr,xarray_1,xarray_2)

    def _combine_metadata(self,attrs1, attrs2):
        attrs = {}
        attrs['delta'] = attrs1['delta']
        attrs['starttime'] = self._get_lower(attrs1, attrs2, 'starttime')
        attrs['endtime'] = self._get_upper(attrs1, attrs2, 'endtime')
        attrs['stacks'] = attrs1['stacks'] + attrs2['stacks']
        attrs['operations'] = attrs1['operations']
        if 'location' in attrs1.keys():
            attrs['location'] = attrs1['location']
        return attrs

    def _getattr(self,array):
        return array.attrs

    def _get_name(self,one,two):
        if one is not None:
            return one.name
        if two is not None:
            return two.name
        return None

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

    def _time_signature(self,time):
        return time

    def _window_key_convert(self,window):
        return window

    def starttime_parser(self,first,second):
        return 'depth:{}branch:{}'.format(first, second)





