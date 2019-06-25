from typing import List
import  anxcor.abstractions as ab
import  anxcor.utils as os_utils
from  obspy.core import read
import xarray as xr


class AnxorDatabase:

    def __init__(self):
        pass

    def get_stations(self):
        raise NotImplementedError('Method: \'get_stations()\' method is not implemented')

    def get_waveforms(**kwarg_execute):
        raise NotImplementedError('Method: \'get_waveforms()\' is not implemented!')


class DataLoader(ab.XArrayProcessor):

    def __init__(self, window_length):
        super().__init__()
        self._window_length = window_length
        self._datasets = {}

    def add_dataset(self, dataset: AnxorDatabase, name: str, trace_prep=None, **kwargs):
        if trace_prep is None:
            def _remove_response(stream):
                for trace in stream:
                    try:
                        trace.remove_response(output='DISP')
                    except Exception:
                        pass
                return stream

            trace_prep=_remove_response
        self._datasets[name]=(dataset, trace_prep, kwargs)

    def get_stations(self) -> List[str]:
        """

        Returns
        -------
        station_list
            a list of all possible stations contained in the databases

        """
        station_list = []
        for key, value in self._datasets.items():
            one          = value[0]
            seed_id_list = one.get_stations()
            for station in seed_id_list:
                if station not in station_list:
                    station_list.append(station)

        return station_list

    def _combine(self, total_list, produced_list, key):
        for trace in produced_list:
            trace.stats.data_type = key
        total_list.extend(produced_list)
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

    def _single_thread_execute(self, starttime, station, *args):
        network, station = station.split('.')

        kwarg_execute    = {
            'network' : network,
            'station' : station,
            'starttime':starttime,
            'endtime':starttime + self._window_length
            }

        traces = []
        for k, value in self._datasets.items():

            stream = value[0].get_waveforms(**kwarg_execute)
            treated= value[1](stream)
            traces = self._combine(traces, treated, k )
        return traces

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

class XArrayCombine(ab.XDatasetProcessor):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def _single_thread_execute(self,first_data, second_data,**kwargs):
        if isinstance(first_data,xr.DataArray) and isinstance(second_data,xr.DataArray):
            if first_data.name==second_data.name:
                result = xr.concat([first_data,second_data],dim='pair')
            else:
                result = xr.merge([first_data,second_data])

        elif isinstance(first_data,xr.Dataset) and isinstance(second_data,xr.DataArray):
            result = self._merge_DataArray_Dataset(first_data, second_data)
        elif isinstance(first_data,xr.DataArray) and isinstance(second_data,xr.Dataset):
            result = self._merge_DataArray_Dataset(second_data,first_data)
        else:
            result = xr.merge([first_data,second_data])
        if not isinstance(result,xr.Dataset):
            name   = result.name
            coords = result.coords
            result = xr.Dataset(data_vars={name: result},coords=coords)
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


    def _single_thread_execute(self,first: xr.DataArray, second: xr.DataArray):
        result       = first + second
        return result


    def _metadata_to_persist(self, xarray_1,xarray_2, **kwargs):
        attrs = {}
        attrs['delta']     = xarray_1.attrs['delta']
        attrs['starttime'] = self._get_lower(xarray_1.attrs, xarray_2.attrs, 'starttime')
        attrs['endtime']   = self._get_upper(xarray_1.attrs, xarray_2.attrs, 'endtime')
        attrs['stacks']    = xarray_1.attrs['stacks'] + xarray_2.attrs['stacks']
        attrs['operations']= xarray_1.attrs['operations']
        if 'location' in xarray_1.attrs.keys():
            attrs['location'] = xarray_1.attrs['location']
        return attrs

    def _get_name(self,one,two):
        return one.name

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





