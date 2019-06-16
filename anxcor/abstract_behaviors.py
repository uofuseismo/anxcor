import os_utils as os_utils
from obspy.core import UTCDateTime
import xarray as xr


class _XDaskTask:

    def __init__(self):
        self._file=None

    def __call__(self, *args,dask_client=None, **kwargs):
        if dask_client is None:
            result = self._single_thread_execute(*args,**kwargs)
        else:
            result = self._dask_task_execute(*args, dask_client=dask_client,**kwargs)

        if self._file is not None:
            result = self._io_result(result,*args,**kwargs)
        return result

    def write_combined_result(self,result,dask_client=None):
        pass


    def write_to_file(self,file,**kwargs):
        if not os_utils.folder_exists(file):
            os_utils.make_dir(file)
        self._file = file
        self._io_kwargs = kwargs

    def _single_thread_execute(self,*args,**kwargs):
        pass

    def _dask_task_execute(self,*args,**kwargs):
        pass

    def _get_process_signature(self):
        return '.'

    def _get_station_key(self,result):
        return list(result['station_id'].values)[0]

    def _get_window_key(self,result):
        utc = UTCDateTime(int(result.attrs['starttime']*100)/100)
        fmt = utc.isoformat()
        return  fmt

    def _io_result(self, result, *args, **kwargs):
        full_path = self._get_io_path(result)

        self._write(result,full_path)
        xarray = self._read(full_path)
        return xarray

    def _get_io_path(self, result):
        station = self._get_station_key(result)
        path = [self._file, self._get_process_signature(), self._get_window_key(result)]
        path = [i for i in path if i]
        full_path = os_utils.make_path_from_list(path)
        if station is not None:
            full_path = full_path + os_utils.sep + station
        return full_path

    def _write(self,result,path):
        if isinstance(result, xr.DataArray):
            result_ds = result.to_dataset()
            result_ds.attrs = result.attrs
        else:
            result_ds = result
        result_ds.to_netcdf(path + '.nc')

    def _read(self,path):
        read_ds = xr.open_dataset(path + '.nc')
        xarray = read_ds.to_array()
        xarray.name = xarray['variable'].values[0]
        del xarray.coords['variable']
        xarray = xarray.squeeze(dim='variable', drop=True)
        return xarray