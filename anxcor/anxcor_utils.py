import xarray as xr
import pandas as pd
import itertools
from obspy.core import Stream, Trace, UTCDateTime

def xarray_3D_to_2D( xdataset: xr.Dataset):
    # get vars
    new_ds = xdataset.assign_coords(component='{}{}{}'.format(xdataset.src_chan, ':', xdataset.rec_chan))
    new_ds = new_ds.drop_dims(['src_chan', 'rec_chan'])
    return new_ds


def xarray_2D_to_3D( xdataset: xr.Dataset):
    new_ds = xdataset.assign_coords(src_chan=xdataset.component.split(':')[0])
    new_ds = new_ds.assign_coords(rec_chan=xdataset.component.split(':')[1])
    new_ds = new_ds.drop_dims(['component'])
    return new_ds


def xarray_to_obspy( xdataset: xr.Dataset):
    df = xdataset.attrs['df']
    traces = []
    starttime = list(xdataset.coords['time'].values)[0]
    starttime = _extract_timestamp(starttime)
    for name in xdataset.data_vars:
        xarray = xdataset[name]

        srcs = xarray.coords['src'].values
        recs = xarray.coords['rec'].values
        src_chans = xarray.coords['src_chan'].values
        rec_chans = xarray.coords['rec_chan'].values
        unique_stations = set(list(srcs) + list(recs))
        unique_channels = set(list(src_chans) + list(rec_chans))
        unique_pairs = itertools.combinations(unique_stations, 2)
        arg_list = itertools.product(unique_pairs, unique_channels, unique_channels)
        for parameter in arg_list:
            src = parameter[0][0]
            rec = parameter[0][1]
            src_chan = parameter[1]
            rec_chan = parameter[2]
            arg_combos = [dict(src=src, rec=rec, src_chan=src_chan, rec_chan=rec_chan),
                          dict(src=src, rec=rec, src_chan=rec_chan, rec_chan=src_chan),
                          dict(src=rec, rec=src, src_chan=src_chan, rec_chan=rec_chan),
                          dict(src=rec, rec=src, src_chan=rec_chan, rec_chan=src_chan)]

            arg_dict_to_use = None
            for subdict in arg_combos:
                meta_record = df.loc[(df['src'] == subdict['src']) & (df['rec'] == subdict['rec']) &
                                     (df['src channel'] == subdict['src_chan']) & (
                                                 df['rec channel'] == subdict['rec_chan'])]
                arg_dict_to_use = subdict
                if not meta_record.empty:
                    break
            record = xarray.loc[arg_dict_to_use]

            if not meta_record.empty:
                station_1, network_1 = _extract_station_network_info(src)
                station_2, network_2 = _extract_station_network_info(rec)
                header_dict = {
                    'delta': meta_record['delta'].values[0],
                    'npts': record.data.shape[-1],
                    'starttime': starttime,
                    'station': '{}.{}'.format(station_1, station_2),
                    'channel': '{}.{}'.format(src_chan, rec_chan),
                    'network': '{}.{}'.format(network_1, network_2)
                }
                trace = Trace(data=record.data, header=header_dict)
                if 'rec_latitude' in meta_record.columns:
                    trace.stats.coordinates = {
                        'src_latitude': meta_record['src_latitude'].values[0],
                        'src_longitude': meta_record['src_longitude'].values[0],
                        'rec_latitude': meta_record['rec_latitude'].values[0],
                        'rec_longitude': meta_record['rec_longitude'].values[0]
                    }
                traces.append(trace)

    return Stream(traces=traces)


def _extract_timestamp( starttimestamp):
    starttimestamp = starttimestamp.astype(np.float64) / 1e9
    return UTCDateTime(starttimestamp)


def _extract_station_network_info( station):
    pairs = station.split('.')
    return pairs[1], pairs[0]

def _create_rotation_matrix(coords,df):
    pass

def align_station_pairs( xdataset: xr.Dataset):
    attrs = xdataset.attrs.copy()
    del attrs['delta']
    del attrs['operations']
    component_possibilities=['n','z','e']
    components = itertools.permutations(component_possibilities,2)
    for name in xdataset.data_vars:
        xarray = xdataset[name]
        coords            = xarray.coords
        df                = xarray.attrs['df']#unsure how this works
        rot_array,new_coords = _create_rotation_matrix(coords,df)
        xarray.data       = rot_array.T @ xarray.data @ rot_array