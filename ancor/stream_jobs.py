import os
from obspy.core import UTCDateTime

def _collect_components(stream):
    station_dict = {}

    for trace in stream.traces:
        network = trace.stats.network
        station = trace.stats.station

        if network not in station_dict:
            station_dict[network]={}

        if station not in station_dict[network]:
            station_dict[network][station]=[]

        station_dict[network][station].append(trace)

    return station_dict


def process_with_file_write(window_number,
                            directory,
                            format,
                            starttime,
                            endtime,
                            worker,
                            database):
    print('starting job #{} start{} stop{}'.format(window_number,   UTCDateTime(starttime).isoformat(),
                                                                    UTCDateTime(endtime).isoformat()))
    process_directory   = directory+'/'+str(window_number)
    stream              = database.get_waveforms(starttime=starttime,endtime=endtime)
    component_dict      = _collect_components(stream)
    if not os.path.exists(process_directory):
        os.mkdir(process_directory)
    for key, value in component_dict.items():
        output = worker(value)
        for trace in output:
            _write_trace_to_file(format, process_directory, trace)


def _write_trace_to_file(format, process_directory, trace):
    station = trace.stats['station']
    component = trace.stats['channel']
    starttime = trace.stats['starttime'].isoformat()
    name = '/{}.{}.{}.{}'.format(station, component, starttime, format)
    trace.write(process_directory + name, format=format)


def process_and_save_to_file(args):
    process_with_file_write(*args)

def window_correlator(stream,worker):
    correlations = worker(stream)
    return correlations

def window_worker(starttime,duration,worker,database):
    stream         = database.get_waveforms(starttime=starttime, endtime=starttime+duration)
    network_dict = _collect_components(stream)
    traces = []
    for network, station_dict in network_dict.items():
        for station, component_traces in station_dict.items():
            traces = traces + worker(component_traces)

    return traces

def write_worker(filepath,traces,format='sac'):
    for trace in traces:
        station = trace.stats['station']
        component = trace.stats['channel']
        starttime = trace.stats['starttime'].isoformat()
        name = '/{}.{}.{}.{}'.format(station, component, starttime, format)
        trace.write(filepath + name, format=format)
