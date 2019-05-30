import os
from obspy.core import UTCDateTime

def _collect_components(stream):
    station_dict = {}
    for trace in stream.traces:

        key = trace.stats.station

        if key not in station_dict:
            station_dict[key] = []

        station_dict[key].append(trace)

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
    name = '/station:{}.{}.{}.{}'.format(station, component, starttime, format)
    trace.write(process_directory + name, format=format)


def process_and_save_to_file(args):
    process_with_file_write(*args)