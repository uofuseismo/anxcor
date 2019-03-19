from obspy import read
directory_1 = 'date_key_dir'
date_string_1 = '%Y%m%d'

streams = read()

for day, trace in enumerate(streams):
    trace.stats.starttime+= 60*60*24*day*2
    file_name = trace.stats.starttime.strftime(date_string_1) + '.sac'
    trace.write(directory_1 +'/'+ file_name, format="sac")
