
from obspy import read
directory_1 = 'date_string_1'
directory_2 = 'date_string_2'
date_string_1 = '%Y%m%d'
date_string_2 = '%m%d%Y'

streams = read()

for day, trace in enumerate(streams):
    trace.stats.starttime+= 60*60*24*day
    file_name = trace.stats.starttime.strftime(date_string_1) + '.sac'
    trace.write(directory_1 +'/'+ file_name, format="sac")

for day, trace in enumerate(streams):
    trace.stats.starttime+= 60*60*24*day
    file_name = trace.stats.starttime.strftime(date_string_2) + '.sac'
    trace.write(directory_2 +'/'+ file_name, format="sac")