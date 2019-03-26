
from obspy import read
directory_1 = 'obspy_dates'

streams = read()

for day, trace in enumerate(streams):
    trace.stats.starttime+= 60*60*24*day
    trace.write(directory_1 +'/'+ str(day) + '.sac', format="sac")
