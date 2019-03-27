import numpy as np
import obspy
stream = obspy.core.Stream()
arry1  = np.zeros((10000))
trace1 = obspy.core.Trace(data=arry1)

# make sure you adjust the stats for your problem
trace1.stats

stream.traces=[trace1]
#writing to file
stream.write('test.sac',format='sac')
#reading from file
new_stream = obspy.read('test.sac')