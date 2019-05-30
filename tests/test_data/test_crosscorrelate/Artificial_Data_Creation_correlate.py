import obspy
import matplotlib.pyplot as plt
import numpy as np


#%%
#GenerateSourceFile
stream = obspy.core.Stream()

'''Create random noise with tau offset'''
tau = 1000

arry1 = np.random.rand(10000)
arry2 = arry1
tau1 = np.random.rand(tau)
tau2 = np.random.rand(tau)
arry1 = np.append(arry1, tau1)
arry1 = arry1 - np.mean(arry1)
arry2 = np.append(tau2, arry2)
arry2 = arry2 - np.mean(arry2)

# make sure you adjust the stats for your problem

trace1 = obspy.core.Trace(data=arry1)
trace1.stats.delta = 0.01
trace2 = obspy.core.Trace(data=arry2)
trace2.stats.delta = 0.01

# Directories and Names
directory = "/home/dwells/ancor/ancor/tests/test_data/"

subdir = "test_crosscorrelate/"

name = "random"+str(trace1.stats.delta)+"_"

offset = str(tau)

step1 = "source"+str(1)+".sac"
step2 = "source"+str(2)+".sac"

# Write to file as Source
# Note the Naming Convention: Type of wave, delta t, offset of taper, percentage of taper, source/target, .sac
stream.traces=[trace1]
stream.write(directory+subdir+name+offset+step1, format='sac')
stream.traces=[trace2]
stream.write(directory+subdir+name+offset+step2, format='sac')


#%%
# Reading from file as Source
step1 = "source"+str(1)+".sac"
new_stream1 = obspy.read(directory+subdir+name+offset+step1)
step2 = "source"+str(2)+".sac"
new_stream2 = obspy.read(directory+subdir+name+offset+step2)
# Plot to Verify Source
new_stream1.plot()
new_stream2.plot()

#%%
# Load the Data to process.py for Target

new_stream1 = obspy.read(directory+subdir+name+offset+step1)
new_stream2 = obspy.read(directory+subdir+name+offset+step2)
stept = "target.sac"



#%%
# Correlate the traces
correlated = np.correlate(new_stream1[0], new_stream2[0], 'same')

ntrace1 = obspy.core.Trace(data=correlated)
ntrace1.stats.delta = 0.01
print(ntrace1.stats)

stream.traces = [ntrace1]
# Write to file as Target
stream.write(directory+subdir+name+offset+stept, format='sac')

#%%
# Reading from file as Target
stept="target.sac"
new_stream = obspy.read(directory+subdir+name+offset+stept)

# Plot to Verify Target
new_stream.plot()