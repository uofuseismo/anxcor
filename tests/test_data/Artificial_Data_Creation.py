import obspy
import matplotlib.pyplot as plt
import numpy as np


#%%
#GenerateSourceFile

stream = obspy.core.Stream()

'''Sine Function'''

arry1 = np.linspace(0,50000,50001)
arry2 = np.sin(2*np.pi*arry1/1500)

'''Step Function'''

#arry1 = np.zeros(25000)
#arry2 = [1]*25001
#arry3 = np.append(arry2, arry1)


trace = obspy.core.Trace(data=arry2)

# make sure you adjust the stats for your problem
trace.stats.delta=0.01
print(trace.stats)

stream.traces=[trace]

#writing to file
directory="/home/dwells/ancor/ancor/tests/test_data/"
subdir="test_taper/"
name="sinewave1_"
shape="trapezoid_"
step="source.sac"

stream.traces=[trace]
stream.write(directory+subdir+name+shape+step, format='sac')
#%%
#reading from file
step="source.sac"
new_stream = obspy.read(directory+subdir+name+shape+step)

#Plot to Verify Source
new_stream.plot()

#%%
step="source.sac"
new_stream = obspy.read(directory+subdir+name+shape+step)
step = "target.sac"
tp = 0.05
tapers = np.linspace(0,1,int(tp*len(new_stream[0])))
taperf = np.linspace(1,0,int(tp*len(new_stream[0])))
middle = np.linspace(1,1, (int(len(new_stream[0]) - len(tapers) - len(taperf))))
full_taper = np.append(tapers, middle)
full_taper = np.append(full_taper, taperf)
#Taper the Stream
tapered_data = np.multiply((new_stream[0]), full_taper)

trace = obspy.core.Trace(data=tapered_data)

# make sure you adjust the stats for your problem
trace.stats.delta = 0.01
print(trace.stats)

stream.traces = [trace]

stream.write(directory+subdir+name+shape+step, format='sac')

#plt.plot(full_taper)
#plt.show()

#%%
#reading from file
step="target.sac"
new_stream = obspy.read(directory+subdir+name+shape+step)

#Plot to Verify Target
new_stream.plot()