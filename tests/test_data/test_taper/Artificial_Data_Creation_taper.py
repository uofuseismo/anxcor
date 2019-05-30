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

# Percentage of taper
tp=0.05

# Directories and Names
directory = "/home/dwells/ancor/ancor/tests/test_data/"

subdir = "test_taper/"

name = "sinewave"+str(trace.stats.delta)+"_"

# Current options are trapezoid, cosine, and hamming:
shape = "cosine"

step = "source.sac"

# Write to file as Source
# Note the Naming Convention: Type of wave, delta t, shape of taper, percentage of taper, source/target, .sac
stream.traces=[trace]
stream.write(directory+subdir+name+shape+"_"+str(tp)+"_"+step, format='sac')

#%%
# Reading from file as Source
step="source.sac"
new_stream = obspy.read(directory+subdir+name+shape+"_"+str(tp)+"_"+step)

# Plot to Verify Source
new_stream.plot()

#%%
# Load the Data to process.py for Target
step="source.sac"
new_stream = obspy.read(directory+subdir+name+shape+"_"+str(tp)+"_"+step)
step = "target.sac"
# tp = 0.05

#%%
# Cosine Taper
if shape == "cosine":

    # tapers is the array at the start
    tapers = np.linspace(-np.pi/2,0,int(tp*len(new_stream[0])))
    tapers = np.cos(tapers)

    # taperf is the array at the end

    taperf = np.linspace(0,np.pi/2,int(tp*len(new_stream[0])))
    taperf = np.cos(taperf)

    # middle is the array in the middle
    middle = np.linspace(1,1, (int(len(new_stream[0]) - len(tapers) - len(taperf))))
    full_taper = np.append(tapers, middle)
    full_taper = np.append(full_taper, taperf)

# Trapezoid Taper
if shape == "trapezoid":
    tapers = np.linspace(0,1,int(tp*len(new_stream[0])))
    taperf = np.linspace(1,0,int(tp*len(new_stream[0])))
    middle = np.linspace(1,1, (int(len(new_stream[0]) - len(tapers) - len(taperf))))
    full_taper = np.append(tapers, middle)
    full_taper = np.append(full_taper, taperf)

# Hamming Tpaer
if shape == "hamming":
    tapers = np.linspace(0, np.pi,int(tp*len(new_stream[0])))
    tapers = 0.53836 - 0.46164 * np.cos(tapers)
    taperf = np.linspace(np.pi, 2*np.pi,int(tp*len(new_stream[0])))
    taperf = 0.53836 - 0.46164*np.cos(taperf)
    middle = np.linspace(1,1, (int(len(new_stream[0]) - len(tapers) - len(taperf))))
    full_taper = np.append(tapers, middle)
    full_taper = np.append(full_taper, taperf)

# Taper the Stream
tapered_data = np.multiply((new_stream[0]), full_taper)

trace = obspy.core.Trace(data=tapered_data)

trace.stats.delta = 0.01
print(trace.stats)

stream.traces = [trace]
# Write to file as Target
stream.write(directory+subdir+name+shape+"_"+str(tp)+"_"+step, format='sac')

#%%
# Reading from file as Target
step="target.sac"
new_stream = obspy.read(directory+subdir+name+shape+"_"+str(tp)+"_"+step)

# Plot to Verify Target
new_stream.plot()