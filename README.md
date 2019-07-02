[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/uofuseismo/anxcor.png?branch=master)](https://travis-ci.org/uofuseismo/anxcor)
![PyPI](https://img.shields.io/pypi/v/anxcor.svg?color=blue&style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dm/anxcor.svg?style=plastic)


## ANXCOR: Ambient Noise X (cross) Correlation



**ANXCOR** is a python library for performing seismic ambient noise crosscorrelations.


ANXCOR's object oriented design allows it to leverage common datascience packages like [pandas](https://pandas.pydata.org/), [numpy](https://www.numpy.org/), and [scipy](https://www.scipy.org/), as well as the popular seismology package [obspy](https://github.com/obspy/obspy) to streamline its computation. Furthermore, it leverages both [xarray](http://xarray.pydata.org) and [dask](https://dask.org) to achieve embarassingly parallel execution. Use of these popular packages makes working with ANXCOR intuitive, concise, and extensible without deep domain experience in compiled languages.

ANXCOR is not intended to replace existing seismic ambient noise processing codes. Instead, our goal is to provide a framework for rapid prototyping of new processing routines, and small-batch production of seismic ambient noise correlation functions. ANXCOR is also designed with readability in mind; it should be immediately obvious from code alone what functions functions do and how they work without comments (we do, however, aim to provide extensive documentation of the code base). 


## Acknowledgements
ANXCOR's development was inspired by Dr. Fan-Chi Lin's work in Ambient Noise Seismic Tomography. Many of the routines implemented here were written after careful consultation with him and his Graduate Student work group (However, none of their code was copied or directly translated into anxcor). Ph.D. Candidate Elizableth Berg especially was integral to the success of this project.

## Attribution

## Installation
Anxcor was designed to seamlessly integrate with anaconda and pip. Eventually, we'd like to get it up on the conda-forge channel, but for now we reccomend a two part installation process. First, in a new anaconda environment, install the following:
```
conda install numpy scipy pandas xarray obspy bottleneck
```
Anxcor can work without optional dask or [obsplus](https://github.com/niosh-mining/obsplus) dependencies, but in case you want parallel execution and/or a great waveform repo manager, we suggest installing them via:
```
conda install dask
pip install obsplus
```
Finally, you can install Anxcor, at the moment via pip:
```
pip install anxcor
```
## Basic Usage
We're gonna use obsplus to help us with loading waveform data, so go ahead install obsplus as indicated above.

Next, using your favorite editor, import the following:
```python
from obsplus.bank import WaveBank
from obspy.core import Stream, Trace
from anxcor.core import Anxcor, AnxcorDatabase
```
Anxcor needs to be provided AnxcorDatabase objects in order to access your data. So first, lets create one by encapsulating an obsplus wavebank. Our object needs to implement a get_waveforms() and a get_stations() method. We'll go ahead and instantiate this class with a directory link pointing to our seismic data on file. If there is a ton of data in this directory, it might take awhile to instantiate properly. Note that Anxcor expects you, the user, to remove the response from your data as desired. 
```python
class WavebankWrapper(AnxcorDatabase):

    def __init__(self, directory):
        super().__init__()
        self.bank = WaveBank(directory)
        import warnings
        warnings.filterwarnings("ignore")

    def get_waveforms(self, **kwargs):
        stream =  self.bank.get_waveforms(**kwargs)
        traces = []
        for trace in stream:
            data = trace.data[:-1]
            header = {'delta':np.floor(trace.stats.delta*1000)/1000.0,
                      'station': trace.stats.station,
                      'starttime':trace.stats.starttime,
                      'channel': trace.stats.channel,
                      'network': trace.stats.network}
            traces.append(Trace(data,header=header))
        return Stream(traces=traces)

    def get_stations(self):
        df = self.bank.get_uptime_df()

        def create_seed(row):
            network = row['network']
            station = row['station']
            return network + '.' + station

        df['seed'] = df.apply(lambda row: create_seed(row), axis=1)
        unique_stations = df['seed'].unique().tolist()
        return unique_stations
        
source_dir = '..path/to/your/seismic/data..'
bank = WaveBank(source_dir)
```
Now we can make our Anxcor object. We need to provide it a window_length in seconds. say.. 15 minutes.
```python
anxcor = Anxcor(window_length=15*60.0)
```
Anxcor applies this window to a list of starttimes provided by us, and can generate that list of windows if we provide it  start times, stop times, and window overlap percent. All times used by ancor are assumed to be UTCDateTime.timestamp seconds. See [UTCDateTime](https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html) for more details. 

We'll get starttimes to correlate over an hour after the given starttime with 25% window overlap.
```python
# replace starttime_stamp with your specific starttime_stamp
starttime_stamp = 0.0
times = anxcor.get_starttimes(starttime_stamp, starttime_stamp + 4* 15*60, 0.25)
```
Now lets add the dataset and process the times! Each dataset provided to Anxcor requires an accompanying name to keep track of metadata. we'll call this name 'test' for now
```python
anxcor.add_dataset(bank, 'test')
result = anxcor.process(times)
```
If you want to parallelize this process using dask, provide a dask client:
```python
from dask.distributed import Client
client = Client()
result=anxcor.process(times,dask_client=client)
```
There you have it. Within a few seconds, anxcor will give you back a correlated, stacked, and combined [XArray DataSet](http://xarray.pydata.org/en/stable/data-structures.html#dataset). The Dataset has dimensions of channel, station_pair, and time. Attached metadata can be accessed via the ```.attrs``` property, and contains all relevant metadata extracted from the original crosscorrelations. 

if you'd rather have the data as an obspy stream, anxcor provides a conversion option:

```python
 streams = anxcor.xarray_to_obspy(result)
```


## Known Issues
## Planned Enhancements

- Component Rotation along azimuth and backazimuth
- FTAN and beamforming routines
- Custom crosscorrelation preprocessing functions
- GPU implementations of crosscorrelation

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Contributors
- PhD Student Kevin A. Mendoza was the primary developer, and is responsible for the original architecture of the project.
- PhD Student Daniel Wells contributed a number of test suites.



