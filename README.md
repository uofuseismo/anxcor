[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## ANCOR: Ambient Noise Cross cORellation
ANCOR is a python library for conducting seismic ambient noise crosscorrelations.



## Acknowledgements

ANCOR would not be possible without the hard and extensive work by Obsy and Obsplus teams and MtPy team. 


MT-Explore also leverages numpy, pandas, scipy, and psutil

## Installation


## Usage

There are four core objects that ancor utilizes: worker_processes, Workers, Databases and WindowManagers.

In Ambient Noise Crosscorrelation, (workflow)

 

## Known Issues
There are a few cosmetic issues that I probably wont fix as they dont impact the usability of MT-Explore. However, if you know how to fix them feel free to submit a pull request.

By overriding some of the key bindings of Matplotlib, a few unexpected plot and map behaviors were introduced. If things mess up just kill the plot window and instantiate the Main object again. 

tick locators are temporarily broken. this shouldn't impact usability however.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Attribution

If you use this package, please cite [mtpy](https://github.com/MTgeophysics/mtpy) and [cartopy](https://scitools.org.uk/cartopy/docs/latest/). Much of the functionality of this package comes from their work. 

At this time I dont ask that you use an official citation of MT-Explore, but if it is helpful in class projects, publications, training, surveys, or any other endeavor, I would appreciate an acknowledgement. Something like [Mendoza, K; Mt-Explore 2019](https://github.com/El-minadero/mt-explore.git) could be appropriate.


