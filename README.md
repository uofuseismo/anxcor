[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/uofuseismo/anxcor.png?branch=master)](https://travis-ci.org/uofuseismo/anxcor)
![PyPI](https://img.shields.io/pypi/v/anxcor.svg?color=blue&style=plastic)
![PyPI - Downloads](https://img.shields.io/pypi/dm/anxcor.svg?style=plastic)


## ANXCOR: Ambient Noise X (cross) Correlation



**ANXCOR** is a python library for performing seismic ambient noise crosscorrelations.

ANXCOR is not intended to replace existing seismic ambient noise processing codes.
Instead, our goal is to provide a framework for rapid prototyping of new processing routines,
and small-batch production of seismic ambient noise correlation functions.
ANXCOR is designed with readability and explicit documentation in mind; all algorithms are well documented, grounded in
research, and written following most of the practices outlined in the [Clean Code Handbook by Robert C. Martin](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882).



ANXCOR integrates seamlessly into the current python datascience stack, leverage common datascience packages 
like [pandas](http://pandas.pydata.org), [NumPy](http://www.numpy.org), and [SciPy](http://www.scipy.org), 
as well as the popular seismology package [ObsPy](https://github.com/obspy/obspy/wiki). 
Furthermore, it leverages both [xarray](http://xarray.pydata.org/en/stable/) and [dask](http://dask.org)
to achieve embarassingly parallel execution. Use of these popular packages makes working with ANXCOR intuitive,
concise, and extensible without deep domain experience in compiled languages.

## Why ANXCOR ?

Crosscorrelation functions derived from Ambient Seismic Noise have broad applicability across geophysics,
from monitoring volcanic activity [[Lobkis and Weaver, 2001](https://scholar.google.com/scholar_lookup?hl=en&volume=110&publication_year=2001&pages=3011&journal=J.+Acoust.+Soc.+Am.&issue=6&author=O.+I.+Lobkis&author=R.+L.+Weaver&title=On+the+emergence+of+the+Green%27s+function+in+the+correlations+of+a+diffuse+field)],
to informing seismic vunerablility assessments [[Prieto, GA and Beroza GC 2008](https://scholar.google.com/scholar?cluster=4969353848435547473&hl=en&as_sdt=0,45)]. The advent of cheap geophones like the [Magseis-Fairfield Zland 3C](https://www.passcal.nmt.edu/content/fairfieldnodal-zland-3-channel-sensor)
have research departments and USGS offices flush with data. These data range from the gigabyte to multi-terabyte scale, and can be a pain to keep organized.

Typical crosscorrelation workflows include read/write operations at every step of signal processing. Even with downsampling and Signal-Noise Ratio data curation,
directories can easily become crowded with thousands of files. The problem is exacerbated by attempts to shove every bit of
meaningful metadata into file names, creating a logistical nightmare for all but those intimately familiar with the datasets.
Additionaly, the optimized code used to analyse these data often use workflows with a mish-mash of bash, tcsh, sac, fortran, and c scripts;
compiled codes with succinctly (but unreadable) defined variables, and few modularized functions. 

Solutions like [MS Noise](http://www.msnoise.org/), are excellent, well-cited, and stable. At the risk of introducing yet another
standard, we present ANXCOR as a possible solution for the non-programmers among us. 

We approach mitigating the above problems by 
using [xarray datastructures](http://xarray.pydata.org/en/stable/data-structures.html) for vectorized computations and metadata
persistence. Use of [dask](http://dask.org), [NumPy](http://www.numpy.org), and [SciPy](http://www.scipy.org), allows us to 
create an embarassingly parallel compute graph. This allows computation to take place almost entirelly in RAM, eliminating redundant file proliferation,
while allowing the user to select specific outputs to save to file.

Anxcor also provides abstract classes useful for the user who would like to implement their own crosscorrelation methods or
preprocessing steps. Because we defer all parallelization to be handled by [dask](http://dask.org), we can make such 
interfaces concise, readable, and highly modular. 

Because ANXCOR is aimed specifically at [python's datascience ecosystem](https://scipy-lectures.org/intro/intro.html), we believe
we can provide an approach to seismic ambient noise crosscorrelation that provides great utilty to users already familiar 
with these packages.

## Documentation

learn more about ANXCOR at the [wiki](https://github.com/uofuseismo/anxcor/wiki).


## Acknowledgements
Kevin A. Mendoza served as chief architect and programmer of ANXCOR. Development of the project was inspired by Dr. Fan-Chi Lin's work in Ambient Noise Seismic Tomography. Many of the routines implemented here were written after careful consultation with him and his Graduate Student work group (However, none of their code was copied or directly translated into anxcor). Ph.D. Candidate Elizableth Berg especially was integral to the success of this project.

## Attribution

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


## LICENSE

Copyright 2019 Kevin A Mendoza

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.


