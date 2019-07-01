[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.org/uofuseismo/anxcor.png?branch=master)](https://travis-ci.org/uofuseismo/anxcor)
[![Coverage Status](https://coveralls.io/repos/github/uofuseismo/anxcor/badge.png?branch=master)](https://coveralls.io/github/uofuseismo/anxcor?branch=master)

## ANXCOR: Ambient Noise X (cross) Correlation

**ANXCOR** is a python library for performing seismic ambient noise crosscorrelations.


ANXCOR's object oriented design allows it to leverage common datascience packages like pandas, numpy, and scipy, as well as the popular seismology package Obspy to streamline its computation. Furthermore, it leverages both xarray and dask to achieve embarassingly parallel execution. Use of these popular packages makes working with ANXCOR intuitive, concise, and extensible without deep domain experience in compiled languages.

ANXCOR is not intended to replace existing seismic ambient noise processing codes. Instead, our goal is to provide a framework for rapid prototyping of new processing routines, and small-batch production of seismic ambient noise correlation functions. ANXCOR is also designed with readability in mind; it should be immediately obvious from code alone what functions functions do and how they work without comments (we do, however, aim to provide extensive documentation of the code base). 


## Acknowledgements
ANXCOR's development was inspired by Dr. Fan-Chi Lin's work in Ambient Noise Seismic Tomography. Many of the routines implemented here were written after careful consultation with him and his Graduate Student work group. PhD Candidate Elizableth Berg especially was integral to the success of this project.

## Attribution

## Installation

## Basic Usage
 

## Known Issues

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Contributors
- PhD Student Kevin A. Mendoza was the primary developer, and is responsible for the original architecture of the project.
- PhD Student Daniel Wells contributed a number of test suites.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Attribution


