import setuptools
import sys
with open("README.md", "r") as fh:
    long_description = fh.read()

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
SETUP_REQUIRES = ['pytest-runner >= 4.2'] if needs_pytest else []
TESTS_REQUIRE = ['pytest >= 2.7.1']

DISTNAME = 'anxcor'
LICENSE = 'MIT'
AUTHOR = 'Kevin A. Mendoza'
AUTHOR_EMAIL = 'kevin.mendoza@utah.edu'
URL = 'https://github.com/uofuseismo/anxcor'

setuptools.setup(
    name=DISTNAME,
    license = LICENSE,
    version='0.1.0',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description="ANXCOR is a python library for performing seismic ambient noise crosscorrelations",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url=URL,
    packages=setuptools.find_packages(),
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    install_requires = [
    'obspy>=1.1',
    'xarray>=0.14',
    'numpy>=1.17',
    'scipy>=1.0',
    'pandas>=0.25',
    'sparse>=0.8',
    'bottleneck>=1.2',
    'pympler>=0.7'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        "Programming Language :: Python :: 3.6",
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',
    ],
)
print(setuptools.find_packages())