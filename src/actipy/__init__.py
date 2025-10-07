"""
Actipy: A Python package for processing accelerometer data.

This package provides tools to read and process data from wearable accelerometer
devices including Axivity3/6 (.cwa), Actigraph (.gt3x), GENEActiv (.bin), and
Matrix (.bin) files.

Main Functions
--------------
read_device : Read and process accelerometer device file
process : Process pandas.DataFrame of acceleration time-series

Modules
-------
reader : Device file reading and high-level processing
processing : Signal processing functions (filtering, calibration, resampling, etc.)
matrix_reader : Matrix device-specific binary file reader

Examples
--------
Basic usage with default processing:

>>> import actipy
>>> data, info = actipy.read_device("sample.cwa.gz")

With custom processing options:

>>> data, info = actipy.read_device(
...     "sample.cwa.gz",
...     lowpass_hz=20,
...     calibrate_gravity=True,
...     detect_nonwear=True,
...     resample_hz=50
... )

See Also
--------
For detailed documentation, visit: https://actipy.readthedocs.io/
"""

name = "actipy"
__author__ = "Shing Chan, Aiden Doherty"
__email__ = "shing.chan@ndph.ox.ac.uk, aiden.doherty@ndph.ox.ac.uk"
__maintainer__ = "Shing Chan"
__maintainer_email__ = "shing.chan@ndph.ox.ac.uk"
__license__ = "See LICENSE.md"

from actipy.reader import read_device, process

from . import _version
__version__ = _version.get_versions()['version']
