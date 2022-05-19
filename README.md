# actipy

Python package to process Axivity3 (`.cwa`), GENEActiv (`.bin`) and Actigraph (`.gt3x`) files.

## Installation

#### Pip
```bash
pip install actipy
```

#### Conda
```bash
conda install -c oxwear actipy
```

*Note:* Use either Pip or Conda, not both.

## Usage

```python
import actipy

data, info = actipy.read_device("sample.cwa",
                                 lowpass_hz=20,
                                 calibrate_gravity=True,
                                 detect_nonwear=True,
                                 resample_hz=50)

# data [pandas.DataFrame]
#                                 x         y         z          T
# time
# 2014-05-07 13:29:50.430 -0.514335  0.069683  1.674354  20.000000
# 2014-05-07 13:29:50.450 -0.514335  0.069683  1.674354  20.000000
# 2014-05-07 13:29:50.470 -0.089527 -0.805399 -0.593359  20.000000
# 2014-05-07 13:29:50.490 -0.120995 -0.914784 -0.529480  20.000000
# 2014-05-07 13:29:50.510 -0.105261 -0.836652 -0.641269  20.000000
# ...                           ...       ...       ...        ...

# info [dict]
# Filename              : sample.cwa
# Filesize(MB)          : 209
# DeviceID              : 1020
# Device                : Axivity
# ReadOK                : 1
# ReadErrors            : 0
# SampleRate            : 100.0
# Resampled             : 1
# NumTicksBeforeResample: 51391800
# NumTicksAfterResample : 50515693
# DetectNonwear         : 1
# NumNonWearEpisodes    : 0
# NonwearTime(days)     : 0.0
# Calibrated            : 1
# CalibErrorBefore(mg)  : 84.72403883934021
# CalibErrorAfter(mg)   : 3.4564044541239087
# ...

```

See the [documentation](https://actipy.readthedocs.io/en/latest/) for more details.

## License
See [license](LICENSE.md) before using this software.
