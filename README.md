# Pywear

Python tool to process Axivity (`.cwa`), GENEActiv (`.bin`) and Actigraph (`.gt3x`) wearables data.

## Usage

```python
import pywear

data, info = pywear.read_device("/path/to/data/sample.cwa",
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
# Filename              : data/sample.cwa
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

## Installation

Poor man's installation steps in Linux:

```bash
# Clone repository
git clone https://github.com/activityMonitoring/pywear.git /path/to/pywear

# Export the repo path to a PYWEARPATH system variable, required by Pywear
echo export PYWEARPATH="/path/to/pywear" >> ~/.bashrc

# Compile java files
javac /path/to/pywear/pywear/*.java

# Make repo path visible to Python
echo export PYTHONPATH="$PYWEARPATH:$PYTHONPATH" >> ~/.bashrc
```

### Dependencies
An anaconda installation should cover most of the dependencies. One special package required is [`JPype`](https://jpype.readthedocs.io/en/devel/install.html) which allows interfacing with Java (in which the core parsing code is written).