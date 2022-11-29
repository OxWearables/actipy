# actipy

A Python package to process accelerometer data.

Axivity3 (`.cwa`), Actigraph (`.gt3x`), and GENEActiv (`.bin`) files are supported,
as well as custom CSV files.

Axivity3 is the activity tracker watch used in the large-scale
[UK-Biobank accelerometer study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649).

## Getting started

### Prerequisite

- Python 3.8 or greater
    ```console
    $ python --version  # or python3 --version
    ```

- Java 8 (1.8.0) or greater
    ```console
    $ java -version
    ```

### Install

```bash
$ pip install actipy
```

<!-- With Conda:
```bash
$ conda install -c oxwear actipy
``` -->

## Usage

Process an Axivity3 (.cwa) file:
```python
import actipy

data, info = actipy.read_device("sample.cwa.gz",
                                 lowpass_hz=20,
                                 calibrate_gravity=True,
                                 detect_nonwear=True,
                                 resample_hz=50)
```

Output:
```console
data [pandas.DataFrame]
                                 x         y         z  temperature
 time
 2014-05-07 13:29:50.430 -0.513936  0.070043  1.671264    20.000000
 2014-05-07 13:29:50.440 -0.233910 -0.586894  0.081946    20.000000
 2014-05-07 13:29:50.450 -0.080303 -0.951132 -0.810433    20.000000
 2014-05-07 13:29:50.460 -0.067221 -0.976200 -0.864934    20.000000
 2014-05-07 13:29:50.470 -0.109617 -0.857322 -0.508587    20.000000
 ...                           ...       ...       ...          ...

info [dict]
 Filename                 : data/sample.cwa.gz
 Filesize(MB)             : 69.4
 Device                   : Axivity
 DeviceID                 : 13110
 ReadErrors               : 0
 SampleRate               : 100.0
 ReadOK                   : 1
 StartTime                : 2014-05-07 13:29:50
 EndTime                  : 2014-05-13 09:50:33
 NumTicks                 : 51391800
 WearTime(days)           : 5.847725231481482
 NumInterrupts            : 1
 ResampleRate             : 100.0
 NumTicksAfterResample    : 25262174
 LowpassOK                : 1
 LowpassCutoff(Hz)        : 20.0
 CalibErrorBefore(mg)     : 82.95806873592024
 CalibErrorAfter(mg)      : 4.434966371604519
 CalibOK                  : 1
 NonwearTime(days)        : 0.0
 NumNonwearEpisodes       : 0
 ...

```

If you have a CSV file that you want to process, you can also use the data processing routines from `actipy.processing`:

```python
import actipy.processing as P

data, info_lowpass = P.lowpass(data, 100, 20)
data, info_calib = P.calibrate_gravity(data)
data, info_nonwear = P.detect_nonwear(data)
data, info_resample = P.resample(data, sample_rate)
```

See the [documentation](https://actipy.readthedocs.io/en/latest/) for more.

## License
See [LICENSE.md](LICENSE.md).
