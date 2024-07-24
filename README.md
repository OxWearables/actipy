# actipy

A Python package to process accelerometer data.

Axivity3 and Axivity6 (`.cwa`), Actigraph (`.gt3x`), and GENEActiv (`.bin`) files are supported,
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

```python
import actipy

data, info = actipy.read_device("sample.cwa.gz",  # or "sample.gt3x.gz" or "sample.bin.gz"
                                 lowpass_hz=20,
                                 calibrate_gravity=True,
                                 detect_nonwear=True,
                                 resample_hz=50)
```

Output:
```console
data [pandas.DataFrame]
                                 x         y         z  temperature      light
 time
 2014-05-07 13:29:50.430 -0.513990  0.070390  1.671922    20.000000  78.420235
 2014-05-07 13:29:50.440 -0.233940 -0.586568  0.082067    20.000000  78.420235
 2014-05-07 13:29:50.450 -0.080319 -0.950817 -0.810613    20.000000  78.420235
 2014-05-07 13:29:50.460 -0.067236 -0.975886 -0.865132    20.000000  78.420235
 2014-05-07 13:29:50.470 -0.109636 -0.857004 -0.508666    20.000000  78.420235
 ...                           ...       ...       ...          ...        ...

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
Refer to the [Data Dictionary](data-dictionary.md) for a comprehensive list of outputs.

### Processing a custom CSV file
You can also use the routines in `actipy.processing` to process custom CSV files, or for more fine-grained control:

```python
import actipy.processing as P

data, info_lowpass = P.lowpass(data, 100, 20)
data, info_calib = P.calibrate_gravity(data)
data, info_nonwear = P.detect_nonwear(data)
data, info_resample = P.resample(data, sample_rate)
```

See the [documentation](https://actipy.readthedocs.io/en/latest/) for more.

### Processing from the command line
We also provide a command-line tool to process a device file (currently, only AX3 and AX6 are supported) and output to CSV:
```bash
$ read_cwa sample.cwa.gz -o outputs --lowpass-hz 20 --resample-hz 50 --calibrate-gravity --detect-nonwear
```

Outputs:
  - "outputs/sample/sample.csv.gz"
  - "outputs/sample/sample-Info.json"


## Contributing
If you would like to contribute to this repository, please check out [CONTRIBUTING.md](https://github.com/OxWearables/actipy/blob/main/CONTRIBUTING.md).
We welcome contributions in the form of bug reports, feature requests, and pull requests. 

## License
See [LICENSE.md](LICENSE.md).
