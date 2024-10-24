Usage
=====

Prerequisite 
------------

- Python 3.8 or greater
    .. code-block:: console

        $ python --version  # or python3 --version

- Java 8 (1.8.0) or greater
    .. code-block:: console

        $ java -version


Installation
------------

.. code-block:: console

    $ pip install actipy


Basic example
-------------

Process an Axivity file (.cwa):

.. code-block:: python

    import actipy

    data, info = actipy.read_device("sample.cwa.gz",
                                    lowpass_hz=20,
                                    calibrate_gravity=True,
                                    detect_nonwear=True,
                                    resample_hz=50)

Output:

.. code-block:: none

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


Custom data format
..................
If you have data in another format (e.g. CSV) and still want to leverage the
data processing routines (low-pass filtering, calibration, non-wear detection
and resampling), you can do it by converting your data to a Pandas dataframe and
using :code:`actipy.process`.


.. code-block:: python

    import actipy

    data, info = actipy.process(data, sample_rate=100,
                                lowpass_hz=20,
                                calibrate_gravity=True,
                                detect_nonwear=True,
                                resample_hz=50)

See the :doc:`API reference <api>`.


Fine-tune processing
....................
You can access the individual processing routines at
:code:`actipy.processing.*` for more fine-grained control.

.. code-block:: python

    import actipy.processing as P

    data, info_calib = P.calibrate_gravity(data, calib_cube=0.2)
    data, info_nonwear = P.detect_nonwear(data, patience='2h')

See the :doc:`API reference <api>`.
