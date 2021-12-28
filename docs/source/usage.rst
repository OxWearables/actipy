Usage
=====

Installation
------------

Pip:

.. code-block:: console

    $ pip install actipy


Conda:

.. code-block:: console

    $ conda install -c oxwear actipy


Basic example
-------------

Process an Axivity file (.cwa):

.. code-block:: python

    import actipy

    data, info = actipy.read_device("sample.cwa",
                                    lowpass_hz=20,
                                    calibrate_gravity=True,
                                    detect_nonwear=True,
                                    resample_hz=50)

    # OUTPUT:

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


Custom data format
..................
If you have data in another format (e.g. CSV) and still want to leverage the
data processing routines (low-pass filtering, calibration, non-wear detection
and resampling), you can do it by converting your data to a pandas dataframe and
using :code:`actipy.process`. See the :doc:`API reference <api>`.


Fine-tune processing
....................
You can access the individual processing functions at
:code:`actipy.processing.*` for more fine-grained control.
See the :doc:`API reference <api>`.
