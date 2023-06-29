import io
from functools import lru_cache
from pytest import approx
import pandas as pd
import joblib

import actipy
from actipy import processing as P



def test_read_device():
    """ Test reading a device. """

    data, info = read_device()

    info_ref = {
        "Filename": 'tests/data/tiny-sample.cwa.gz',
        "Filesize(MB)": 1.6,
        "Device": 'Axivity',
        "DeviceID": 43923,
        "ReadErrors": 0,
        "SampleRate": 100.0,
        "ReadOK": 1,
        "StartTime": '2023-06-08 12:21:04',
        "EndTime": '2023-06-08 15:19:33',
        "NumTicks": 1021800,
        "WearTime(days)": 0.1211432638888889,
        "NumInterrupts": 1
    }
    assert_dict_equal(info, info_ref)

    data_ref = pd.read_csv(io.StringIO(
        'time,x,y,z,temperature,light\n'
        '2023-06-08 15:19:33.898,  0.375000, -0.765625, -0.218750,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.909,  0.296875, -0.890625, -0.218750,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.919,  0.421875, -0.812500, -0.171875,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.929,  0.375000, -0.890625, -0.250000,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.939,  0.359375, -1.203125,  0.125000,  19.85,  41.568882'
    ), parse_dates=['time'], index_col='time', dtype='f4')

    pd.testing.assert_frame_equal(data.tail(), data_ref)


def test_lowpass():
    """ Test lowpass filtering. """

    data, info = read_device()
    data, info_lowpass = P.lowpass(data, info['SampleRate'], cutoff_rate=20)

    info_lowpass_ref = {
        'LowpassOK': 1, 
        'LowpassCutoff(Hz)': 20
    }
    assert_dict_equal(info_lowpass, info_lowpass_ref)

    data_ref = pd.read_csv(io.StringIO(
        'time,x,y,z,temperature,light\n'
        '2023-06-08 15:19:33.898,  0.379264, -0.783015, -0.253166,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.909,  0.366240, -0.792810, -0.263782,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.919,  0.368996, -0.837900, -0.225322,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.929,  0.369180, -0.977334, -0.090964,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.939,  0.359296, -1.203209,  0.125051,  19.85,  41.568882'
    ), parse_dates=['time'], index_col='time', dtype='f4')

    pd.testing.assert_frame_equal(data.tail(), data_ref)


def test_resample():
    """ Test resampling. """

    data, info = read_device()
    data, info_resample = P.resample(data, sample_rate=info['SampleRate'])

    info_resample_ref = {
        'ResampleRate': 100.0, 
        'NumTicksAfterResample': 1070944
    }
    assert_dict_equal(info_resample, info_resample_ref)

    data_ref = pd.read_csv(io.StringIO(
        'time,x,y,z,temperature,light\n'
        '2023-06-08 15:19:33.900,  0.375000, -0.765625, -0.218750,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.910,  0.296875, -0.890625, -0.218750,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.920,  0.421875, -0.812500, -0.171875,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.930,  0.375000, -0.890625, -0.250000,  19.85,  41.568882\n'
        '2023-06-08 15:19:33.940,  0.359375, -1.203125,  0.125000,  19.85,  41.568882'
    ), parse_dates=['time'], index_col='time', dtype='f4')

    pd.testing.assert_frame_equal(data.tail(), data_ref)


def test_calibrate_gravity():
    """ Test calibration. """

    data, info = read_device()
    # Use a bad calibration cube to force calibration
    data, info_calib = P.calibrate_gravity(data, calib_cube=0, calib_min_samples=1)

    info_calib_ref = {
        'CalibErrorBefore(mg)': 31.963828951120377,
        'CalibErrorAfter(mg)': 2.108112210407853,
        'CalibOK': 1,
        'CalibNumIters': 60,
        'CalibNumSamples': 266,
        'CalibxIntercept': 0.00942089,
        'CalibyIntercept': -0.11357996,
        'CalibzIntercept': 0.22651094,
        'CalibxSlope': 1.0049648,
        'CalibySlope': 1.0017107,
        'CalibzSlope': 1.0250089,
        'CalibxSlopeT': -0.00085395266,
        'CalibySlopeT': 0.0067764097,
        'CalibzSlopeT': -0.009050926
    }
    assert_dict_equal(info_calib, info_calib_ref)


def test_detect_nonwear():
    """ Test nonwear detection. """

    data, info = read_device()
    # Use a bad patience to force nonwear detection
    data, info_nonwear = P.detect_nonwear(data, patience='1m')

    info_nonwear_ref = {
        'WearTime(days)': 0.12022313657407407,
        'NonwearTime(days)': 0.0009201273148148148,
        'NumNonwearEpisodes': 1
    }
    assert_dict_equal(info_nonwear, info_nonwear_ref)


def test_joblib():
    """ Test joblib. """

    results = joblib.Parallel(n_jobs=2)(
        joblib.delayed(read_device)(
            f'tests/data/tiny-sample{i}.cwa.gz', 
            lowpass_hz=20,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=50,
        ) 
        for i in (1, 2)
    )


def assert_dict_equal(dict1, dict2, **kwargs):
    """ Assert that two dictionaries are equal. """
    assert dict1.keys() == dict2.keys()
    for key, value in dict2.items():
        assert dict1[key] == approx(value, **kwargs)


@lru_cache
def read_device(
    fpath="tests/data/tiny-sample.cwa.gz",
    lowpass_hz=None,
    calibrate_gravity=False,
    detect_nonwear=False,
    resample_hz=None,
):
    """ Cached version of read_device, with default no processing. """
    return actipy.read_device(
        fpath,
        lowpass_hz=lowpass_hz,
        calibrate_gravity=calibrate_gravity,
        detect_nonwear=detect_nonwear,
        resample_hz=resample_hz,
    )
