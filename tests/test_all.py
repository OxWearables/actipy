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
    assert_dict_equal(info, info_ref, rel=0.01)

    data_ref = read_csv('tests/data/read.csv.gz')
    pd.testing.assert_frame_equal(data, data_ref, rtol=0.01)  # 1% tolerance


def test_lowpass():
    """ Test lowpass filtering at 20 Hz . """

    data, info = read_device()
    data, info_lowpass = P.lowpass(data, info['SampleRate'], cutoff_rate=20, chunksize=10_000)

    info_ref = {
        'LowpassOK': 1,
        'LowpassCutoff(Hz)': 20,
    }
    assert_dict_equal(info_lowpass, info_ref, rel=0.01)

    data_ref = read_csv('tests/data/lowpass.csv.gz')
    pd.testing.assert_frame_equal(data, data_ref, rtol=0.01)  # 1% tolerance


def test_resample():
    """ Test resampling to 25 Hz. """

    data, info = read_device()
    # Use a small chunk size to test chunking
    data, info_resample = P.resample(data, sample_rate=25, chunksize=10_000)

    info_resample_ref = {
        'ResampleRate': 25,
        'NumTicksAfterResample': 267737,
    }
    assert_dict_equal(info_resample, info_resample_ref, rel=0.01)

    data_ref = read_csv('tests/data/resample.csv.gz')
    pd.testing.assert_frame_equal(data, data_ref, rtol=0.01)  # 1% tolerance


def test_calibrate_gravity():
    """ Test calibration. """

    data, info = read_device()
    # Use a bad calibration cube to force calibration
    data, info_calib = P.calibrate_gravity(data, calib_cube=0, calib_min_samples=1, chunksize=10_000)

    info_calib_ref = {
        'CalibErrorBefore(mg)': 33.75431150197983,
        'CalibErrorAfter(mg)': 1.5364194987341762,
        'CalibOK': 1,
        'CalibNumIters': 73,
        'CalibNumSamples': 122,
        'CalibxIntercept': -0.03442875,
        'CalibyIntercept': -0.16496603,
        'CalibzIntercept': 0.29612103,
        'CalibxSlope': 1.0060189,
        'CalibySlope': 1.0023165,
        'CalibzSlope': 1.0265391,
        'CalibxSlopeT': 0.0014661448,
        'CalibySlopeT': 0.009421193,
        'CalibzSlopeT': -0.012653602
    }
    assert_dict_equal(info_calib, info_calib_ref, rel=0.01)

    data_ref = read_csv('tests/data/calib.csv.gz')
    pd.testing.assert_frame_equal(data, data_ref, rtol=0.01)  # 1% tolerance


def test_detect_nonwear():
    """ Test nonwear detection. """

    data, info = read_device()
    # Use a bad patience to force nonwear detection
    data, info_nonwear = P.detect_nonwear(data, patience='1m')

    info_nonwear_ref = {
        'WearTime(days)': 0.1203330787037037,
        'NonwearTime(days)': 0.0008101851851851852,
        'NumNonwearEpisodes': 1
    }
    assert_dict_equal(info_nonwear, info_nonwear_ref, rel=0.01)

    data_ref = read_csv('tests/data/nonwear.csv.gz')
    pd.testing.assert_frame_equal(data, data_ref, rtol=0.01)  # 1% tolerance


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


def read_csv(fpath):
    """ Read a CSV file. """
    return pd.read_csv(
        fpath,
        parse_dates=['time'], index_col='time',
        dtype={
            'x': 'f4',
            'y': 'f4',
            'z': 'f4',
            'temperature': 'f4',
            'light': 'f4'
        }
    )
