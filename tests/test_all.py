from functools import lru_cache, reduce
import operator
import struct
import zipfile

import numpy as np
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
        "DataSpan(days)": 0.12395172453703704,
        "NumInterrupts": 1,
        "Covers24hOK": 0
    }
    assert_dict_equal(info, info_ref, rel=0.01)

    data_ref = read_csv('tests/data/read.csv.gz')
    pd.testing.assert_frame_equal(data, data_ref, rtol=0.01)  # 1% tolerance


def _write_gt3x(path, metadata, raw_xyz):
    timestamp = 1_700_000_000
    payload = struct.pack("<hhh", *raw_xyz)
    header = struct.pack("<BBIH", 0x1E, 26, timestamp, len(payload))
    checksum = (~reduce(operator.xor, header + payload)) & 0xFF

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.txt", metadata)
        archive.writestr("log.bin", header + payload + bytes([checksum]))

    return timestamp


def _read_gt3x(path):
    return actipy.read_device(
        str(path),
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=None,
        verbose=False,
    )


def test_read_actigraph_uses_metadata_scale(tmp_path):
    """GT3X metadata scale supports new device serial families."""

    gt3x_file = tmp_path / "sample.gt3x"
    timestamp = _write_gt3x(
        gt3x_file,
        "\ufeffSerial Number: STM2E24245655\n"
        "Sample Rate: 30\n"
        "Start Date: 638355968000000000\n"
        "Acceleration Min: -8.0\n"
        "Acceleration Max: 8.0\n"
        "Acceleration Scale: 256.0\n",
        (256, -256, 128),
    )
    data, info = _read_gt3x(gt3x_file)

    assert info["DeviceID"] == "STM2E24245655"
    assert info["ReadErrors"] == 0
    assert data.index[0] == pd.Timestamp(timestamp, unit="s")
    np.testing.assert_allclose(data.iloc[0], [1.0, -1.0, 0.5])


def test_read_actigraph_bom_preserves_legacy_scale_fallback(tmp_path):
    """A BOM must not hide the serial used by legacy scale fallback."""

    gt3x_file = tmp_path / "bom-legacy.gt3x"
    _write_gt3x(
        gt3x_file,
        "\ufeffSerial Number: NEO123\n"
        "Sample Rate: 30\n"
        "Start Date: 638355968000000000\n"
        "Acceleration Min: -6.0\n"
        "Acceleration Max: 6.0\n",
        (341, -341, 0),
    )
    data, info = _read_gt3x(gt3x_file)

    assert info["DeviceID"] == "NEO123"
    assert info["ReadErrors"] == 0
    np.testing.assert_allclose(data.iloc[0], [1.0, -1.0, 0.0])


def test_read_actigraph_replaces_implausible_metadata_scale(tmp_path):
    """A corrupt positive scale must not silently produce extreme g values."""

    gt3x_file = tmp_path / "invalid-scale.gt3x"
    _write_gt3x(
        gt3x_file,
        "Serial Number: NEO123\n"
        "Sample Rate: 30\n"
        "Start Date: 638355968000000000\n"
        "Acceleration Min: -6.0\n"
        "Acceleration Max: 6.0\n"
        "Acceleration Scale: 1.0\n",
        (341, -341, 0),
    )
    data, info = _read_gt3x(gt3x_file)

    assert info["ReadErrors"] == 0
    np.testing.assert_allclose(data.iloc[0], [1.0, -1.0, 0.0])


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
    data, info_nonwear = P.flag_nonwear(data, patience='1m')

    info_nonwear_ref = {
        'NonwearTime(days)': 0.0008101851851851852,
        'NumNonwearEpisodes': 1,
        'WearTime(days)': 0.1203330787037037,
        'NumInterrupts': 2,
        'Covers24hOK': 0,
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
    data = pd.read_csv(
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
    data.index = data.index.astype('datetime64[ns]')
    return data
