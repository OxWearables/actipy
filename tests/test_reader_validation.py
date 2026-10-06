import pandas as pd
import pytest

from actipy import reader

INVALID_RESAMPLE_FREQUENCIES = [
    pytest.param("50", id="numeric-string"),
    pytest.param("invalid", id="unsupported-mode"),
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
]


@pytest.mark.parametrize("resample_hz", INVALID_RESAMPLE_FREQUENCIES)
def test_read_device_rejects_invalid_resample_frequency_before_read(
    monkeypatch, resample_hz
):
    def fail_if_called(*args, **kwargs):
        pytest.fail("_read_device must not run for an invalid resample_hz")

    monkeypatch.setattr(reader, "_read_device", fail_if_called)

    with pytest.raises(ValueError, match="resample_hz must be"):
        reader.read_device("unused.cwa", resample_hz=resample_hz)


@pytest.mark.parametrize("resample_hz", INVALID_RESAMPLE_FREQUENCIES)
def test_process_rejects_invalid_resample_frequency(resample_hz):
    with pytest.raises(ValueError, match="resample_hz must be"):
        reader.process(pd.DataFrame(), sample_rate=100, resample_hz=resample_hz)


@pytest.mark.parametrize(
    ("resample_hz", "expected_rate"),
    [
        pytest.param(1, 1, id="integer-one"),
        pytest.param(1.0, 1.0, id="float-one"),
        pytest.param(True, 100, id="uniform-boolean"),
    ],
)
def test_process_distinguishes_one_hertz_from_uniform(
    monkeypatch, resample_hz, expected_rate
):
    observed_rates = []

    def record_resample(data, sample_rate, **kwargs):
        observed_rates.append(sample_rate)
        return data, {}

    monkeypatch.setattr(reader.P, "resample", record_resample)

    reader.process(
        pd.DataFrame(),
        sample_rate=100,
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=resample_hz,
        verbose=False,
    )

    assert observed_rates == [expected_rate]


def test_read_device_forwards_processing_options(monkeypatch):
    data = pd.DataFrame(
        {"x": [0.0], "y": [0.0], "z": [1.0]},
        index=pd.date_range("2024-01-01", periods=1, freq="1s"),
    )
    sample_rates = []
    calibration_calls = []
    nonwear_calls = []

    def fake_read_device(input_file, verbose):
        return data, {"SampleRate": 100.0, "ReadErrors": 2}

    def fake_quality_control(frame, sample_rate):
        sample_rates.append(sample_rate)
        return frame, {"ReadErrors": 3}

    def fake_calibrate_gravity(frame, **kwargs):
        calibration_calls.append(kwargs)
        return frame, {}

    def fake_flag_nonwear(frame, **kwargs):
        nonwear_calls.append(kwargs)
        return frame, {}

    monkeypatch.setattr(reader, "_read_device", fake_read_device)
    monkeypatch.setattr(reader.P, "quality_control", fake_quality_control)
    monkeypatch.setattr(reader.P, "calibrate_gravity", fake_calibrate_gravity)
    monkeypatch.setattr(reader.P, "flag_nonwear", fake_flag_nonwear)

    result, info = reader.read_device(
        "unused.cwa",
        lowpass_hz=None,
        calibrate_gravity_kwargs={
            "calib_cube": 0.4,
            "calib_min_samples": 20,
            "window": "5s",
            "stdtol": 0.02,
            "stdtol_min": 0.001,
            "chunksize": 500,
        },
        flag_nonwear_kwargs={
            "patience": "60m",
            "window": "5s",
            "stdtol": 0.02,
        },
        resample_hz=None,
        verbose=False,
    )

    assert result is data
    assert sample_rates == [100.0]
    assert info["ReadErrors"] == 5
    assert calibration_calls == [{
        "calib_cube": 0.4,
        "calib_min_samples": 20,
        "window": "5s",
        "stdtol": 0.02,
        "stdtol_min": 0.001,
        "chunksize": 500,
        "_inplace": True,
    }]
    assert nonwear_calls == [{
        "patience": "60m",
        "window": "5s",
        "stdtol": 0.02,
        "_inplace": True,
    }]

    calibration_calls.clear()
    reader.read_device(
        "unused.cwa",
        lowpass_hz=None,
        calibrate_gravity_kwargs={"return_coeffs": True},
        detect_nonwear=False,
        resample_hz=None,
        verbose=False,
    )
    assert calibration_calls == [{
        "return_coeffs": True,
        "_inplace": True,
    }]


def test_process_forwards_processing_options(monkeypatch):
    data = pd.DataFrame()
    calibration_calls = []
    nonwear_calls = []

    def fake_calibrate_gravity(frame, **kwargs):
        calibration_calls.append(kwargs)
        return frame, {}

    def fake_flag_nonwear(frame, **kwargs):
        nonwear_calls.append(kwargs)
        return frame, {}

    monkeypatch.setattr(reader.P, "calibrate_gravity", fake_calibrate_gravity)
    monkeypatch.setattr(reader.P, "flag_nonwear", fake_flag_nonwear)

    result, _ = reader.process(
        data,
        sample_rate=100,
        lowpass_hz=None,
        calibrate_gravity_kwargs={
            "calib_cube": 0.4,
            "return_coeffs": False,
        },
        flag_nonwear_kwargs={"patience": "60m"},
        resample_hz=None,
        verbose=False,
    )

    assert result is data
    assert calibration_calls == [{
        "calib_cube": 0.4,
        "return_coeffs": False,
    }]
    assert nonwear_calls == [{"patience": "60m"}]
