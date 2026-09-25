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
