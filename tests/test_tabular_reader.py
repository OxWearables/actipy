import numpy as np
import pandas as pd
import pytest

from actipy import reader


def _frame(n=6000, sample_rate=100):
    rng = np.random.default_rng(0)
    index = pd.date_range(
        "2024-01-01", periods=n, freq=pd.Timedelta(seconds=1 / sample_rate),
        name="time",
    )
    return pd.DataFrame(
        {
            "x": rng.normal(0, 0.01, n),
            "y": rng.normal(0, 0.01, n),
            "z": rng.normal(1, 0.01, n),
        },
        index=index,
    ).astype("f4")


def _read(path, **kwargs):
    return reader.read_device(
        str(path),
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=None,
        verbose=False,
        **kwargs,
    )


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz"])
def test_read_device_reads_csv(tmp_path, suffix):
    expected = _frame()
    path = tmp_path / f"sample{suffix}"
    expected.to_csv(path)

    data, info = _read(path)

    pd.testing.assert_frame_equal(data, expected, check_freq=False)
    assert info["Device"] == "unknown (.csv)"
    assert info["SampleRate"] == 100.0
    assert info["NumTicks"] == len(expected)
    assert info["ReadErrors"] == 0


def test_read_device_reads_csv_with_custom_columns(tmp_path):
    expected = _frame()
    path = tmp_path / "sample.csv"
    expected.rename_axis("timestamp").rename(
        columns={"x": "ax", "y": "ay", "z": "az"}
    ).assign(extra=1).to_csv(path)

    data, _ = _read(path, csv_txyz="timestamp,ax,ay,az")

    pd.testing.assert_frame_equal(data, expected, check_freq=False)


def test_read_device_rejects_malformed_csv_txyz(tmp_path):
    path = tmp_path / "sample.csv"
    _frame().to_csv(path)

    with pytest.raises(ValueError, match="csv_txyz"):
        _read(path, csv_txyz="time,x,y")


@pytest.mark.parametrize("time_as_column", [False, True])
def test_read_device_reads_pickle(tmp_path, time_as_column):
    expected = _frame(sample_rate=50)
    path = tmp_path / "sample.pkl"
    (expected.reset_index() if time_as_column else expected).to_pickle(path)

    data, info = _read(path)

    pd.testing.assert_frame_equal(data, expected, check_freq=False)
    assert info["Device"] == "unknown (.pkl)"
    assert info["SampleRate"] == 50.0


def test_read_device_uses_given_sample_rate(tmp_path):
    path = tmp_path / "sample.pkl"
    _frame().to_pickle(path)

    _, info = _read(path, sample_rate=25)

    assert info["SampleRate"] == 25.0


def test_read_device_runs_pipeline_on_csv(tmp_path):
    path = tmp_path / "sample.csv"
    _frame().to_csv(path)

    data, info = reader.read_device(
        str(path), calibrate_gravity=False, resample_hz=50, verbose=False,
    )

    assert info["ResampleRate"] == 50
    assert len(data) == pytest.approx(3000, abs=1)


def test_read_device_counts_duplicate_timestamps_as_read_errors(tmp_path):
    frame = _frame()
    path = tmp_path / "sample.pkl"
    pd.concat([frame, frame.iloc[:10]]).sort_index().to_pickle(path)

    data, info = _read(path)

    # ReadErrors is reported in seconds' worth of bad samples, rounded up.
    assert info["ReadErrors"] == 1
    assert len(data) == len(frame)
    assert data.index.is_unique
