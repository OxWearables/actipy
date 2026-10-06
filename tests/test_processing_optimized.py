import numpy as np
import pandas as pd
import pytest

from actipy import processing as P
from actipy import reader as R


def _reference_nonwear_segments(data, patience, window, stdtol):
    stationary = (
        data["x"].ffill().resample(window, origin="start").std().lt(stdtol)
        & data["y"].ffill().resample(window, origin="start").std().lt(stdtol)
        & data["z"].ffill().resample(window, origin="start").std().lt(stdtol)
    )
    edges = stationary != stationary.shift(1)
    edges.iloc[0] = True
    segment_ids = edges.cumsum()
    stationary_ids = segment_ids[stationary]
    lengths = (
        stationary_ids.groupby(stationary_ids)
        .agg(
            start_time=lambda values: values.index[0],
            length=lambda values: values.index[-1] - values.index[0],
        )
        .set_index("start_time")
        .squeeze(axis=1)
        .astype("timedelta64[ns]")
    )
    return lengths[lengths > pd.Timedelta(patience)]


def test_increasing_mask_carries_maximum_across_chunks():
    index = pd.to_datetime(
        [
            "2024-01-01 00:00:05",
            "2024-01-01 00:00:06",
            "2024-01-01 00:00:01",
            "2024-01-01 00:00:02",
            "2024-01-01 00:00:07",
        ]
    )

    result = P._increasing_mask(index, chunksize=2)

    np.testing.assert_array_equal(result, [True, True, False, False, True])
    assert P._count_nonincreasing(index, chunksize=2) == 1


def test_quality_control_matches_pandas_reference_with_missing_data():
    index = pd.date_range("2024-01-01", periods=24 * 6, freq="10min")
    values = np.ones((len(index), 4), dtype=np.float32)
    values[13:19, :3] = np.nan
    values[30:36, :] = np.nan
    data = pd.DataFrame(values, index=index, columns=["x", "y", "z", "light"])
    data.index.name = "time"

    _, info = P.quality_control(data, sample_rate=0.1)

    differences = data.dropna(subset=["x", "y", "z"]).index.to_series().diff()
    coverage = data.notna().any(axis=1).groupby(data.index.hour).mean()
    assert info["WearTime(days)"] == (
        differences[differences < pd.Timedelta("1s")].sum().total_seconds()
        / 86_400
    )
    assert info["NumInterrupts"] == (differences > pd.Timedelta("1s")).sum()
    assert info["Covers24hOK"] == int(
        len(coverage) == 24 and coverage.min() >= 0.01
    )


def test_quality_control_rejects_each_missing_acceleration_axis():
    index = pd.date_range("2024-01-01", periods=3, freq="500ms")
    for missing in ("x", "y", "z"):
        data = pd.DataFrame(
            {
                column: np.ones(len(index), dtype=np.float32)
                for column in ("x", "y", "z")
                if column != missing
            },
            index=index,
        )

        with pytest.raises(KeyError, match=missing):
            P.quality_control(data, sample_rate=2)


def test_window_statistics_match_pandas_across_chunk_boundaries():
    rng = np.random.default_rng(2026)
    index = pd.date_range(
        "2024-01-01 00:00:00.123", periods=61, freq="700ms"
    )
    values = rng.normal(size=(len(index), 4)).astype(np.float32)
    values[[0, 4, 5, 12, 13, 14, 40], 0] = np.nan
    values[[3, 4, 10, 11, 25], 1] = np.nan
    data = pd.DataFrame(
        values, index=index, columns=["x", "y", "z", "temperature"]
    )

    means, deviations, width_ns = P._window_statistics(
        data,
        tuple(data.columns),
        "3s",
        forward_fill=True,
        chunksize=5,
    )
    expected = data.ffill().resample("3s", origin="start")

    assert width_ns == pd.Timedelta("3s").value
    np.testing.assert_allclose(
        means, expected.mean().to_numpy(), rtol=1e-6, atol=1e-7, equal_nan=True
    )
    np.testing.assert_allclose(
        deviations,
        expected.std().to_numpy(),
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", ["Float32", "Float64"])
def test_window_statistics_converts_nullable_columns_in_chunks(
    monkeypatch,
    dtype,
):
    index = pd.date_range("2024-01-01", periods=20, freq="1s")
    values = np.arange(40, dtype=np.float64).reshape(20, 2)
    values[[0, 4, 9, 15], 0] = np.nan
    values[[2, 7, 12, 19], 1] = np.nan
    data = pd.DataFrame(
        {
            column: pd.array(values[:, axis], dtype=dtype)
            for axis, column in enumerate(("x", "y"))
        },
        index=index,
    )
    expected = data.ffill().resample("3s", origin="start")
    expected_means = expected.mean().to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    expected_deviations = expected.std().to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    original_to_numpy = pd.Series.to_numpy
    conversions = []

    def tracked_to_numpy(series, *args, **kwargs):
        if series.name in data.columns:
            conversions.append((len(series), kwargs))
        return original_to_numpy(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, "to_numpy", tracked_to_numpy)

    means, deviations, _ = P._window_statistics(
        data,
        tuple(data.columns),
        "3s",
        forward_fill=True,
        chunksize=4,
    )

    assert conversions
    assert max(length for length, _ in conversions) <= 4
    assert all(options["dtype"] == np.float64 for _, options in conversions)
    assert all(np.isnan(options["na_value"]) for _, options in conversions)
    np.testing.assert_allclose(means, expected_means, equal_nan=True)
    np.testing.assert_allclose(
        deviations,
        expected_deviations,
        equal_nan=True,
    )


def test_timestamp_arithmetic_supports_all_datetime_resolutions():
    ticks_per_second = {
        "s": 1,
        "ms": 1_000,
        "us": 1_000_000,
        "ns": 1_000_000_000,
    }
    for unit, ticks in ticks_per_second.items():
        index = pd.DatetimeIndex(
            (np.arange(21, dtype=np.int64) * 2 * ticks).astype(
                f"datetime64[{unit}]"
            )
        )
        data = pd.DataFrame(
            np.ones((len(index), 3), dtype=np.float32),
            index=index,
            columns=["x", "y", "z"],
        )

        assert P._has_uniform_rate(index, sample_rate=0.5, chunksize=4)
        rebuilt = P._index_from_ns(
            P._timestamps_to_ns(index.asi8, P._timestamp_scale(index)),
            index,
        )
        assert rebuilt.dtype == index.dtype
        pd.testing.assert_index_equal(rebuilt, index)
        _, info = P.quality_control(data, sample_rate=0.5)
        assert info["NumInterrupts"] == len(index) - 1

        means, deviations, _ = P._window_statistics(
            data, ("x", "y", "z"), "5s", chunksize=4
        )
        expected = data.resample("5s", origin="start")
        np.testing.assert_allclose(means, expected.mean().to_numpy())
        np.testing.assert_allclose(
            deviations, expected.std().to_numpy(), equal_nan=True
        )


def test_find_nonwear_segments_matches_pandas_reference():
    rng = np.random.default_rng(7)
    index = pd.date_range("2024-01-01", periods=10 * 60 * 2, freq="500ms")
    values = np.zeros((len(index), 3), dtype=np.float32)
    values[:120] = rng.normal(scale=0.1, size=(120, 3))
    values[370:380] = np.nan
    data = pd.DataFrame(values, index=index, columns=["x", "y", "z"])

    result = P.find_nonwear_segments(
        data, patience="2min", window="10s", stdtol=0.015
    )
    expected = _reference_nonwear_segments(
        data, patience="2min", window="10s", stdtol=0.015
    )

    pd.testing.assert_series_equal(result, expected)


def test_timezone_aware_resample_and_nonwear_preserve_timezone():
    for timezone in ("Etc/GMT+5", "Europe/London"):
        index = pd.date_range(
            "2024-03-31 00:59:30",
            periods=80,
            freq="2s",
            tz=timezone,
        )
        data = pd.DataFrame(
            np.zeros((len(index), 3), dtype=np.float32),
            index=index,
            columns=["x", "y", "z"],
        )

        resampled, _ = P.resample(data, sample_rate=1, chunksize=7)
        assert resampled.index.tz == data.index.tz
        assert resampled.index[0] == data.index[0]

        result = P.find_nonwear_segments(
            data, patience="20s", window="10s", stdtol=0.015
        )
        expected = _reference_nonwear_segments(
            data, patience="20s", window="10s", stdtol=0.015
        )
        pd.testing.assert_series_equal(result, expected)


def test_resample_result_does_not_share_column_storage():
    index = pd.Timestamp("2024-01-01") + pd.to_timedelta([0, 0.6, 1], unit="s")
    data = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "z": [7.0, 8.0, 9.0]},
        index=index,
    )

    result, _ = P.resample(data, sample_rate=2, chunksize=2)
    result.iloc[0, 0] = 100.0

    assert data.iloc[0, 0] == 1.0


def test_resample_promotes_integer_and_boolean_columns_only_for_gaps():
    index = pd.to_datetime(
        ["2024-01-01 00:00:00", "2024-01-01 00:00:05"]
    )
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "z": [1.0, 2.0],
            "count": [1, 2],
            "active": [True, False],
        },
        index=index,
    )
    target = pd.date_range(index[0], index[-1], periods=6)
    expected = data.reindex(
        target,
        method="nearest",
        tolerance=pd.Timedelta("1s"),
        limit=1,
    )

    result, _ = P.resample(data, sample_rate=1, chunksize=2)

    pd.testing.assert_frame_equal(result, expected)


def test_calibration_statistics_do_not_truncate_integer_inputs():
    index = pd.date_range("2024-01-01", periods=40, freq="1s")
    values = np.tile([[1, 1, 1], [2, 2, 2]], (20, 1))
    integer_data = pd.DataFrame(
        values, index=index, columns=["x", "y", "z"]
    )
    float_data = integer_data.astype(np.float64)

    with pytest.warns(UserWarning, match="Insufficient stationary samples"):
        _, integer_info = P.calibrate_gravity(
            integer_data,
            calib_min_samples=100,
            window="10s",
        )
    with pytest.warns(UserWarning, match="Insufficient stationary samples"):
        _, float_info = P.calibrate_gravity(
            float_data,
            calib_min_samples=100,
            window="10s",
        )

    assert integer_info["CalibNumSamples"] == float_info["CalibNumSamples"] == 0


def test_calibration_uses_only_first_100000_stationary_points(monkeypatch):
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    means = np.resize(directions, (P._MAX_CALIBRATION_SAMPLES + 1, 3))
    means[-1] = 100.0
    deviations = np.zeros_like(means)

    def stationary_statistics(*args, **kwargs):
        return means, deviations, pd.Timedelta("10s").value

    monkeypatch.setattr(P, "_window_statistics", stationary_statistics)
    data = pd.DataFrame(
        [[0.0, 0.0, 1.0]],
        columns=["x", "y", "z"],
        index=pd.date_range("2024-01-01", periods=1, freq="1s"),
    )

    result, info = P.calibrate_gravity(data)

    assert result is data
    assert info["CalibNumSamples"] == 100_000
    assert info["CalibErrorBefore(mg)"] == 0
    assert info["CalibNumIters"] == 0
    assert info["CalibOK"] == 1


def test_xyz_fallback_assignment_preserves_float32_storage():
    result = pd.DataFrame(
        np.zeros((3, 3), dtype=np.float32), columns=["x", "y", "z"]
    )
    arrays = tuple(
        result[column].to_numpy(copy=False) for column in result.columns
    )
    values = np.array(
        [
            [0.123456789, 1.23456789, 2.3456789],
            [3.456789, 4.56789, 5.6789],
        ],
        dtype=np.float64,
    )

    P._write_xyz(result, arrays, (0, 1, 2), 0, values)

    assert all(dtype == np.dtype(np.float32) for dtype in result.dtypes)
    np.testing.assert_array_equal(
        result.iloc[:2].to_numpy(), values.astype(np.float32)
    )


def test_inplace_lowpass_matches_nonmutating_path_across_chunks():
    rng = np.random.default_rng(13)
    index = pd.date_range("2024-01-01", periods=500, freq="10ms")
    data = pd.DataFrame(
        rng.normal(size=(len(index), 4)).astype(np.float32),
        index=index,
        columns=["x", "y", "z", "temperature"],
    )

    expected, _ = P.lowpass(data, 100, chunksize=37)
    inplace_input = data.copy(deep=True)
    result, _ = P.lowpass(inplace_input, 100, chunksize=37, _inplace=True)

    assert result is inplace_input
    pd.testing.assert_frame_equal(result, expected)
    assert not result[["x", "y", "z"]].equals(data[["x", "y", "z"]])


def test_inplace_lowpass_updates_nullable_float_columns():
    rng = np.random.default_rng(24)
    index = pd.date_range("2024-01-01", periods=500, freq="10ms")
    data = pd.DataFrame(
        {
            column: pd.array(rng.normal(size=len(index)), dtype="Float32")
            for column in ("x", "y", "z")
        },
        index=index,
    )
    data.iloc[25:28, :] = pd.NA

    expected, _ = P.lowpass(data, 100, chunksize=37)
    inplace_input = data.copy(deep=True)
    result, _ = P.lowpass(
        inplace_input, 100, chunksize=37, _inplace=True
    )

    assert result is inplace_input
    pd.testing.assert_frame_equal(result, expected)
    assert not result.equals(data)
    direct_result, _ = P.lowpass(
        data.astype(np.float32), 100, chunksize=37
    )
    np.testing.assert_allclose(
        result.to_numpy(dtype=np.float64, na_value=np.nan),
        direct_result.to_numpy(dtype=np.float64, na_value=np.nan),
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


def test_inplace_calibration_matches_nonmutating_path_across_chunks():
    rng = np.random.default_rng(7)
    targets = rng.normal(size=(120, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    raw = (
        targets - np.array([0.04, -0.03, 0.02])
    ) / np.array([1.08, 0.94, 1.05])
    values = np.repeat(raw, 10, axis=0)
    index = pd.date_range("2024-01-01", periods=len(values), freq="1s")
    data = pd.DataFrame(values, index=index, columns=["x", "y", "z"])

    expected, expected_info = P.calibrate_gravity(
        data,
        calib_min_samples=50,
        window="10s",
        chunksize=37,
    )
    inplace_input = data.copy(deep=True)
    result, info = P.calibrate_gravity(
        inplace_input,
        calib_min_samples=50,
        window="10s",
        chunksize=37,
        _inplace=True,
    )

    assert expected_info["CalibOK"] == 1
    assert expected_info["CalibNumIters"] > 0
    assert "CalibxIntercept" not in expected_info
    assert result is inplace_input
    assert info == expected_info
    pd.testing.assert_frame_equal(result, expected)
    assert not result.equals(data)


@pytest.mark.parametrize(
    ("xyz_dtype", "temperature_dtype", "dtype"),
    [
        (np.float32, np.float32, np.float32),
        (np.float64, np.float64, np.float64),
        (np.float32, np.float64, np.float64),
    ],
)
def test_calibration_uses_caller_precision(
    xyz_dtype,
    temperature_dtype,
    dtype,
):
    rng = np.random.default_rng(81)
    targets = rng.normal(size=(120, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    temperature = np.linspace(18.0, 28.0, len(targets))
    intercept = np.array([0.04, -0.03, 0.02])
    slope = np.array([1.08, 0.94, 1.05])
    temperature_slope = np.array([0.001, -0.002, 0.0015])
    raw = (
        targets
        - intercept
        - temperature[:, None] * temperature_slope
    ) / slope
    values = np.repeat(raw, 10, axis=0).astype(xyz_dtype)
    temperatures = np.repeat(temperature, 10).astype(temperature_dtype)
    if xyz_dtype == np.float64:
        values[0, 0] = 1e39
    index = pd.date_range("2024-01-01", periods=len(values), freq="1s")
    data = pd.DataFrame(values, index=index, columns=["x", "y", "z"])
    data["temperature"] = temperatures

    result, info = P.calibrate_gravity(
        data,
        calib_min_samples=50,
        window="10s",
        return_coeffs=True,
        chunksize=37,
    )

    assert info["CalibOK"] == 1
    expected = values.astype(dtype)
    expected_temperature = temperatures.astype(dtype)
    for axis, name in enumerate(("x", "y", "z")):
        np.multiply(
            expected[:, axis],
            dtype(info[f"Calib{name}Slope"]),
            out=expected[:, axis],
        )
        np.add(
            expected[:, axis],
            dtype(info[f"Calib{name}Intercept"]),
            out=expected[:, axis],
        )
        temperature_term = np.multiply(
            expected_temperature,
            dtype(info[f"Calib{name}SlopeT"]),
        )
        np.add(
            expected[:, axis],
            temperature_term,
            out=expected[:, axis],
        )

    assert all(result[column].dtype == xyz_dtype for column in ("x", "y", "z"))
    assert all(
        np.asarray(info[f"Calib{name}{coefficient}"]).dtype
        == np.dtype(dtype)
        for name in ("x", "y", "z")
        for coefficient in ("Intercept", "Slope", "SlopeT")
    )
    np.testing.assert_array_equal(
        result[["x", "y", "z"]].to_numpy(),
        expected.astype(xyz_dtype),
    )
    if xyz_dtype == np.float64:
        assert np.isfinite(result.iloc[0]["x"])


@pytest.mark.parametrize("inplace", [False, True])
def test_nullable_calibration_converts_inputs_in_chunks(
    monkeypatch,
    inplace,
):
    rng = np.random.default_rng(83)
    targets = rng.normal(size=(120, 3))
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    temperature = np.linspace(18.0, 28.0, len(targets))
    raw = (
        targets
        - np.array([0.04, -0.03, 0.02])
        - temperature[:, None] * np.array([0.001, -0.002, 0.0015])
    ) / np.array([1.08, 0.94, 1.05])
    values = np.repeat(raw, 10, axis=0).astype(np.float32)
    temperatures = np.repeat(temperature, 10).astype(np.float32)
    index = pd.date_range("2024-01-01", periods=len(values), freq="1s")
    data = pd.DataFrame(
        {
            column: pd.array(values[:, axis], dtype="Float32")
            for axis, column in enumerate(("x", "y", "z"))
        },
        index=index,
    )
    data["temperature"] = pd.array(temperatures, dtype="Float32")
    data.iloc[:10] = pd.NA

    original_prepare = P._prepare_xyz_output
    original_to_numpy = pd.Series.to_numpy
    application_started = False
    application_conversions = []

    def tracked_prepare(*args, **kwargs):
        nonlocal application_started
        application_started = True
        return original_prepare(*args, **kwargs)

    def tracked_to_numpy(series, *args, **kwargs):
        if application_started and series.name in (*P._XYZ_COLUMNS, "temperature"):
            application_conversions.append((series.name, len(series)))
        return original_to_numpy(series, *args, **kwargs)

    monkeypatch.setattr(P, "_prepare_xyz_output", tracked_prepare)
    monkeypatch.setattr(pd.Series, "to_numpy", tracked_to_numpy)

    result, info = P.calibrate_gravity(
        data,
        calib_cube=0,
        calib_min_samples=50,
        window="10s",
        chunksize=37,
        _inplace=inplace,
    )

    assert info["CalibOK"] == 1
    assert all(str(result[column].dtype) == "Float32" for column in data.columns)
    assert result.iloc[:10].isna().all(axis=None)
    assert application_conversions
    assert {name for name, _ in application_conversions} == {
        "x",
        "y",
        "z",
        "temperature",
    }
    assert max(length for _, length in application_conversions) <= 37


def test_inplace_nonwear_matches_nonmutating_path_when_episode_detected():
    rng = np.random.default_rng(42)
    moving_before = rng.normal(scale=0.2, size=(30, 3))
    stationary = np.tile([0.0, 0.0, 1.0], (50, 1))
    moving_after = rng.normal(scale=0.2, size=(30, 3))
    values = np.vstack((moving_before, stationary, moving_after))
    index = pd.date_range("2024-01-01", periods=len(values), freq="1s")
    data = pd.DataFrame(values, index=index, columns=["x", "y", "z"])

    expected, expected_info = P.flag_nonwear(
        data, patience="20s", window="5s"
    )
    inplace_input = data.copy(deep=True)
    result, info = P.flag_nonwear(
        inplace_input,
        patience="20s",
        window="5s",
        _inplace=True,
    )

    assert expected_info["NumNonwearEpisodes"] > 0
    assert expected.isna().all(axis=1).any()
    assert result is inplace_input
    assert info == expected_info
    pd.testing.assert_frame_equal(result, expected)
    assert not result.equals(data)


def test_filtered_reader_output_owns_compact_storage(monkeypatch):
    index = pd.date_range("2024-01-01", periods=100, freq="1s")
    arrays = {
        column: np.arange(100, dtype=np.float32)
        for column in ("x", "y", "z", "light")
    }
    full = pd.DataFrame(arrays, index=index, copy=False)

    def fake_read_device(input_file, verbose):
        return full, {"SampleRate": 1.0, "ReadErrors": 0}

    monkeypatch.setattr(R, "_read_device", fake_read_device)
    result, _ = R.read_device(
        "recording.cwa",
        start_time=index[40],
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=None,
        verbose=False,
    )

    for column in result.columns:
        assert not np.shares_memory(
            result[column].to_numpy(copy=False),
            full[column].to_numpy(copy=False),
        )
    assert not np.shares_memory(result.index.asi8, full.index.asi8)


def test_flag_nonwear_noop_returns_independent_frame():
    rng = np.random.default_rng(31)
    index = pd.date_range("2024-01-01", periods=100, freq="1s")
    data = pd.DataFrame(
        rng.normal(scale=0.1, size=(len(index), 3)).astype(np.float32),
        index=index,
        columns=["x", "y", "z"],
    )

    result, info = P.flag_nonwear(data, patience="90m", window="10s")
    result.iloc[0, 0] = 100.0

    assert result is not data
    assert info["NumNonwearEpisodes"] == 0
    assert data.iloc[0, 0] != 100.0
