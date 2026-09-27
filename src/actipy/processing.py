"""
Signal processing functions for accelerometer data.

This module provides a suite of signal processing operations for cleaning,
calibrating, and analyzing accelerometer time-series data. All functions
operate on pandas DataFrames with DateTimeIndex and return both processed
data and metadata dictionaries.

Main Processing Functions
-------------------------
quality_control : Basic data quality checks and statistics
lowpass : Butterworth lowpass filtering
calibrate_gravity : Gravity-based calibration (van Hees et al. 2014)
flag_nonwear : Detect and flag non-wear periods
resample : Nearest-neighbor resampling to uniform frequency

Utility Functions
-----------------
find_nonwear_segments : Identify non-wear periods without flagging data
butterfilt : Butterworth filter implementation
chunker : Generator for processing data in time-based chunks

Memory Efficiency
-----------------
Functions that process large datasets use bounded-size chunks and preallocated
outputs to limit temporary allocations. Resampling returns independent in-memory
storage, so its peak memory includes both the input and complete output frames.

Notes
-----
All functions preserve the input DataFrame structure and return tuples of
(processed_data, info_dict) where info_dict contains processing metadata.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import scipy.signal as signal
import statsmodels.api as sm
from numpy.typing import NDArray

__all__ = ['quality_control', 'lowpass', 'calibrate_gravity', 'flag_nonwear', 'find_nonwear_segments', 'resample']

Info = Dict[str, Any]
Array = NDArray[Any]

_NS_PER_SECOND = 1_000_000_000
_DEFAULT_CHUNKSIZE = 1_000_000
_XYZ_COLUMNS = ('x', 'y', 'z')
_TIME_UNIT_TO_NS = {
    's': _NS_PER_SECOND,
    'ms': 1_000_000,
    'us': 1_000,
    'ns': 1,
}


def _timestamp_scale(index: pd.DatetimeIndex) -> int:
    """Return the number of nanoseconds represented by one index tick."""

    unit = cast(str, getattr(index, 'unit', 'ns'))
    try:
        return _TIME_UNIT_TO_NS[unit]
    except KeyError as error:
        raise ValueError(f"Unsupported datetime index unit: {unit}") from error


def _timestamp_to_ns(value: int, scale: int) -> int:
    """Convert one timestamp integer to nanoseconds without overflowing."""

    if scale == 1:
        return value
    limits: Any = np.iinfo(np.int64)
    lower = limits.min // scale + 1
    upper = limits.max // scale
    if value < lower or value > upper:
        raise OverflowError("Datetime index cannot be represented in nanoseconds")
    return value * scale


def _timestamps_to_ns(
    values: NDArray[np.int64],
    scale: int,
) -> NDArray[np.int64]:
    """Convert timestamp integers to nanoseconds with bounded allocation."""

    if scale == 1 or len(values) == 0:
        return values
    _timestamp_to_ns(int(values.min()), scale)
    _timestamp_to_ns(int(values.max()), scale)
    return values * np.int64(scale)


def _index_from_ns(
    values: NDArray[np.int64],
    template: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Build an index from epoch nanoseconds while preserving its resolution."""

    unit = cast(str, getattr(template, 'unit', 'ns'))
    scale = _timestamp_scale(template)
    ticks = values
    if scale != 1 and np.all(values % scale == 0):
        ticks = values // scale
    else:
        unit = 'ns'
    result = pd.DatetimeIndex(
        ticks.astype(f'datetime64[{unit}]', copy=False),
        name=template.name,
    )
    if template.tz is not None:
        result = result.tz_localize('UTC').tz_convert(template.tz)
    return result


def _count_nonincreasing(index: pd.DatetimeIndex, chunksize: int = _DEFAULT_CHUNKSIZE) -> int:
    """Count non-increasing adjacent timestamps without a full-size diff."""

    times = index.asi8
    count = 0
    previous: Optional[int] = None
    for start in range(0, len(times), chunksize):
        values = times[start:start + chunksize]
        if previous is not None and values[0] <= previous:
            count += 1
        if len(values) > 1:
            count += int(np.count_nonzero(values[1:] <= values[:-1]))
        previous = int(values[-1])
    return count


def _has_uniform_rate(
    index: pd.DatetimeIndex,
    sample_rate: float,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> bool:
    """Check a fixed sample interval without allocating an index-sized diff."""

    if len(index) < 3:
        return False
    expected_ns = int(round(_NS_PER_SECOND / sample_rate))
    if not np.isclose(expected_ns / _NS_PER_SECOND, 1 / sample_rate):
        return False

    times = index.asi8
    scale = _timestamp_scale(index)
    previous = _timestamp_to_ns(int(times[0]), scale)
    for start in range(1, len(times), chunksize):
        values = _timestamps_to_ns(times[start:start + chunksize], scale)
        if int(values[0]) - previous != expected_ns:
            return False
        if len(values) > 1 and np.any(np.diff(values) != expected_ns):
            return False
        previous = int(values[-1])
    return True


def _increasing_mask(index: pd.DatetimeIndex, chunksize: int = _DEFAULT_CHUNKSIZE) -> NDArray[np.bool_]:
    """Select strict record-high timestamps, matching the former cummax filter."""

    times = index.asi8
    keep: NDArray[np.bool_] = np.empty(len(times), dtype=bool)
    previous_max: Optional[int] = None
    for start in range(0, len(times), chunksize):
        values = times[start:start + chunksize]
        stop = start + len(values)
        cumulative = np.maximum.accumulate(values)
        if previous_max is not None:
            np.maximum(cumulative, previous_max, out=cumulative)
        chunk_keep = keep[start:stop]
        if previous_max is None:
            chunk_keep[0] = True
        else:
            chunk_keep[0] = values[0] > previous_max
        if len(values) > 1:
            chunk_keep[1:] = values[1:] > cumulative[:-1]
        previous_max = int(cumulative[-1])
    return keep


def _validity_summary(
    data: pd.DataFrame,
    required_columns: Optional[Tuple[str, ...]] = None,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> Tuple[float, int, int]:
    """Calculate wear time, interruptions, and hourly coverage in chunks."""

    all_columns = tuple(cast(str, column) for column in data.columns)
    required = set(required_columns or all_columns)
    missing = required.difference(all_columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise KeyError(f"Missing required columns: {missing_columns}")
    column_arrays = tuple(
        (column, data[column].to_numpy(copy=False)) for column in all_columns
    )
    total_by_hour: NDArray[np.int64] = np.zeros(24, dtype=np.int64)
    valid_by_hour: NDArray[np.int64] = np.zeros(24, dtype=np.int64)
    total_time_ns = 0
    num_interrupts = 0
    previous_valid_time: Optional[int] = None
    timestamp_scale = _timestamp_scale(data.index)

    for start in range(0, len(data), chunksize):
        stop = min(start + chunksize, len(data))
        size = stop - start
        valid_required: NDArray[np.bool_] = np.ones(size, dtype=bool)
        valid_any: NDArray[np.bool_] = np.zeros(size, dtype=bool)

        for column, values in column_arrays:
            valid = pd.notna(values[start:stop])
            valid_any |= valid
            if column in required:
                valid_required &= valid

        times = _timestamps_to_ns(
            data.index.asi8[start:stop], timestamp_scale
        )
        valid_times = times[valid_required]
        if len(valid_times):
            if previous_valid_time is not None:
                difference = int(valid_times[0]) - previous_valid_time
                if difference < _NS_PER_SECOND:
                    total_time_ns += difference
                elif difference > _NS_PER_SECOND:
                    num_interrupts += 1
            differences = np.diff(valid_times)
            total_time_ns += int(differences[differences < _NS_PER_SECOND].sum())
            num_interrupts += int(np.count_nonzero(differences > _NS_PER_SECOND))
            previous_valid_time = int(valid_times[-1])

        hours = data.index[start:stop].hour
        total_by_hour += np.bincount(hours, minlength=24)
        valid_by_hour += np.bincount(hours[valid_any], minlength=24)

    present = total_by_hour > 0
    coverage_ok = bool(
        np.count_nonzero(present) == 24
        and np.all(valid_by_hour[present] / total_by_hour[present] >= 0.01)
    )
    return total_time_ns / _NS_PER_SECOND, num_interrupts, int(coverage_ok)


def _forward_fill(values: Array, previous: float) -> Tuple[Array, float]:
    """Forward-fill one array chunk while carrying state across chunks."""

    missing = np.isnan(values)
    if not missing.any():
        return values, float(values[-1]) if len(values) else previous

    filled = values.copy()
    valid_positions = np.arange(len(filled))
    valid_positions[missing] = -1
    np.maximum.accumulate(valid_positions, out=valid_positions)
    fillable = missing & (valid_positions >= 0)
    filled[fillable] = filled[valid_positions[fillable]]
    leading = valid_positions < 0
    if not np.isnan(previous):
        filled[leading] = previous
    if len(filled) and not np.isnan(filled[-1]):
        previous = float(filled[-1])
    return filled, previous


def _prepare_xyz_output(
    data: pd.DataFrame,
    inplace: bool,
    calculation_dtype: np.dtype[Any],
) -> Tuple[pd.DataFrame, Tuple[Array, ...], Optional[Tuple[int, ...]]]:
    """Prepare an XYZ output frame and its fastest safe assignment path."""

    result = data if inplace else data.copy(deep=True)
    for column in _XYZ_COLUMNS:
        values = result[column].to_numpy(copy=False)
        if not np.issubdtype(values.dtype, np.floating):
            result[column] = result[column].astype(calculation_dtype)

    arrays = tuple(
        result[column].to_numpy(copy=False) for column in _XYZ_COLUMNS
    )
    storage = tuple(result[column].values for column in _XYZ_COLUMNS)
    positions = None
    if not all(
        isinstance(backing, np.ndarray)
        and values.flags.writeable
        and np.shares_memory(values, backing)
        for values, backing in zip(arrays, storage)
    ):
        positions = tuple(
            cast(int, result.columns.get_loc(column))
            for column in _XYZ_COLUMNS
        )
    return result, arrays, positions


def _xyz_calculation_dtype(data: pd.DataFrame) -> np.dtype[Any]:
    """Choose a floating dtype that can represent every acceleration axis."""

    dtypes = []
    for column in _XYZ_COLUMNS:
        values = data[column].to_numpy(copy=False)
        if not pd.api.types.is_numeric_dtype(data[column].dtype):
            raise TypeError(f"Column {column} must be numeric")
        dtypes.append(values.dtype)
    common = np.result_type(*dtypes, np.float32)
    if not np.issubdtype(common, np.floating):
        return np.dtype(np.float64)
    return np.dtype(common)


def _missing_output_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return a NumPy dtype that can retain values plus a missing marker."""

    if np.issubdtype(dtype, np.bool_):
        return np.dtype(object)
    if np.issubdtype(dtype, np.integer):
        return np.dtype(np.float64)
    if dtype.kind in 'fcMmO':
        return dtype
    return np.dtype(object)


def _missing_value(dtype: np.dtype[Any]) -> Any:
    """Return the missing marker appropriate for an output dtype."""

    if np.issubdtype(dtype, np.datetime64):
        return np.datetime64('NaT')
    if np.issubdtype(dtype, np.timedelta64):
        return np.timedelta64('NaT')
    return np.nan


def _write_xyz(
    result: pd.DataFrame,
    arrays: Tuple[Array, ...],
    positions: Optional[Tuple[int, ...]],
    start: int,
    values: Array,
) -> None:
    """Write one XYZ chunk through NumPy or pandas copy-on-write storage."""

    stop = start + len(values)
    if positions is None:
        for output, column_values in zip(arrays, values.T):
            output[start:stop] = column_values
    else:
        for output, position, column_values in zip(
            arrays, positions, values.T
        ):
            result.iloc[start:stop, position] = column_values.astype(
                output.dtype, copy=False
            )


def _window_statistics(
    data: pd.DataFrame,
    columns: Tuple[str, ...],
    window: str,
    forward_fill: bool = False,
    chunksize: int = _DEFAULT_CHUNKSIZE,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Return per-window means and sample deviations using bounded memory."""

    width_ns = int(pd.Timedelta(window).value)
    if width_ns <= 0:
        raise ValueError("window must be positive")
    if len(data) == 0:
        empty: NDArray[np.float64] = np.empty(
            (0, len(columns)), dtype=np.float64
        )
        return empty, empty.copy(), width_ns
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be monotonically increasing")

    times = data.index.asi8
    timestamp_scale = _timestamp_scale(data.index)
    origin_ns = _timestamp_to_ns(int(times[0]), timestamp_scale)
    final_ns = _timestamp_to_ns(int(times[-1]), timestamp_scale)
    num_windows = int((final_ns - origin_ns) // width_ns) + 1
    counts: NDArray[np.int64] = np.zeros(
        (len(columns), num_windows), dtype=np.int64
    )
    sums: NDArray[np.float64] = np.zeros(
        (len(columns), num_windows), dtype=np.float64
    )
    sum_squares: NDArray[np.float64] = np.zeros(
        (len(columns), num_windows), dtype=np.float64
    )
    previous: NDArray[np.float64] = np.full(
        len(columns), np.nan, dtype=np.float64
    )
    column_arrays = tuple(
        data[column].to_numpy(copy=False) for column in columns
    )

    for start in range(0, len(data), chunksize):
        stop = min(start + chunksize, len(data))
        times_ns = _timestamps_to_ns(
            times[start:stop], timestamp_scale
        )
        bins: Array = ((times_ns - origin_ns) // width_ns).astype(
            np.intp, copy=False
        )
        for column_number, column_values in enumerate(column_arrays):
            values = column_values[start:stop]
            if forward_fill:
                values, previous[column_number] = _forward_fill(
                    values, previous[column_number]
                )
            valid = ~np.isnan(values)
            if not valid.any():
                continue
            selected_bins = bins[valid]
            selected = values[valid].astype(np.float64, copy=False)
            first_bin = int(selected_bins[0])
            last_bin = int(selected_bins[-1]) + 1
            selected_bins -= first_bin
            span = last_bin - first_bin
            target = slice(first_bin, last_bin)
            counts[column_number, target] += np.bincount(
                selected_bins, minlength=span
            )
            sums[column_number, target] += np.bincount(
                selected_bins, weights=selected, minlength=span
            )
            sum_squares[column_number, target] += np.bincount(
                selected_bins, weights=selected * selected, minlength=span
            )

    means = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    variances = np.full_like(sums, np.nan)
    valid_std = counts > 1
    variance_numerators = sum_squares - sums * means
    np.maximum(variance_numerators, 0, out=variance_numerators)
    np.divide(
        variance_numerators,
        counts - 1,
        out=variances,
        where=valid_std,
    )
    np.sqrt(variances, out=variances)
    return means.T, variances.T, width_ns


def quality_control(data: pd.DataFrame, sample_rate: float) -> Tuple[pd.DataFrame, Info]:
    """
    Perform basic quality control on the provided data.

    This function performs the following tasks:
    1. Returns a dictionary with general information about the data.
    2. Checks for non-increasing timestamps and corrects them if necessary, returning the corrected data.

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame
    :param sample_rate: Target sample rate (Hz) to achieve.
    :type sample_rate: int or float
    :return: A tuple containing the processed data and a dictionary with general information about the data.
        The dictionary contains the following:

        - **NumTicks**: Total number of ticks (samples) in the data.
        - **StartTime**: First timestamp of the data.
        - **EndTime**: Last timestamp of the data.
        - **WearTime(days)**: Total wear time, in days. This is simply the total \
            duration of valid (non-NaN) data and does not account for potential \
            nonwear segments. See ``find_nonwear_segments`` and ``flag_nonwear`` to \
            find and flag nonwear segments in the data.
        - **DataSpan(days)**: Time span of the data (difference between last and first timestamps).
        - **NumInterrupts**: The number of interruptions in the data (gaps or NaNs between samples).
        - **ReadErrors**: The number of data errors (if non-increasing timestamps are found).
        - **Covers24hOK**: Whether the data covers all 24 hours of the day.
    :rtype: (pandas.DataFrame, dict)
    """

    info: Info = {}

    if len(data) == 0:
        info['ReadErrors'] = 0
        info['StartTime'] = None
        info['EndTime'] = None
        info['NumTicks'] = 0
        info['WearTime(days)'] = 0
        info['DataSpan(days)'] = 0
        info['NumInterrupts'] = 0
        return data, info

    # Check for non-increasing timestamps. This is rare but can happen with
    # buggy devices. TODO: Parser should do this.
    errs = _count_nonincreasing(data.index)
    if errs > 0:
        print("Found non-increasing data timestamps. Fixing...")
        data = data[_increasing_mask(data.index)]
        info['ReadErrors'] = int(np.ceil(errs / sample_rate))
    else:
        info['ReadErrors'] = 0

    # Start/end times, wear time, interrupts
    time_format = "%Y-%m-%d %H:%M:%S"
    info['StartTime'] = data.index[0].strftime(time_format)
    info['EndTime'] = data.index[-1].strftime(time_format)
    info['NumTicks'] = len(data)
    total_time, num_interrupts, covers24hok = _validity_summary(
        data, required_columns=_XYZ_COLUMNS
    )
    info['WearTime(days)'] = total_time / (60 * 60 * 24)
    info['DataSpan(days)'] = (data.index[-1] - data.index[0]).total_seconds() / (60 * 60 * 24)
    info['NumInterrupts'] = num_interrupts
    info['Covers24hOK'] = covers24hok

    return data, info


def resample(  # noqa: C901
    data: pd.DataFrame,
    sample_rate: float,
    dropna: bool = False,
    start_first_complete_minute: bool = False,
    chunksize: int = 1_000_000,
) -> Tuple[pd.DataFrame, Info]:
    """
    Nearest neighbor resampling. For downsampling, it is recommended to first
    apply an antialiasing filter (e.g. a low-pass filter, see ``lowpass``).

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param sample_rate: Target sample rate (Hz) to achieve.
    :type sample_rate: int or float
    :param dropna: Whether to drop NaN values after resampling. Defaults to False.
    :type dropna: bool, optional
    :param start_first_complete_minute: Whether to start data from the first complete minute.
        Uses 1 second tolerance - if within 1 second of minute boundary, uses that minute,
        otherwise advances to next minute. Defaults to False.
    :type start_first_complete_minute: bool, optional
    :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
    :type chunksize: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)

    Notes
    -----
    The result owns independent in-memory column storage. Because this public
    operation does not mutate its input, peak memory is approximately the input
    frame plus the complete output frame, in addition to bounded chunk-local
    temporaries.
    """

    info: Info = {}

    if _has_uniform_rate(data.index, sample_rate):
        print(f"Skipping resample: Rate {sample_rate} already achieved")
        return data, info

    info['ResampleRate'] = sample_rate

    t0, tf = data.index[0], data.index[-1]

    # Start from first complete minute if specified
    if start_first_complete_minute and len(data) > 0:
        # Check how far we are from the start of the minute
        seconds_into_minute = t0.second + t0.microsecond / 1_000_000

        if seconds_into_minute < 1.0:
            # Within 1 second of the minute boundary, use this minute
            t0 = t0.replace(second=0, microsecond=0)
        else:
            # More than 1 second into the minute, advance to next minute
            t0 = t0.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)

        # Trim data to start from the adjusted time
        data = data.loc[t0:]
        if len(data) == 0:
            return data, info
        tf = data.index[-1]
        info['FirstCompleteMinuteStart'] = t0.strftime("%Y-%m-%d %H:%M:%S")
    nt = int(np.around((tf - t0).total_seconds() * sample_rate)) + 1  # integer number of ticks we need

    source_arrays = {
        cast(str, column): data[column].to_numpy(copy=False)
        for column in data.columns
    }
    output_index: NDArray[np.int64] = np.empty(nt, dtype=np.int64)
    output = {
        column: np.empty(nt, dtype=values.dtype)
        for column, values in source_arrays.items()
    }
    tolerance = pd.Timedelta('1s')

    for i in range(0, nt, chunksize):
        chunk_size = min(chunksize, nt - i)

        # Use pd.Timedelta(n/r) instead of n * pd.Timedelta(1/r): it's not the same due to numerical precision
        t = pd.date_range(
            t0 + pd.Timedelta(i / sample_rate, unit='s'),
            t0 + pd.Timedelta((i + chunk_size - 1) / sample_rate, unit='s'),
            periods=chunk_size,
            name=data.index.name,
        )
        indexer = data.index.get_indexer(
            t, method='nearest', tolerance=tolerance, limit=1
        )
        missing = indexer < 0
        has_missing = missing.any()
        if has_missing:
            present = ~missing
            source_positions = indexer[present]
        output_index[i:i + chunk_size] = _timestamps_to_ns(
            t.asi8, _timestamp_scale(t)
        )
        for column, source in source_arrays.items():
            if has_missing:
                destination_dtype = _missing_output_dtype(output[column].dtype)
                if destination_dtype != output[column].dtype:
                    output[column] = output[column].astype(destination_dtype)
                destination = output[column][i:i + chunk_size]
                destination[:] = _missing_value(destination_dtype)
                destination[present] = source[source_positions]
            else:
                destination = output[column][i:i + chunk_size]
                destination[:] = source[indexer]

    data = pd.DataFrame(
        output,
        index=_index_from_ns(output_index, data.index),
        copy=False,
    )

    if dropna:
        # TODO: This may force a copy of the data
        data = data.dropna()

    info['NumTicksAfterResample'] = len(data)

    return data, info


def lowpass(
    data: pd.DataFrame,
    data_sample_rate: float,
    cutoff_rate: float = 20,
    chunksize: int = 1_000_000,
    _inplace: bool = False,
) -> Tuple[pd.DataFrame, Info]:
    """
    Apply Butterworth low-pass filter.

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param data_sample_rate: The data's original sample rate.
    :type data_sample_rate: int or float
    :param cutoff_rate: Cutoff (Hz) for low-pass filter. Defaults to 20.
    :type cutoff_rate: int, optional
    :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
    :type chunksize: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info: Info = {}

    # Skip this if the Nyquist freq is too low
    if data_sample_rate / 2 <= cutoff_rate:
        print(f"Skipping lowpass filter: data sample rate {data_sample_rate} too low for cutoff rate {cutoff_rate}")
        info['LowpassOK'] = 0
        return data, info

    calculation_dtype = _xyz_calculation_dtype(data)
    result, result_xyz, xyz_positions = _prepare_xyz_output(
        data, _inplace, calculation_dtype
    )
    n = len(data)
    leeway = 100  # used to minimize edge effects
    previous_tail = np.empty((0, 3), dtype=calculation_dtype)
    for i in range(0, n, chunksize):
        chunk_size = min(chunksize, n - i)
        leeway0 = min(i, leeway)
        istart = i - leeway0
        istop = min(i + chunk_size + leeway, n)

        xyz = data.iloc[istart:istop][list(_XYZ_COLUMNS)].to_numpy(
            dtype=calculation_dtype,
            na_value=np.nan,
        )
        if not xyz.flags.writeable:
            xyz = xyz.copy()
        if _inplace and leeway0:
            xyz[:leeway0] = previous_tail[-leeway0:]
        if _inplace:
            original_core = xyz[leeway0:leeway0 + chunk_size]
            if chunk_size >= leeway:
                previous_tail = original_core[-leeway:].copy()
            else:
                previous_tail = np.concatenate((previous_tail, original_core))[-leeway:].copy()
        na = np.isnan(xyz).any(1)
        xyz[na] = 0.0  # temporarily replace nans with 0s for butterfilt
        xyz = butterfilt(xyz, cutoff_rate, fs=data_sample_rate, axis=0)
        xyz[na] = np.nan  # restore nans
        xyz = xyz[leeway0:leeway0 + chunk_size]
        _write_xyz(result, result_xyz, xyz_positions, i, xyz)

    data = result

    info['LowpassOK'] = 1
    info['LowpassCutoff(Hz)'] = cutoff_rate

    return data, info


def flag_nonwear(
    data: pd.DataFrame,
    patience: str = '90m',
    window: str = '10s',
    stdtol: float = 15 / 1000,
    _inplace: bool = False,
) -> Tuple[pd.DataFrame, Info]:
    """
    Flag nonwear episodes in the data by setting them to NA. Non-wear episodes are inferred from long periods of no movement.

    :param pandas.DataFrame data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param patience: The minimum duration that a stationary episode must have to be classified as non-wear episode. Defaults to 90 minutes ("90m").
    :type patience: str, optional
    :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
    :type window: str, optional
    :param stdtol: Standard deviation under which the window is considered stationary. Defaults to 15 milligravity (0.015).
    :type stdtol: float, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info: Info = {}

    nonwear_segments = find_nonwear_segments(data, patience=patience, window=window, stdtol=stdtol)

    # Num nonwear episodes and total nonwear time
    count_nonwear = len(nonwear_segments)
    total_nonwear = nonwear_segments.sum().total_seconds()

    if not _inplace:
        data = data.copy(deep=True)

    if count_nonwear:
        for start_time, length in nonwear_segments.items():
            data.loc[start_time:start_time + length] = np.nan
    del nonwear_segments

    total_time, num_interrupts, covers24hok = _validity_summary(data)

    info['NonwearTime(days)'] = total_nonwear / (60 * 60 * 24)
    info['NumNonwearEpisodes'] = count_nonwear
    info['WearTime(days)'] = total_time / (60 * 60 * 24)
    info['NumInterrupts'] = num_interrupts
    info['Covers24hOK'] = covers24hok

    return data, info


def calibrate_gravity(  # noqa: C901
    data: pd.DataFrame,
    calib_cube: float = 0.3,
    calib_min_samples: int = 50,
    window: str = '10s',
    stdtol: float = 15 / 1000,
    stdtol_min: Optional[float] = None,
    return_coeffs: bool = True,
    chunksize: int = 1_000_000,
    _inplace: bool = False,
) -> Tuple[pd.DataFrame, Info]:
    """
    Gravity calibration method of van Hees et al. 2014 (https://pubmed.ncbi.nlm.nih.gov/25103964/)

    :param data: A pandas.DataFrame of acceleration time-series. It must contain
        at least columns `x,y,z` and the index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param calib_cube: Calibration cube criteria. See van Hees et al. 2014 for details. Defaults to 0.3.
    :type calib_cube: float, optional.
    :param calib_min_samples: Minimum number of stationary samples required to run calibration. Defaults to 50.
    :type calib_min_samples: int, optional.
    :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
    :type window: str, optional
    :param stdtol: Standard deviation under which a window is considered stationary. Defaults to 15 milligravity (0.015).
    :type stdtol: float, optional
    :param stdtol_min: Minimum standard deviation above which a window is considered valid. Defaults to None (no filtering).
    :type stdtol_min: float, optional
    :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
    :type chunksize: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info: Info = {}

    hasT = 'temperature' in data
    columns = _XYZ_COLUMNS + ('temperature',) if hasT else _XYZ_COLUMNS
    means, deviations, _ = _window_statistics(data, columns, window)
    xyz_deviations = deviations[:, :3]
    stationary_indicator = np.all(xyz_deviations < stdtol, axis=1)
    if stdtol_min is not None:
        stationary_indicator &= np.all(xyz_deviations > stdtol_min, axis=1)

    xyz = means[stationary_indicator, :3]
    xyz = xyz[~np.isnan(xyz).any(axis=1)]
    # Remove any nonzero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]

    if hasT:
        T = means[stationary_indicator, 3]
        T = T[~np.isnan(T)]
        T = T[nonzero]

    del means, deviations, stationary_indicator, xyz_deviations
    del nonzero

    info['CalibNumSamples'] = len(xyz)

    if len(xyz) < calib_min_samples:
        info['CalibErrorBefore(mg)'] = np.nan
        info['CalibErrorAfter(mg)'] = np.nan
        info['CalibOK'] = 0
        warnings.warn(
            f"Skipping calibration: Insufficient stationary samples: {len(xyz)} < {calib_min_samples}",
            stacklevel=2,
        )
        return data, info

    intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
    slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
    best_intercept = np.copy(intercept)
    best_slope = np.copy(slope)

    if hasT:
        slopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
        best_slopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr - target, axis=1)
    err = np.mean(errors)  # MAE more robust than RMSE. This is different from the paper
    init_err = err
    best_err = 1e16

    MAXITER = 1000
    IMPROV_TOL = 0.0001
    ERR_TOL = 0.01

    info['CalibErrorBefore(mg)'] = init_err * 1000

    # Check that we have sufficiently uniformly distributed points:
    # need at least one point outside each face of the cube
    if (np.max(xyz, axis=0) < calib_cube).any() or (np.min(xyz, axis=0) > -calib_cube).any():
        info['CalibErrorAfter(mg)'] = init_err * 1000
        info['CalibNumIters'] = 0
        info['CalibOK'] = 0

        return data, info

    # If initial error is already below threshold, skip and return
    if init_err < ERR_TOL:
        info['CalibErrorAfter(mg)'] = init_err * 1000
        info['CalibNumIters'] = 0
        info['CalibOK'] = 1

        return data, info

    for _iteration in range(MAXITER):

        # Weighting. Outliers are zeroed out
        # This is different from the paper
        maxerr = np.quantile(errors, .995)
        weights = np.maximum(1 - errors / maxerr, 0)

        # Optimize params for each axis
        for k in range(3):

            inp = curr[:, k]
            out = target[:, k]
            if hasT:
                inp = np.column_stack((inp, T))
            inp = sm.add_constant(inp, prepend=True, has_constant='add')
            params = sm.WLS(out, inp, weights=weights).fit().params
            # In the following,
            # intercept == params[0]
            # slope == params[1]
            # slopeT == params[2]  (if exists)
            intercept[k] = params[0] + (intercept[k] * params[1])
            slope[k] = params[1] * slope[k]
            if hasT:
                slopeT[k] = params[2] + (slopeT[k] * params[1])

        # Update current solution and target
        curr = intercept + (xyz * slope)
        if hasT:
            curr = curr + (T[:, None] * slopeT)
        target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

        # Update errors
        errors = np.linalg.norm(curr - target, axis=1)
        err = np.mean(errors)
        err_improv = (best_err - err) / best_err

        if err < best_err:
            best_intercept = np.copy(intercept)
            best_slope = np.copy(slope)
            if hasT:
                best_slopeT = np.copy(slopeT)
            best_err = err
        if err_improv < IMPROV_TOL:
            break

    info['CalibErrorAfter(mg)'] = best_err * 1000
    info['CalibNumIters'] = _iteration + 1

    if (best_err >= ERR_TOL) or (_iteration + 1 >= MAXITER):
        info['CalibOK'] = 0

        return data, info

    calculation_dtype = np.dtype(np.float64)
    result, result_xyz, xyz_positions = _prepare_xyz_output(
        data, _inplace, calculation_dtype
    )
    n = len(data)
    for i in range(0, n, chunksize):
        chunk_size = min(chunksize, n - i)
        chunk = data.iloc[i:i + chunk_size]
        chunk_xyz = chunk[list(_XYZ_COLUMNS)].to_numpy(
            dtype=calculation_dtype,
            na_value=np.nan,
        )
        chunk_xyz = best_intercept + best_slope * chunk_xyz
        if hasT:
            chunk_T = chunk['temperature'].to_numpy()
            chunk_xyz = chunk_xyz + best_slopeT * chunk_T[:, None]
        _write_xyz(result, result_xyz, xyz_positions, i, chunk_xyz)

    data = result
    info['CalibOK'] = 1

    if return_coeffs:
        info['CalibxIntercept'] = best_intercept[0]
        info['CalibyIntercept'] = best_intercept[1]
        info['CalibzIntercept'] = best_intercept[2]
        info['CalibxSlope'] = best_slope[0]
        info['CalibySlope'] = best_slope[1]
        info['CalibzSlope'] = best_slope[2]
        if hasT:
            info['CalibxSlopeT'] = best_slopeT[0]
            info['CalibySlopeT'] = best_slopeT[1]
            info['CalibzSlopeT'] = best_slopeT[2]

    return data, info


def find_nonwear_segments(
    data: pd.DataFrame,
    patience: str = '90m',
    window: str = '10s',
    stdtol: float = 15 / 1000,
) -> pd.Series:
    """
    Find nonwear episodes based on long periods of no movement.

    :param pandas.DataFrame data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param patience: The minimum duration that a stationary episode must have to be classified as non-wear episode. Defaults to 90 minutes ("90m").
    :type patience: str, optional
    :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
    :type window: str, optional
    :param stdtol: Standard deviation under which the window is considered stationary. Defaults to 15 milligravity (0.015).
    :type stdtol: float, optional
    :return: A Series where the DatetimeIndex indicates the start times of each non-wear segment and the values are the length
        of each segment, in timedelta64[ns].
    :rtype: pandas.Series
    """

    _, deviations, width_ns = _window_statistics(
        data,
        _XYZ_COLUMNS,
        window,
        forward_fill=True,
    )
    stationary = np.all(deviations < stdtol, axis=1)
    transitions = np.flatnonzero(
        np.diff(np.concatenate(([False], stationary, [False])))
    )
    starts = transitions[::2]
    stops = transitions[1::2]
    lengths_ns = (stops - starts - 1).astype(np.int64) * width_ns
    keep = lengths_ns > int(pd.Timedelta(patience).value)

    timestamp_scale = _timestamp_scale(data.index)
    origin_ns = _timestamp_to_ns(
        int(data.index.asi8[0]), timestamp_scale
    )
    start_times = origin_ns + starts[keep].astype(np.int64) * width_ns
    return pd.Series(
        lengths_ns[keep].astype('timedelta64[ns]'),
        index=_index_from_ns(
            start_times,
            data.index.rename('start_time'),
        ),
        name='length',
        dtype='timedelta64[ns]',
    )


def get_wear_time(t: pd.Series, tol: float = 0.1) -> Tuple[float, int]:
    """ Return wear time in seconds and number of interrupts. """
    tdiff = t.diff()
    ttol = tdiff.mode().max() * (1 + tol)
    total_time = tdiff[tdiff <= ttol].sum().total_seconds()
    num_interrupts = (tdiff > ttol).sum()
    return total_time, cast(int, num_interrupts)


def butterfilt(
    x: Array,
    cutoffs: Union[float, Tuple[float, Optional[float]]],
    fs: float,
    order: int = 8,
    axis: int = 0,
) -> Array:
    """ Butterworth filter. """
    nyq = 0.5 * fs
    Wn: Union[float, Tuple[float, float]]
    if isinstance(cutoffs, tuple):
        hicut, lowcut = cutoffs
        if hicut > 0:
            if lowcut is not None:
                btype = 'bandpass'
                Wn = (hicut / nyq, lowcut / nyq)
            else:
                btype = 'highpass'
                Wn = hicut / nyq
        else:
            btype = 'lowpass'
            Wn = cast(float, lowcut) / nyq
    else:
        btype = 'lowpass'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    y = y.astype(x.dtype, copy=False)

    return cast(Array, y)


def chunker(
    data: pd.DataFrame,
    chunksize: str = '4h',
    leeway: str = '0h',
    fn: Optional[Callable[[pd.DataFrame], Any]] = None,
    fntrim: bool = True,
) -> Iterator[Any]:
    """ Return chunk generator for a given datetime-indexed DataFrame.
    A `leeway` parameter can be used to obtain overlapping chunks (e.g. leeway='30m').
    If a function `fn` is provided, it is applied to each chunk. The leeway is
    trimmed after function application by default (set `fntrim=False` to skip).
    """

    chunksize = pd.Timedelta(chunksize)
    leeway = pd.Timedelta(leeway)
    zero = pd.Timedelta(0)

    t0, tf = data.index[0], data.index[-1]

    for ti in pd.date_range(t0, tf, freq=chunksize):
        start = ti - min(ti - t0, leeway)
        stop = ti + chunksize + leeway
        chunk = slice_time(data, start, stop)

        if fn is not None:
            chunk = fn(chunk)

            if leeway > zero and fntrim:
                try:
                    chunk = slice_time(chunk, ti, ti + chunksize)
                except Exception:
                    warnings.warn(
                        f"Could not trim chunk. Ignoring fntrim={fntrim}...",
                        stacklevel=2,
                    )

        yield chunk


def slice_time(x: Any, start: Any, stop: Any) -> Any:
    """ In pandas, slicing DateTimeIndex arrays is right-closed.
    This function performs right-open slicing. """
    x = x.loc[start : stop]
    x = x[x.index != stop]
    return x


def npy2df(data: Array) -> pd.DataFrame:
    """ Convert a numpy structured array to pandas dataframe. Also parse time
    and set as index. This function will avoid copies whenever possible. """

    t = pd.to_datetime(data['time'], unit='ms')
    t.name = 'time'
    names = cast(Tuple[str, ...], data.dtype.names)
    columns = [c for c in names if c != 'time']
    data = pd.DataFrame({c: data[c] for c in columns}, index=t, copy=False)
    return data
