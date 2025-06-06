import os
import tempfile
import numpy as np
import pandas as pd
import scipy.signal as signal
import statsmodels.api as sm
import warnings


__all__ = ['quality_control', 'lowpass', 'calibrate_gravity', 'flag_nonwear', 'find_nonwear_segments', 'resample']


def quality_control(data, sample_rate):
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

    info = {}

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
    tdiff = data.index.to_series().diff()  # Note: Index.diff() was only added in pandas 2.1
    errs = (tdiff <= pd.Timedelta(0)).sum()
    del tdiff  # we're done with this
    if errs > 0:
        print("Found non-increasing data timestamps. Fixing...")
        data = data[data
                    .index
                    .to_series()
                    .cummax()
                    .diff()
                    .fillna(pd.Timedelta(1))
                    .gt(pd.Timedelta(0))]
        info['ReadErrors'] = int(np.ceil(errs / sample_rate))
    else:
        info['ReadErrors'] = 0

    # Start/end times, wear time, interrupts
    time_format = "%Y-%m-%d %H:%M:%S"
    info['StartTime'] = data.index[0].strftime(time_format)
    info['EndTime'] = data.index[-1].strftime(time_format)
    info['NumTicks'] = len(data)
    tol = pd.Timedelta('1s')
    tdiff = data.dropna().index.to_series().diff()  # Note: Index.diff() was only added in pandas 2.1
    total_time = tdiff[tdiff < tol].sum().total_seconds()
    num_interrupts = (tdiff > tol).sum()
    del tdiff  # we're done with this
    info['WearTime(days)'] = total_time / (60 * 60 * 24)
    info['DataSpan(days)'] = (data.index[-1] - data.index[0]).total_seconds() / (60 * 60 * 24)
    info['NumInterrupts'] = num_interrupts

    # Check if data covers all 24 hours of the day
    coverage = data.notna().any(axis=1).groupby(data.index.hour).mean()
    info['Covers24hOK'] = int(len(coverage) == 24 and np.min(coverage) >= 0.01)
    del coverage

    return data, info


def resample(data, sample_rate, dropna=False, chunksize=1_000_000):
    """
    Nearest neighbor resampling. For downsampling, it is recommended to first
    apply an antialiasing filter (e.g. a low-pass filter, see ``lowpass``).

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param sample_rate: Target sample rate (Hz) to achieve.
    :type sample_rate: int or float
    :param dropna: Whether to drop NaN values after resampling. Defaults to False.
    :type dropna: bool, optional
    :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
    :type chunksize: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info = {}

    if np.isclose(
        1 / sample_rate,
        pd.Timedelta(pd.infer_freq(data.index)).total_seconds(),
    ):
        print(f"Skipping resample: Rate {sample_rate} already achieved")
        return data, info

    info['ResampleRate'] = sample_rate

    t0, tf = data.index[0], data.index[-1]
    nt = int(np.around((tf - t0).total_seconds() * sample_rate)) + 1  # integer number of ticks we need

    # # In-memory version
    # tf = t0 + pd.Timedelta((nt - 1) / sample_rate, unit='s')  # adjust tf
    # t = pd.date_range(t0, tf, periods=nt, name=data.index.name)
    # data = data.reindex(t, method='nearest', tolerance=pd.Timedelta('1s'), limit=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # We use TemporaryDirectory() + filename instead of NamedTemporaryFile()
        # because we don't want to open the file just yet:
        # https://stackoverflow.com/questions/26541416/generate-temporary-file-names-without-creating-actual-file-in-python
        # and Windows doesn't allow opening a file twice:
        # https://docs.python.org/3.9/library/tempfile.html#tempfile.NamedTemporaryFile
        mmap_fname = os.path.join(tmpdir, 'data.mmap')

        data_mmap = mmap_like(data, mmap_fname, shape=(nt,))

        for i in range(0, nt, chunksize):

            # If last chunk, adjust chunksize
            if i + chunksize > nt:
                chunksize = nt - i

            # Use pd.Timedelta(n/r) instead of n * pd.Timedelta(1/r): it's not the same due to numerical precision
            t = pd.date_range(
                t0 + pd.Timedelta(i / sample_rate, unit='s'),
                t0 + pd.Timedelta((i + chunksize - 1) / sample_rate, unit='s'),
                periods=chunksize,
                name=data.index.name,
            )
            chunk = data.reindex(t, method='nearest', tolerance=pd.Timedelta('1s'), limit=1)
            copy2mmap(chunk, data_mmap[i:i + chunksize])

        del data

        # We need to copy so that the mmap file can be trully deleted: 
        # https://stackoverflow.com/questions/24178460/in-python-is-it-possible-to-overload-numpys-memmap-to-delete-itself-when-the-m
        data = mmap2df(data_mmap, copy=True)

        del data_mmap

    if dropna:
        # TODO: This may force a copy of the data
        data = data.dropna()

    info['NumTicksAfterResample'] = len(data)

    return data, info


def lowpass(data, data_sample_rate, cutoff_rate=20, chunksize=1_000_000):
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

    info = {}

    # Skip this if the Nyquist freq is too low
    if data_sample_rate / 2 <= cutoff_rate:
        print(f"Skipping lowpass filter: data sample rate {data_sample_rate} too low for cutoff rate {cutoff_rate}")
        info['LowpassOK'] = 0
        return data, info

    # # In-memory version
    # xyz = data[['x', 'y', 'z']].to_numpy()
    # where_nan = np.isnan(xyz).any(1)  # temporarily replace nans with 0s for butterfilt
    # xyz[where_nan] = 0
    # xyz = butterfilt(xyz, cutoff_rate, fs=data_sample_rate, axis=0)
    # xyz[where_nan] = np.nan  # restore nans
    # data = data.copy(deep=True)  # copy to avoid modifying original data
    # data[['x', 'y', 'z']] = xyz

    with tempfile.TemporaryDirectory() as tmpdir:
        # We use TemporaryDirectory() + filename instead of NamedTemporaryFile()
        # because we don't want to open the file just yet:
        # https://stackoverflow.com/questions/26541416/generate-temporary-file-names-without-creating-actual-file-in-python
        # and Windows doesn't allow opening a file twice:
        # https://docs.python.org/3.9/library/tempfile.html#tempfile.NamedTemporaryFile
        mmap_fname = os.path.join(tmpdir, 'data.mmap')

        n = len(data)
        data_mmap = mmap_like(data, mmap_fname, shape=(n,))

        leeway = 100  # used to minimize edge effects
        for i in range(0, n, chunksize):

            leeway0 = min(i, leeway)
            istart = i - leeway0
            istop = i + chunksize + leeway

            chunk = data.iloc[istart : istop]
            xyz = chunk[['x', 'y', 'z']].to_numpy()
            na = np.isnan(xyz).any(1)
            xyz[na] = 0.0  # temporarily replace nans with 0s for butterfilt
            xyz = butterfilt(xyz, cutoff_rate, fs=data_sample_rate, axis=0)
            xyz[na] = np.nan  # restore nans
            chunk = chunk.copy(deep=True)  # copy to avoid modifying original data
            chunk[['x', 'y', 'z']] = xyz
            chunk = chunk.iloc[leeway0 : leeway0 + chunksize]  # trim leeway
            copy2mmap(chunk, data_mmap[i:i + chunksize])

        del data

        # We need to copy so that the mmap file can be trully deleted: 
        # https://stackoverflow.com/questions/24178460/in-python-is-it-possible-to-overload-numpys-memmap-to-delete-itself-when-the-m
        data = mmap2df(data_mmap, copy=True)

        del data_mmap

    info['LowpassOK'] = 1
    info['LowpassCutoff(Hz)'] = cutoff_rate

    return data, info


def flag_nonwear(data, patience='90m', window='10s', stdtol=15 / 1000):
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

    info = {}

    nonwear_segments = find_nonwear_segments(data, patience=patience, window=window, stdtol=stdtol)

    # Num nonwear episodes and total nonwear time
    count_nonwear = len(nonwear_segments)
    total_nonwear = nonwear_segments.sum().total_seconds()

    # Flag nonwear segments
    data = data.copy(deep=True)  # copy to avoid modifying original data
    for start_time, length in nonwear_segments.items():
        data.loc[start_time : start_time + length] = np.nan
    del nonwear_segments

    # Calculate total wear time and interrupts (data gaps)
    tol = pd.Timedelta('1s')
    tdiff = data.dropna().index.to_series().diff()  # Note: Index.diff() was only added in pandas 2.1
    total_time = tdiff[tdiff < tol].sum().total_seconds()
    num_interrupts = (tdiff > tol).sum()
    del tdiff

    # Check if data covers all 24 hours of the day
    coverage = data.notna().any(axis=1).groupby(data.index.hour).mean()
    covers24hok = int(len(coverage) == 24 and np.min(coverage) >= 0.01)
    del coverage

    info['NonwearTime(days)'] = total_nonwear / (60 * 60 * 24)
    info['NumNonwearEpisodes'] = count_nonwear
    info['WearTime(days)'] = total_time / (60 * 60 * 24)
    info['NumInterrupts'] = num_interrupts
    info['Covers24hOK'] = covers24hok

    return data, info


def calibrate_gravity(data, calib_cube=0.3, calib_min_samples=50, window='10s', stdtol=15 / 1000, return_coeffs=True, chunksize=1_000_000):  # noqa: C901
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
    :param stdtol: Standard deviation under which the window is considered stationary. Defaults to 15 milligravity (0.015).
    :type stdtol: float, optional
    :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
    :type chunksize: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info = {}

    stationary_indicator = (  # this is more memory friendly than of data[['x', 'y', 'z']].std()
        data['x'].resample(window, origin='start').std().lt(stdtol)
        & data['y'].resample(window, origin='start').std().lt(stdtol)
        & data['z'].resample(window, origin='start').std().lt(stdtol)
    )

    xyz = (
        data[['x', 'y', 'z']]
        .resample(window, origin='start').mean()
        [stationary_indicator]
        .dropna()
        .to_numpy()
    )
    # Remove any nonzero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]

    hasT = 'temperature' in data
    if hasT:
        T = (
            data['temperature']
            .resample(window, origin='start').mean()
            [stationary_indicator]
            .dropna()
            .to_numpy()
        )
        T = T[nonzero]

    del stationary_indicator
    del nonzero

    info['CalibNumSamples'] = len(xyz)

    if len(xyz) < calib_min_samples:
        info['CalibErrorBefore(mg)'] = np.nan
        info['CalibErrorAfter(mg)'] = np.nan
        info['CalibOK'] = 0
        warnings.warn(f"Skipping calibration: Insufficient stationary samples: {len(xyz)} < {calib_min_samples}")
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

    for it in range(MAXITER):

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
    info['CalibNumIters'] = it + 1

    if (best_err >= ERR_TOL) or (it + 1 >= MAXITER):
        info['CalibOK'] = 0

        return data, info

    else:

        # # In-memory version
        # data = data.copy()
        # data[['x', 'y', 'z']] = (best_intercept
        #                          + best_slope * data[['x', 'y', 'z']].to_numpy())
        # if hasT:
        #     data[['x', 'y', 'z']] = (data[['x', 'y', 'z']]
        #                              + best_slopeT * (data['temperature'].to_numpy()[:, None]))

        with tempfile.TemporaryDirectory() as tmpdir:
            # We use TemporaryDirectory() + filename instead of NamedTemporaryFile()
            # because we don't want to open the file just yet:
            # https://stackoverflow.com/questions/26541416/generate-temporary-file-names-without-creating-actual-file-in-python
            # and Windows doesn't allow opening a file twice:
            # https://docs.python.org/3.9/library/tempfile.html#tempfile.NamedTemporaryFile

            n = len(data)
            mmap_fname = os.path.join(tmpdir, 'data.mmap')
            data_mmap = mmap_like(data, mmap_fname, shape=(n,))

            for i in range(0, n, chunksize):

                # If last chunk, adjust chunksize
                if i + chunksize > n:
                    chunksize = n - i

                chunk = data.iloc[i:i + chunksize]
                chunk_xyz = chunk[['x', 'y', 'z']].to_numpy()
                chunk_xyz = best_intercept + best_slope * chunk_xyz
                if hasT:
                    chunk_T = chunk['temperature'].to_numpy()
                    chunk_xyz = chunk_xyz + best_slopeT * chunk_T[:, None]
                chunk = chunk.copy(deep=True)  # copy to avoid modifying original data
                chunk[['x', 'y', 'z']] = chunk_xyz

                copy2mmap(chunk, data_mmap[i:i + chunksize])

            del data

            # We need to copy so that the mmap file can be trully deleted: 
            # https://stackoverflow.com/questions/24178460/in-python-is-it-possible-to-overload-numpys-memmap-to-delete-itself-when-the-m
            data = mmap2df(data_mmap, copy=True)

            del data_mmap

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


# def get_stationary_indicator(data, window='10s', stdtol=15 / 1000):
#     """
#     Return a boolean pandas.Series indicating stationary (low movement) periods.

#     :param data: A pandas.DataFrame of acceleration time-series. It must contain
#         at least columns `x,y,z` and the index must be a DateTimeIndex.
#     :type data: pandas.DataFrame.
#     :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
#     :type window: str, optional
#     :param stdtol: Standard deviation under which the window is considered stationary.
#         Defaults to 15 milligravity (0.015).
#     :type stdtol: float, optional
#     :return: Boolean pandas.Series indexed as `data` indicating stationary periods.
#     :rtype: pandas.Series
#     """

#     def fn(data):
#         return (
#             (data[['x', 'y', 'z']]
#              .rolling(window)
#              .std()
#              < stdtol)
#             .all(axis=1)
#         )

#     stationary_indicator = pd.concat(
#         chunker(
#             data,
#             chunksize='4h',
#             leeway=window,
#             fn=fn
#         )
#     )

#     return stationary_indicator


def find_nonwear_segments(data, patience='90m', window='10s', stdtol=15 / 1000):
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

    stationary_indicator = (  # this is more memory friendly than data[['x', 'y', 'z']].std()
        # use ffill() so that NA also counted as stationary
        data['x'].ffill().resample(window, origin='start').std().lt(stdtol)
        & data['y'].ffill().resample(window, origin='start').std().lt(stdtol)
        & data['z'].ffill().resample(window, origin='start').std().lt(stdtol)
    )

    segment_edges = (stationary_indicator != stationary_indicator.shift(1))
    segment_edges.iloc[0] = True  # first edge is always True
    segment_ids = segment_edges.cumsum()
    stationary_segment_ids = segment_ids[stationary_indicator]
    stationary_segment_lengths = (
        stationary_segment_ids
        .groupby(stationary_segment_ids)
        .agg(
            start_time=lambda x: x.index[0],
            length=lambda x: x.index[-1] - x.index[0]
        )
        .set_index('start_time')
        .squeeze(axis=1)
        # dtype defaults to int64 when series is empty, so
        # astype('timedelta64[ns]') makes sure it's always a timedelta,
        # otherwise comparison with Timedelta(patience) below will fail
        .astype('timedelta64[ns]')
    )
    nonwear_segment_lengths = stationary_segment_lengths[stationary_segment_lengths > pd.Timedelta(patience)]

    return nonwear_segment_lengths


def get_wear_time(t, tol=0.1):
    """ Return wear time in seconds and number of interrupts. """
    tdiff = t.diff()
    ttol = tdiff.mode().max() * (1 + tol)
    total_time = tdiff[tdiff <= ttol].sum().total_seconds()
    num_interrupts = (tdiff > ttol).sum()
    return total_time, num_interrupts


def butterfilt(x, cutoffs, fs, order=8, axis=0):
    """ Butterworth filter. """
    nyq = 0.5 * fs
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
            Wn = lowcut / nyq
    else:
        btype = 'lowpass'
        Wn = cutoffs / nyq
    sos = signal.butter(order, Wn, btype=btype, analog=False, output='sos')
    y = signal.sosfiltfilt(sos, x, axis=axis)
    y = y.astype(x.dtype, copy=False)

    return y


def chunker(data, chunksize='4h', leeway='0h', fn=None, fntrim=True):
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
                    warnings.warn(f"Could not trim chunk. Ignoring fntrim={fntrim}...")

        yield chunk


def slice_time(x, start, stop):
    """ In pandas, slicing DateTimeIndex arrays is right-closed.
    This function performs right-open slicing. """
    x = x.loc[start : stop]
    x = x[x.index != stop]
    return x


def npy2df(data):
    """ Convert a numpy structured array to pandas dataframe. Also parse time
    and set as index. This function will avoid copies whenever possible. """

    t = pd.to_datetime(data['time'], unit='ms')
    t.name = 'time'
    columns = [c for c in data.dtype.names if c != 'time']
    data = pd.DataFrame({c: data[c] for c in columns}, index=t, copy=False)
    return data


def mmap_like(data, filename, mode='w+', shape=None):
    dtype = np.dtype([
        (data.index.name, data.index.dtype), 
        *[(c, data[c].dtype) for c in data.columns]
    ])
    shape = shape or (len(data),)
    data_mmap = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
    return data_mmap


def copy2mmap(data, data_mmap, flush=True):
    """ Copy a pandas.DataFrame to a numpy.memmap. This operation is in-place.

    :param data: A pandas.DataFrame of acceleration time-series.
    :type data: pandas.DataFrame.
    :param data_mmap: A numpy.memmap to copy data into.
    :type data_mmap: numpy.memmap.
    """
    data_mmap[data.index.name] = data.index.to_numpy()
    for c in data.columns:
        data_mmap[c] = data[c].to_numpy()
    if flush:
        np.memmap.flush(data_mmap)
    return


def mmap2df(data_mmap, index_col='time', copy=True):
    """ Convert a numpy structured array to pandas dataframe. """
    columns = [c for c in data_mmap.dtype.names if c != index_col]
    data = pd.DataFrame(
        {c: np.asarray(data_mmap[c]) for c in columns}, copy=copy,
        index=pd.Index(np.asarray(data_mmap[index_col]), name=index_col, copy=copy),
    )
    return data
