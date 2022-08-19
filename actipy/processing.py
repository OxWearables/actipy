import numpy as np
import pandas as pd
import scipy.signal as signal
import statsmodels.api as sm
import warnings


__all__ = ['lowpass', 'calibrate_gravity', 'detect_nonwear', 'resample', 'get_stationary_indicator']


def resample(data, sample_rate, dropna=False):
    """
    Nearest neighbor resampling. For downsampling, it is recommended to first
    apply an antialiasing filter.

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param sample_rate: Target sample rate (Hz) to achieve.
    :type sample_rate: int or float
    :param dropna: Whether to drop NaN values after resampling. Defaults to False.
    :type dropna: bool, optional
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
    nt = int(np.around((tf - t0).total_seconds() * sample_rate))  # integer number of ticks we need
    tf = t0 + pd.Timedelta(nt / sample_rate, unit='s')  # adjust end tick
    t = pd.date_range(t0, tf, periods=nt + 1, name='time')

    data = data.reindex(t, method='nearest', tolerance=pd.Timedelta('1s'), limit=1)

    if dropna:
        data = data.dropna()

    info['NumTicksAfterResample'] = len(data)

    return data, info


def lowpass(data, data_sample_rate, cutoff_rate=20):
    """
    Apply Butterworth low-pass filter.

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param data_sample_rate: The data's original sample rate.
    :type data_sample_rate: int or float
    :param cutoff_rate: Cutoff (Hz) for low-pass filter. Defaults to 20.
    :type cutoff_rate: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info = {}

    orig_index = data.index
    data, _ = resample(data, data_sample_rate, dropna=False)

    # Butter filter to remove high freq noise.
    # Default: 20Hz (most of human motion is under 20Hz)
    # Skip this if the Nyquist freq is too low
    if data_sample_rate / 2 > cutoff_rate:
        xyz = data[['x', 'y', 'z']].to_numpy()
        # Temporarily replace nans with 0s for butterfilt
        where_nan = np.isnan(xyz).any(1)
        xyz[where_nan] = 0
        xyz = butterfilt(xyz, cutoff_rate, fs=data_sample_rate, axis=0)
        # Now restore nans
        xyz[where_nan] = np.nan
        data[['x', 'y', 'z']] = xyz
        info['LowpassOK'] = 1
        info['LowpassCutoff(Hz)'] = cutoff_rate
    else:
        print(f"Skipping lowpass filter: data sample rate {data_sample_rate} too low for cutoff rate {cutoff_rate}")
        info['LowpassOK'] = 0

    data = data.reindex(orig_index,
                        method='nearest',
                        tolerance=pd.Timedelta('1s'),
                        limit=1)

    return data, info


def detect_nonwear(data, patience='90m', stationary_indicator=None, drop=False):
    """
    Detect nonwear episodes based on long periods of no movement.

    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param patience: Minimum length of the stationary period to be flagged as
        non-wear. Defaults to 90 minutes ("90m").
    :type patience: str, optional
    :param stationary_indicator: A boolean pandas.Series indexed as `data`
        indicating stationary (low movement) periods. If None, it will be
        automatically inferred. Defaults to None.
    :type stationary_indicator: pandas.Series, optional
    :param drop: Wheter to drop the non-wear periods. If False, the non-wear
        periods will be filled with NaNs. Defaults to False.
    :type drop: bool, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info = {}

    if stationary_indicator is None:
        stationary_indicator = get_stationary_indicator(data)

    group = ((stationary_indicator != stationary_indicator.shift(1))
             .cumsum()
             .where(stationary_indicator))
    stationary_len = (group.groupby(group, dropna=True)
                           .apply(lambda g: g.index[-1] - g.index[0]))
    nonwear_len = stationary_len[stationary_len > pd.Timedelta(patience)]

    nonwear_time = nonwear_len.sum().total_seconds()
    wear_time, _ = get_wear_time(data.index.to_series())
    info['WearTime(days)'] = (wear_time - nonwear_time) / (60 * 60 * 24)  # update wear time
    info['NonwearTime(days)'] = nonwear_time / (60 * 60 * 24)
    info['NumNonwearEpisodes'] = len(nonwear_len)

    # Flag nonwear
    nonwear_indicator = group.isin(nonwear_len.index)
    if drop:
        data = data[~nonwear_indicator]
    else:
        data = data.mask(nonwear_indicator)

    return data, info


def calibrate_gravity(data, calib_cube=0.3, calib_min_samples=50, stationary_indicator=None):  # noqa: C901
    """
    Gravity calibration method of van Hees et al. 2014 (https://pubmed.ncbi.nlm.nih.gov/25103964/)

    :param data: A pandas.DataFrame of acceleration time-series. It must contain
        at least columns `x,y,z` and the index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param calib_cube: Calibration cube criteria. See van Hees et al. 2014 for details. Defaults to 0.3.
    :type calib_cube: float, optional.
    :param calib_min_samples: Minimum number of stationary samples required to run calibration. Defaults to 50.
    :type calib_min_samples: int, optional.
    :param stationary_indicator: A boolean pandas.Series indexed as `data`
        indicating stationary (low movement) periods. If None, it will be
        automatically inferred. Defaults to None.
    :type stationary_indicator: pandas.Series, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    info = {}

    if stationary_indicator is None:
        stationary_indicator = get_stationary_indicator(data)

    # The paper uses 10sec averages instead of the raw ticks.
    # This reduces computational cost. Also reduces influence of outliers.
    stationary_data = (data[stationary_indicator]
                       .resample('10s')
                       .mean()
                       .dropna())

    hasT = 'temperature' in stationary_data

    xyz = stationary_data[['x', 'y', 'z']].to_numpy()
    # Remove any nonzero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]
    if hasT:
        T = stationary_data['temperature'].to_numpy()
        T = T[nonzero]
    del stationary_data
    del nonzero

    if len(xyz) < calib_min_samples:
        info['CalibOK'] = 0
        info['CalibErrorBefore(mg)'] = np.nan
        info['CalibErrorAfter(mg)'] = np.nan
        warnings.warn(f"Skipping calibration: Insufficient stationary samples")
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
    if (np.max(xyz, axis=0) < calib_cube).any() \
            or (np.min(xyz, axis=0) > -calib_cube).any():
        info['CalibOK'] = 0
        info['CalibErrorAfter(mg)'] = init_err * 1000

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

    if (best_err > ERR_TOL) or (it + 1 == MAXITER):
        info['CalibOK'] = 0

        return data, info

    else:
        data = data.copy()
        data[['x', 'y', 'z']] = (best_intercept
                                 + best_slope * data[['x', 'y', 'z']].to_numpy())
        if hasT:
            data[['x', 'y', 'z']] = (data[['x', 'y', 'z']]
                                     + best_slopeT * (data['temperature'].to_numpy()[:, None]))

        info['CalibOK'] = 1
        info['CalibNumIters'] = it + 1
        info['CalibNumSamples'] = len(xyz)
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


def get_stationary_indicator(data, window='10s', stdtol=15 / 1000):
    """
    Return a boolean pandas.Series indicating stationary (low movement) periods.

    :param data: A pandas.DataFrame of acceleration time-series. It must contain
        at least columns `x,y,z` and the index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
    :type window: str, optional
    :param stdtol: Standard deviation under which the window is considered stationary.
        Defaults to 15 milligravity (0.015).
    :type stdtol: float, optional
    :return: Boolean pandas.Series indexed as `data` indicating stationary periods.
    :rtype: pandas.Series
    """

    def fn(data):
        return (
            (data[['x', 'y', 'z']]
             .rolling(window)
             .std()
             < stdtol)
            .all(axis=1)
        )

    stationary_indicator = pd.concat(
        chunker(
            data,
            chunksize='4h',
            leeway=window,
            fn=fn
        )
    )

    return stationary_indicator


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
    columns = [c for c in ['x', 'y', 'z', 'T'] if c in data.dtype.names]
    data = pd.DataFrame({c: data[c] for c in columns}, index=t, copy=False)

    return data
