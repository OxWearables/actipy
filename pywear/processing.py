import numpy as np
import pandas as pd
import bottleneck as bn
import scipy.stats as stats
import scipy.signal as signal
import statsmodels.api as sm


__all__ = ['lowpass', 'detect_nonwear', 'calibrate_gravity', 'get_stationary_indicator', 'resample']


def resample(data, sample_rate, dropna=False):
    """ Resample data to sample_rate. This uses simple nearest neighbor
    resampling, so be sure to use antialiasing filters if using a rate
    much lower than data's original rate. """

    info = {}

    # Round-up sample_rate if non-integer
    if isinstance(sample_rate, float) and not sample_rate.is_integer():
        print(f"Found non-integer sample_rate {sample_rate},", end=" ")
        sample_rate = np.ceil(sample_rate)
        print(f"rounded-up to {sample_rate}.")

    info['resampleRate'] = sample_rate
    info['numTicksBeforeResample'] = len(data)

    # Fix if time non-increasing (rarely occurs)
    if (data.index.to_series().diff() <= pd.Timedelta(0)).any():
        data = data[data.index.to_series()
                    .cummax()
                    .diff()
                    .fillna(pd.Timedelta(1))
                    > pd.Timedelta(0)]

    # Create a new index with intended sample_rate. Start and end times are
    # rounded to seconds so that the number of ticks (periods) is round
    start = data.index[0].round('S')
    end = data.index[-1].round('S')
    periods = int((end - start).total_seconds() * sample_rate + 1)  # +1 for the last tick
    new_index = pd.date_range(start, end, periods=periods, name='time')
    data = data.reindex(new_index,
                        method='nearest',
                        tolerance=pd.Timedelta('1s'),
                        limit=1)

    if dropna:
        data = data.dropna()

    info['numTicksAfterResample'] = len(data)

    return data, info


def lowpass(data, data_sample_rate, cutoff_rate=20):

    info = {}

    orig_index = data.index
    data, _ = resample(data, data_sample_rate, dropna=False)

    # Butter filter to remove high freq noise.
    # Default: 20Hz (most of human motion is under 20Hz)
    # Skip this if the Nyquist freq is too low
    if data_sample_rate / 2 > cutoff_rate:
        xyz = data[['x', 'y', 'z']].to_numpy()
        # Temporarily replace nans with 0s for butterfilt
        where_nan = bn.anynan(xyz, axis=1)
        xyz[where_nan] = 0
        xyz = butterfilt(xyz, cutoff_rate, fs=data_sample_rate, axis=0)
        # Now restore nans
        xyz[where_nan] = np.nan
        data[['x', 'y', 'z']] = xyz
        info['lowpassFilterOK'] = 1
        info['lowpassCutoff(Hz)'] = cutoff_rate
    else:
        print(f"Skipping lowpass filter: data sample rate {data_sample_rate} too low for cutoff rate {cutoff_rate}")
        info['lowpassFilterOK'] = 0

    data = data.reindex(orig_index,
                        method='nearest',
                        tolerance=pd.Timedelta('1s'),
                        limit=1)

    return data, info


def detect_nonwear(data, patience='90m', stationary_indicator=None, drop=False):
    """ Detect nonwear episodes based on long durations of no movement """

    info = {}

    if stationary_indicator is None:
        stationary_indicator = get_stationary_indicator(data)

    group = ((stationary_indicator != stationary_indicator.shift(1))
             .cumsum()
             .where(stationary_indicator))
    stationary_len = (group.groupby(group, dropna=True)
                           .apply(lambda g: g.index[-1] - g.index[0]))
    nonwear_len = stationary_len[stationary_len > pd.Timedelta(patience)]

    info['numNonWearEpisodes'] = len(nonwear_len)
    info['nonwearOverall(days)'] = nonwear_len.sum().total_seconds() / (60 * 60 * 24)

    # Flag nonwear
    nonwhere_indicator = group.isin(nonwear_len.index)
    if drop:
        data = data[~nonwhere_indicator]
    else:
        data = data.mask(nonwhere_indicator)

    return data, info


def calibrate_gravity(data, calib_cube=0.3, stationary_indicator=None):
    """ Gravity calibration method of https://pubmed.ncbi.nlm.nih.gov/25103964/ """

    info = {}

    if stationary_indicator is None:
        stationary_indicator = get_stationary_indicator(data)

    # The paper uses 10sec averages instead of the raw ticks.
    # This reduces computational cost. Also reduces influence of outliers.
    stationary_data = (data[stationary_indicator]
                       .resample('10s')
                       .mean()
                       .dropna())

    hasT = 'T' in stationary_data

    xyz = stationary_data[['x', 'y', 'z']].to_numpy()
    # Remove any nonzero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]
    if hasT:
        T = stationary_data['T'].to_numpy()
        # No need to follow the paper that uses a reference temperature
        # Tref = np.mean(T)
        # dT = T - Tref
        dT = T
        dT = dT[nonzero]
    del stationary_data
    del nonzero

    intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
    slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
    best_intercept = np.copy(intercept)
    best_slope = np.copy(slope)

    if hasT:
        slopeT = np.array([0.0, 0.0, 0.0], dtype=dT.dtype)
        best_slopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr - target, axis=1)
    # err = np.sqrt(np.mean(np.square(errors)))  # root mean square error (RMSE)
    err = np.median(errors)  # MAE more robust than RMSE. This is different from the paper
    init_err = err
    best_err = 1e16

    MAXITER = 1000
    IMPROV_TOL = 0.0001
    ERR_TOL = 0.01

    info['calibErrorBefore(mg)'] = init_err * 1000

    # Check that we have sufficiently uniformly distributed points:
    # need at least one point outside each face of the cube
    if (np.max(xyz, axis=0) < calib_cube).any() \
            or (np.min(xyz, axis=0) > -calib_cube).any():
        info['calibOK'] = 0
        info['calibErrorAfter(mg)'] = init_err * 1000

        return data, info

    for it in range(MAXITER):

        # Weighting. Outliers are zeroed out
        # This is different from the paper
        maxerr = err + 1.5 * stats.iqr(errors)
        weights = np.maximum(1 - errors / maxerr, 0)

        # Optimize params for each axis
        for k in range(3):

            inp = curr[:, k]
            out = target[:, k]
            if hasT:
                inp = np.column_stack((inp, dT))
            inp = sm.add_constant(inp, prepend=True)  # add intercept term
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
            curr = curr + (dT[:, None] * slopeT)
        target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

        # Update errors
        errors = np.linalg.norm(curr - target, axis=1)
        # err = np.sqrt(np.mean(np.square(errors)))
        err = np.median(errors)
        err_improv = (best_err - err) / best_err

        if err < best_err:
            best_intercept = np.copy(intercept)
            best_slope = np.copy(slope)
            if hasT:
                best_slopeT = np.copy(slopeT)
            best_err = err
        if err_improv < IMPROV_TOL:
            break

    info['calibErrorAfter(mg)'] = best_err * 1000

    if best_err > ERR_TOL:
        info['calibOK'] = 0

        return data, info

    else:
        # Calibrate
        data = data.copy()
        data[['x', 'y', 'z']] = (best_intercept
                                 + best_slope * data[['x', 'y', 'z']].to_numpy())
        if hasT:
            data[['x', 'y', 'z']] = (data[['x', 'y', 'z']]
                                     # + best_slopeT * (data['T'].to_numpy()[:,None]-Tref))
                                     + best_slopeT * (data['T'].to_numpy()[:, None]))

        info['calibOK'] = 1
        info['calibNumIters'] = it + 1
        info['calibNumSamples'] = len(xyz)
        info['calibxIntercept'] = best_intercept[0]
        info['calibyIntercept'] = best_intercept[1]
        info['calibzIntercept'] = best_intercept[2]
        info['calibxSlope'] = best_slope[0]
        info['calibySlope'] = best_slope[1]
        info['calibzSlope'] = best_slope[2]
        if hasT:
            info['calibxSlopeT'] = best_slopeT[0]
            info['calibySlopeT'] = best_slopeT[1]
            info['calibzSlopeT'] = best_slopeT[2]
            # info['calibTref'] = Tref

    return data, info


def get_stationary_indicator(data, window='10s', stdtol=15 / 1000):
    """ Return a boolean column indicating stationary points """

    # What happens if there are NaNs?
    # Ans: It evaluates to False so we're good
    stationary_indicator = ((data[['x', 'y', 'z']]
                            .rolling(window)
                            .std()
                            < stdtol)
                            .all(axis=1))

    return stationary_indicator


def butterfilt(x, cutoffs, fs, order=8, axis=0):
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
