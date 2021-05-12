import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import functools


__all__ = ['resample', 'detect_nonwear', 'calibrate_gravity', 'get_stationary_indicator']


def timer(msg):
    """Print runtime of the decorated function"""
    def inner(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            before = time.perf_counter()
            print(msg, end=" ", flush=True)
            value = func(*args, **kwargs)
            after = time.perf_counter()
            runtime = after - before
            print(f"Done! ({runtime:.2f}s)")
            return value
        return wrapper_timer
    return inner


@timer(msg="Resampling... ")
def resample(data, sample_rate):
    """ Resample data to sample_rate. This uses simple nearest neighbor
    resampling, so should only be used for sample_rate near the data's
    original sample rate. For sample_rate that is far from the original
    rate, resampling with antialiasing filters should be done instead. """

    info = {}

    info['numTicksBeforeResample'] = len(data)

    # Create a new index with intended sample_rate
    # Start and end times are rounded to seconds so that the number of ticks
    # (periods) can be roundly derived from sample_rate
    start = data.index[0].round('S')
    end = data.index[-1].round('S')
    periods = int((end - start).total_seconds() * sample_rate + 1)  # +1 for the last tick
    new_index = pd.date_range(start, end, periods=periods, name='time')
    data = data.reindex(new_index,
                        method='nearest',
                        tolerance=pd.Timedelta('1s'),
                        limit=1).dropna()

    info['numTicksAfterResample'] = len(data)

    return data, info


@timer(msg="Nonwear detection...")
def detect_nonwear(data, patience='90m', stationary_indicator=None, drop=True):
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
    info['nonwearOverall(days)'] = nonwear_len.sum().total_seconds() / (60*60*24)

    # Flag nonwear
    nonwhere_indicator = group.isin(nonwear_len.index)
    if drop:
        data = data[~nonwhere_indicator]
    else:
        data = data.mask(nonwhere_indicator)

    return data, info


@timer(msg="Gravity calibration...")
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

    xyz = stationary_data[['x','y','z']].to_numpy()
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

    intercept = np.array([0.0, 0.0, 0.0])
    slope = np.array([1.0, 1.0, 1.0])
    best_intercept = np.copy(intercept)
    best_slope = np.copy(slope)

    if hasT:
        slopeT = np.array([0.0, 0.0, 0.0])
        best_slopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr-target, axis=1)
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
        maxerr = err + 1.5*stats.iqr(errors)
        weights = np.maximum(1-errors/maxerr, 0)

        # Optimize params for each axis
        for k in range(3):

            inp = curr[:,k]
            out = target[:,k]
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
            curr = curr + (dT[:,None] * slopeT)
        target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

        # Update errors
        errors = np.linalg.norm(curr-target, axis=1)
        # err = np.sqrt(np.mean(np.square(errors)))
        err = np.median(errors)
        err_improv = (best_err-err)/best_err

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
        data[['x','y','z']] = (best_intercept
                            + best_slope * data[['x','y','z']].to_numpy())
        if hasT:
            data[['x','y','z']] = (data[['x','y','z']]
                            # + best_slopeT * (data['T'].to_numpy()[:,None]-Tref))
                            + best_slopeT * (data['T'].to_numpy()[:,None]))

        info['calibOK'] = 1
        info['calibNumIters'] = it+1
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


@timer(msg="Getting stationary points...")
def get_stationary_indicator(data, window='10s', stdtol=15/1000):
    """ Return a boolean column indicating stationary points """

    stationary_indicator = ((data[['x','y','z']]
                            .rolling(window)
                            .std()
                            < stdtol
                             ).all(axis=1))

    return stationary_indicator
