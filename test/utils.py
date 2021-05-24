import json
import numpy as np
from itertools import product


PROCESS_ARGS = [
    {'name': 'resample',
     'arg': 'resample_uniform',
     'val': [True, False]},
    {'name': 'noise',
     'arg': 'remove_noise',
     'val': [True, False]},
    {'name': 'calib',
     'arg': 'calibrate_gravity',
     'val': [True, False]},
    {'name': 'nonwear',
     'arg': 'detect_nonwear',
     'val': [True, False]},
]


def create_tests(process_args=PROCESS_ARGS):
    """ Create tests from combinations of processing args """

    names = [_['name'] for _ in PROCESS_ARGS]
    args = [_['arg'] for _ in PROCESS_ARGS]
    vals = [_['val'] for _ in PROCESS_ARGS]
    valcombos = list(product(*vals))
    testparams = [dict(zip(args, valcombo)) for valcombo in valcombos]
    testnames = [list(zip(names, valcombo)) for valcombo in valcombos]
    testnames = ['_'.join(f"{arg}={val}" for arg, val in testname) for testname in testnames]

    return dict(zip(testnames, testparams))


def np2py(x):
    """ Convert numpy types to native as json can't deal with them """

    if isinstance(x, np.generic):
        return x.item()
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        else:
            return x.tolist()
    return x


def dict_np2py(adict):
    """ np2py every value of a dict """

    return {k: np2py(v) for k, v in adict.items()}


def save_dict2json(adict, outfile):
    """ Save a dictionary to disk in JSON format """

    with open(outfile, 'w') as f:
        json.dump(adict, f, default=np2py, indent=4)

    return
