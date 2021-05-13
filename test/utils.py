import json
import numpy as np


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

    return {k: np2py(v) for k,v in adict.items()}


def save_dict2json(adict, outfile):
    """ Save a dictionary to disk in JSON format """

    with open(outfile, 'w') as f:
        json.dump(adict, f, default=np2py, indent=4)

    return
