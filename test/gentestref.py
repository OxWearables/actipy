""" This is for generating the reference test files.
It runs the same things as the tests but without any checks.
Run with `python -m test.gentestref` from project root.
"""
import os
import pywear
from test.utils import save_dict2json


TESTS = [
    {'input_file': 'data/sample.cwa.gz',
     'outdir': 'test/outputs/axivity'},
    {'input_file': 'data/sample_genea.bin.gz',
     'outdir': 'test/outputs/genea'},
    {'input_file': 'data/sample_actigraph.gt3x',
     'outdir': 'test/outputs/actigraph'}
]


def main():

    for test in TESTS:
        print("Running:", test)
        gentestref(test['input_file'], test['outdir'])

    return


def gentestref(input_file, outdir):

    data, info_read = pywear.read_device(input_file)
    save_dict2json(info_read, os.path.join(outdir, 'test_read.json'))

    _, info_resample = pywear.reader._process(data, info_read,
                                              resample_uniform=True)
    save_dict2json(info_resample, os.path.join(outdir, 'test_resample.json'))

    _, info_calib = pywear.reader._process(data, info_read,
                                           calibrate_gravity=True)
    save_dict2json(info_calib, os.path.join(outdir, 'test_calib.json'))

    _, info_nonwear = pywear.reader._process(data, info_read,
                                             detect_nonwear=True)
    save_dict2json(info_nonwear, os.path.join(outdir, 'test_nonwear.json'))

    _, info_resample_calib = pywear.reader._process(data, info_read,
                                                    resample_uniform=True,
                                                    calibrate_gravity=True)
    save_dict2json(info_resample_calib, os.path.join(outdir, 'test_resample_calib.json'))

    _, info_resample_nonwear = pywear.reader._process(data, info_read,
                                                      resample_uniform=True,
                                                      detect_nonwear=True)
    save_dict2json(info_resample_nonwear, os.path.join(outdir, 'test_resample_nonwear.json'))

    _, info_calib_nonwear = pywear.reader._process(data, info_read,
                                                   calibrate_gravity=True,
                                                   detect_nonwear=True)
    save_dict2json(info_calib_nonwear, os.path.join(outdir, 'test_calib_nonwear.json'))

    _, info_resample_calib_nonwear = pywear.reader._process(data, info_read,
                                                            resample_uniform=True,
                                                            calibrate_gravity=True,
                                                            detect_nonwear=True)
    save_dict2json(info_resample_calib_nonwear, os.path.join(outdir, 'test_resample_calib_nonwear.json'))


if __name__ == '__main__':
    main()
