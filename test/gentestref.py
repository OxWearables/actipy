""" Generate the reference test output files.
It runs the tests without any checks and saving the outputs.
Run with `python -m test.gentestref` from project root.
"""
import os
import actipy
from test import utils


TEST_DEVICES = [
    {'file': 'data/sample.cwa.gz',
     'outdir': 'test/outputs/axivity'},
    {'file': 'data/sample_genea.bin.gz',
     'outdir': 'test/outputs/genea'},
    {'file': 'data/sample_actigraph.gt3x',
     'outdir': 'test/outputs/actigraph'}
]

TESTS = utils.create_tests()


def main():

    for device in TEST_DEVICES:
        print("Running:", device)
        gentestref(device['file'], device['outdir'], TESTS)

    return


def gentestref(input_file, outdir, tests):

    # Minimal file reading test with no other args
    data, info_read = actipy.read_device(input_file,
                                         lowpass_hz=None,
                                         calibrate_gravity=False,
                                         detect_nonwear=False,
                                         resample_hz=None)
    utils.save_dict2json(info_read, os.path.join(outdir, 'read.json'))

    for testname, testparam in tests.items():
        print("Running:", testname)
        _, info_test = actipy.process(data, info_read['SampleRate'], **testparam)
        utils.save_dict2json(info_test, os.path.join(outdir, testname + '.json'))


if __name__ == '__main__':
    main()
