import os
import unittest
import json

import actipy
from test import utils

DATA = 'data/sample_genea.bin.gz'
OUTPUTS = 'test/outputs/genea/'
TESTS = utils.create_tests()


class TestGenea(unittest.TestCase):

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        data, info = actipy.read_device(DATA, lowpass_hz=None,
                                        calibrate_gravity=False,
                                        detect_nonwear=False,
                                        resample_hz=None)
        cls.data = data
        cls.info = info

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.info

    def test_process(self):

        for testname, testparam in TESTS.items():

            with self.subTest("Testing processing...", **testparam):

                _, info = actipy.process(TestGenea.data,
                                         TestGenea.info['SampleRate'],
                                         **testparam)

                with open(os.path.join(OUTPUTS, testname + '.json')) as f:
                    _info = json.load(f)
                self.assertDictEqual(info, _info)
