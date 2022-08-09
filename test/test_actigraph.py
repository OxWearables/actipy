import os
import unittest
import json

import actipy
from test import utils

DATA = 'data/sample_actigraph.gt3x'
OUTPUTS = 'test/outputs/actigraph/'
TESTS = utils.create_tests()


class TestActigraph(unittest.TestCase):

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

                _, info = actipy.process(TestActigraph.data,
                                         TestActigraph.info['SampleRate'],
                                         **testparam)

                with open(os.path.join(OUTPUTS, testname + '.json')) as f:
                    _info = json.load(f)

                for (_k, _v), (k, v) in zip(_info.items(), info.items()):
                    self.assertAlmostEqual(_v, v, places=5)
