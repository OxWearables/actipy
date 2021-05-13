import os
import unittest
import json

import pywear
from test import utils

DATA = 'data/sample.cwa.gz'
OUTPUTS = 'test/outputs/axivity/'


class TestAxivity(unittest.TestCase):

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        data, info = pywear.read_device(DATA)
        cls.data = data
        cls.info = info

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.info

    def test_read(self):

        _, info = pywear.read_device(DATA)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_read.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_resample(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         resample_uniform=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_resample.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_calib(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         calibrate_gravity=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_calib.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_nonwear(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         detect_nonwear=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_nonwear.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_resample_calib(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         resample_uniform=True,
                                         calibrate_gravity=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_resample_calib.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_resample_nonwear(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         resample_uniform=True,
                                         detect_nonwear=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_resample_nonwear.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_calib_nonwear(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         calibrate_gravity=True,
                                         detect_nonwear=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_calib_nonwear.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)

    def test_resample_calib_nonwear(self):

        _, info = pywear.reader._process(TestAxivity.data,
                                         TestAxivity.info,
                                         resample_uniform=True,
                                         calibrate_gravity=True,
                                         detect_nonwear=True)
        info = utils.dict_np2py(info)

        with open(os.path.join(OUTPUTS, 'test_resample_calib_nonwear.json')) as f:
            _info = json.load(f)
        self.assertDictEqual(info, _info)
