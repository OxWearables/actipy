import os
import unittest
import json

import pywear
from test import utils

DATA = 'data/sample_genea.bin.gz'
OUTPUTS = 'test/outputs/genea/'
TESTS = utils.create_tests()


class TestGenea(unittest.TestCase):

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

    def test_process(self):

        for testname, testparam in TESTS.items():

            with self.subTest("Testing processing...", **testparam):

                _, info = pywear.reader.process(TestGenea.data,
                                                TestGenea.info,
                                                **testparam)

                with open(os.path.join(OUTPUTS, testname + '.json')) as f:
                    _info = json.load(f)
                self.assertDictEqual(info, _info)
