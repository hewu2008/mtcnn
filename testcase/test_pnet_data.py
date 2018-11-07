# encoding:utf8

import unittest

from prepare_data import gen_12net_data


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_a_run(self):
        gen_12net_data.check_data_path()
        gen_12net_data.read_annotation_file()

    def test_b_run(self):
        self.assertEqual(2, 2)


if __name__ == "__main__":
    unittest.main()
