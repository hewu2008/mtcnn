# encoding:utf8

import unittest


class MyTest(unittest.TestCase):
    def setUp(self):
        print('222222')

    def tearDown(selfs):
        print('1111')

    def test_a_run(self):
        self.assertEqual(1, 1)

    def test_b_run(self):
        self.assertEqual(2, 2)


if __name__ == "__main__":
    unittest.main()
