import logging
import unittest

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s")


class TestPNet(unittest.TestCase):
    def setUp(self):
        logging.debug("setUp")

    def tearDown(self):
        logging.debug("tearDown")

    def test_foo(self):
        self.assertTrue(True)


if __name__ == "__main__":
    logging.debug("hello world")
