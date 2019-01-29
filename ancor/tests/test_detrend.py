import unittest

class TestDetrend(unittest.TestCase):




    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detrend_whole(self):

        #Test for removing the slope over whole file
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

    def test_detrend_part(self):

        #Test for removing trends within the file
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

    def test_signal_unaltered(self):

        #Test that the rest of the signal hasn't been changed
        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

