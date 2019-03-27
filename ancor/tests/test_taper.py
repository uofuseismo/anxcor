import unittest

class TestTaper(unittest.TestCase):

    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_zero_at_ends(self):
        '''

        Test that the start and end of the file are within some
        margin of error of zero.

        :return:
        '''

        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

    def test_not_zero_in_middle(self):
        '''

        Test that the middle of the file is not zero (unless it was to begin with)

        :return:
        '''

        source = 1
        target = 0
        self.assertAlmostEqual(source,target,5,'not implemented')

    def test_slope_max_min(self):
        '''
        Test for a positive slope at the start of the file for the MAXIMUM values over a given interval
        Test for a negative slope at the end of the for the MAXIMUM values over a given interval
        Test for a negative slope at the start of the file for the MAXIMUM values
        Test for a positive slope at the end of the file for the MINIMUM values

        :return:
        '''

        source = 1
        target = 0
        self.assertAlmostEqual(source, target, 5, 'not implemented')

    def test_signal_unaltered(self):
        '''
        
        Test that the signal is unaltered - Correlate processed with source?

        :return:
        '''
        source = 1
        target = 0
        self.assertAlmostEqual(source, target, 5, 'not implemented')
