import unittest

from .synthetic_trace_factory import linear_noise_ramp

class TestDemean(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_demean(self):
        '''
        Test for removing the mean for the whole file

        Subtract target file from processed file: should equal zero

        :return:



        '''

        source = 1
        target = 0
        self.assertAlmostEqual(source, target, 5, 'not implemented')

    def test_signal_unaltered(self):
        '''
        Test for ensuring the signal is unaltered

        subtract source file from processed file: should equal a constant (can be zero).
        The result will be an array; if the array varies 'significantly' over its length, fail
        will have to determine what is 'significantly'

        :return:
        '''

        source = 1
        target = 0
        self.assertAlmostEqual(source, target, 5, 'not implemented')

    def test_trend_removed(self):
        source = 1
        target = 0
        self.assertAlmostEqual(source, target, 5, 'not implemented')

    def test_edge_effects(self):
        source = 1
        target = 0
        self.assertAlmostEqual(source, target, 5, 'not implemented')

if __name__ == '__main__':
    unittest.main()