import unittest
import test_processor_functions as processor_functions
import test_integration as integration
import test_irisbank as irisbank


def get_all_suites():
    suite1 = unittest.TestLoader().loadTestsFromModule(processor_functions)
    suite2 = unittest.TestLoader().loadTestsFromModule(irisbank)
    suite3 = unittest.TestLoader().loadTestsFromModule(integration)
    print()
    suites = unittest.TestSuite([suite1, suite2, suite3])
    return suites

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    tests = loader.discover('.')
    runner.run(tests)
