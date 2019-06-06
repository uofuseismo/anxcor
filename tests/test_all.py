import unittest
import tests.test_processor_functions as processor_functions
import tests.test_integration as integration
import tests.test_irisbank as irisbank


def get_all_suites():
    suite1 = unittest.TestLoader().loadTestsFromModule(processor_functions)
    suite2 = unittest.TestLoader().loadTestsFromModule(irisbank)
    suite3 = unittest.TestLoader().loadTestsFromModule(integration)
    print()
    suites = unittest.TestSuite([suite1, suite2, suite3])
    return suites

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    tests = loader.discover('.')
    runner.run(tests)
