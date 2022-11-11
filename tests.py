import unittest

import brents_method


class MyTestCase(unittest.TestCase):
    def test_simple_func(self):
        a = -5
        c = 8
        f = lambda x: x ** 2 - 4 * x + 5
        self.assertAlmostEqual(brents_method.one_dim(a, c, f), 2.0, delta=1e-3)

    def test_simple_func_with_derivative(self):
        a = -5
        c = 8
        f = lambda x: x ** 2 - 4 * x + 5
        df = lambda x: 2 * x - 4
        self.assertAlmostEqual(brents_method.one_dim_with_derivative(a, c, f, df), 2.0, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
