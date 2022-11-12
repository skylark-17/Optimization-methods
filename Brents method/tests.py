import unittest

import numpy as np

from brents_method import one_dim, one_dim_with_derivative


class MyTestCase(unittest.TestCase):
    def test_simple_func(self):
        a = -5
        c = 8
        f = lambda x: x ** 2 - 4 * x + 5
        df = lambda x: 2 * x - 4
        res = one_dim(a, c, f)
        res_with_derivative = one_dim_with_derivative(a, c, f, df)
        self.assertAlmostEqual(res.x, 2.0, delta=1e-3)
        self.assertAlmostEqual(res_with_derivative.x, 2.0, delta=1e-3)

    def test_sin(self):
        a = -np.pi + 2 * np.pi
        c = 0 + 2 * np.pi
        f = lambda x: np.sin(x)
        df = lambda x: np.cos(x)
        self.assertAlmostEqual(one_dim(a, c, f).x, -np.pi / 2 + 2 * np.pi, delta=1e-3)
        self.assertAlmostEqual(one_dim_with_derivative(a, c, f, df).x, -np.pi / 2 + 2 * np.pi, delta=1e-3)

    def test_cos(self):
        a = np.pi / 2 - 2 * np.pi
        c = np.pi * 3 / 2 - 2 * np.pi
        f = lambda x: np.cos(x)
        df = lambda x: -np.sin(x)
        self.assertAlmostEqual(one_dim(a, c, f).x, np.pi - 2 * np.pi, delta=1e-3)
        self.assertAlmostEqual(one_dim_with_derivative(a, c, f, df).x, np.pi - 2 * np.pi, delta=1e-3)

    def test_intricate_f_1(self):
        a = -5
        c = -1
        f = lambda x: -np.sin(x + 1) / x
        df = lambda x: (np.sin(x + 1) - x * np.cos(x + 1)) / x ** 2
        self.assertAlmostEqual(one_dim(a, c, f).x, -2.13227, delta=1e-3)
        self.assertAlmostEqual(one_dim_with_derivative(a, c, f, df).x, -2.13227, delta=1e-3)

    def test_intricate_f_2(self):
        a = 0.1
        c = 10
        f = lambda x: np.exp(x ** 2 - 4 * np.sin(x)) + x ** ((np.log(x) + 2) / x)
        df = lambda x: np.exp(x ** 2 - 4 * np.sin(x)) * (2 * x - 4 * np.cos(x)) + (
                (np.log(x) + 2) / x ** 2 + np.log(x) * (1 / x ** 2 - (np.log(x) + 2) / x ** 2)) * x ** (
                               (np.log(x) + 2) / x)
        self.assertAlmostEqual(one_dim(a, c, f).x, 0.416158, delta=1e-3)
        self.assertAlmostEqual(one_dim_with_derivative(a, c, f, df).x, 0.416158, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
