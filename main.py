import numpy as np

from plotting import make_plot

a = -5
c = -1
f = lambda x: -np.sin(x + 1) / x
make_plot(a, c, f)

a = -50
c = 1000
f = lambda x: x ** 2 - 4 * x + 5
df = lambda x: 2 * x - 4
make_plot(a, c, f)
make_plot(a, c, f, df)

a = -5
c = -1
f = lambda x: -np.sin(x + 1) / x
df = lambda x: (np.sin(x + 1) - x * np.cos(x + 1)) / x ** 2

make_plot(a, c, f)
make_plot(a, c, f, df)

a = 0.15
c = 2
f = lambda x: np.exp(x ** 2 - 4 * np.sin(x)) + x ** ((np.log(x) + 2) / x)
df = lambda x: np.exp(x ** 2 - 4 * np.sin(x)) * (2 * x - 4 * np.cos(x)) + (
        (np.log(x) + 2) / x ** 2 + np.log(x) * (1 / x ** 2 - (np.log(x) + 2) / x ** 2)) * x ** (
                       (np.log(x) + 2) / x)

make_plot(a, c, f)
make_plot(a, c, f, df)
