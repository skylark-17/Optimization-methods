import matplotlib.pyplot as plt
import numpy as np

from brents_method import one_dim

# a = -5
# c = -1
# f = lambda x: -np.sin(x + 1) / x
# make_plot(a, c, f)
#
# a = -50
# c = 1000
# f = lambda x: x ** 2 - 4 * x + 5
# df = lambda x: 2 * x - 4
# make_plot(a, c, f)
# make_plot(a, c, f, df)
#
# a = -5
# c = -1
# f = lambda x: -np.sin(x + 1) / x
# df = lambda x: (np.sin(x + 1) - x * np.cos(x + 1)) / x ** 2
#
# make_plot(a, c, f)
# make_plot(a, c, f, df)
#
# a = 0.15
# c = 2
# f = lambda x: np.exp(x ** 2 - 4 * np.sin(x)) + x ** ((np.log(x) + 2) / x)
# df = lambda x: np.exp(x ** 2 - 4 * np.sin(x)) * (2 * x - 4 * np.cos(x)) + (
#         (np.log(x) + 2) / x ** 2 + np.log(x) * (1 / x ** 2 - (np.log(x) + 2) / x ** 2)) * x ** (
#                        (np.log(x) + 2) / x)
#
# make_plot(a, c, f)
# make_plot(a, c, f, df)

a = -5
c = -1
f = lambda x: -np.sin(x + 1) / x
df = lambda x: (np.sin(x + 1) - x * np.cos(x + 1)) / x ** 2

times = []
iterations = []
func_computations = []

N = 15

for i in range(N):
    res = one_dim(a, c, f, 10 ** (-i))
    times.append(res.time)
    iterations.append(res.iterations)
    func_computations.append(res.func_computations)
    # make_plot(a, c, f,None, 10 ** (-i))

plt.figure(figsize=(12, 8))
plt.plot(range(N), iterations)
plt.plot(range(N), func_computations)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(range(N), times)
plt.show()
