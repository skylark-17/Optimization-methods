import matplotlib.pyplot as plt
import numpy as np

from brents_method import one_dim, one_dim_with_derivative


def make_plot(a, c, f, df=None):
    if df is None:
        res = one_dim(a, c, f)
    else:
        res = one_dim_with_derivative(a, c, f, df)
    points = res.points
    values = res.values
    x0 = res.x
    fx0 = res.fx
    grid = np.linspace(a, c, 1000)
    plt.plot(grid, f(grid), color='blue')
    plt.plot(points, values, color='red', alpha=0.3)
    plt.scatter(points, values, color='black', alpha=0.3)
    plt.title("Brents method" if df is None else "Brents method with derivative")
    plt.xlabel(f'$x$')
    plt.ylabel(f'$f(x)$')
    plt.show()
