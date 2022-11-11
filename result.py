import numpy as np


class Result:

    def __init__(self):
        self.x = None
        self.fx = None
        self.dfx = None
        self.points = np.array([])
        self.values = np.array([])

    def set(self, x, fx, dfx=None):
        self.x = x
        self.fx = fx
        self.dfx = dfx

    def add_point(self, x, y):
        self.points = np.append(self.points, x)
        self.values = np.append(self.values, y)
