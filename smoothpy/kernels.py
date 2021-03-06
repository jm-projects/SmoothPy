import numpy as np


class Kernel:
    def __init__(self):
        pass

    def uniform(self, x):
        return 0.5 if abs(x) <= 1 else 0

    def epanechnikov(self, x):
        if abs(x) > 1:
            return 0
        else:
            e = (3/4)*(1-x**2)
            return e

    def gaussian(self, x):
        k = 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)
        return k
