import numpy as np
from kernels import Kernel
k = Kernel()


def w(x1, x2, h):
    return k.uniform((x1-x2)/h)


class nadaraya_watson():
    def __init__(self, xdata, ydata, h):
        self.xdata = xdata
        self.ydata = ydata
        self.h = h
        self.s = []

        
        n = len(self.xdata)
        for i in range(n):
            norm = sum(
                [w(self.xdata[i], self.xdata[j], self.h) for j in range(n)]
            )
            if len(self.s) == 0:
                self.s = [w(self.xdata[i], self.xdata[j], self.h)/norm for j in range(n)]
            else:
                self.s = np.vstack(
                    [self.s, [w(self.xdata[i], self.xdata[j], self.h)/norm for j in range(n)]]
                )

    def fitted_values(self):
        return np.matmul(self.s,y)

    def predict(self, x):
        norm = sum([w(self.xdata[i], self.xdata[j], self.h) for j in range(n)])
        return [w(x, self.xdata[j], self.h)/norm for j in range(n)]

x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 2, 3, 4, 3, 2, 1]
n = nadaraya_watson(x, y, 1)
print(n.fitted_values())
