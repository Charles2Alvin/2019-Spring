import numpy as np
from math import sqrt, pi, exp


class Gaussian(object):
    """Model univariate gaussian"""

    def __init__(self, mu, sigma):
        # Mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    # probability density function
    def pdf(self, num):
        u = (num - self.mu) / (abs(self.sigma))
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y


class MultiGaussian(object):
    """docstring for MultiGaussian"""

    def __init__(self, mu, sigma):
        super(MultiGaussian, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def pdf(self, num):
        diff = num - self.mu
        inv = np.linalg.inv(self.sigma)
        det = np.linalg.det(self.sigma)
        expo = (diff.T.dot(inv)).dot(diff)
        k = self.mu.shape[0]
        try:
            y = (1 / sqrt(((2 * pi) ** k) * det)) * exp(- expo / 2)
        except ValueError:
            print("det", det)
            y = 0
        finally:
            return y





