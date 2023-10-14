from abc import ABC, abstractmethod
import numpy as np
from numpy.random import noncentral_chisquare


class Process(ABC):
    """Abstract base class for stock price processes."""

    def __init__(self, x0, y0, r):
        self.x0 = x0
        self.y0 = y0
        self.r = r

    @abstractmethod
    def sample_vol(self, yt, dt):
        """Abstract method to sample volatility."""
        pass


class GBM(Process):
    """Geometric Brownian Motion stock price process."""

    def __init__(self, x0, y0, r, sigma):
        super().__init__(x0, y0, r)
        self.sigma = sigma

    def sample_vol(self, yt, dt):
        """Sample volatility for GBM. Constant volatility."""
        return self.sigma, self.sigma


class Heston(Process):
    """Heston model for stock price with stochastic volatility."""

    def __init__(self, x0, y0, r, alpha, nu, sigma):
        super().__init__(x0, y0, r)
        self.alpha = alpha
        self.nu = nu
        self.sigma = sigma

    def sample_vol(self, yt, dt):
        """Sample non-central chi-squared distribution."""
        c = 2 * self.alpha / ((1 - np.exp(-self.alpha * dt)) * self.sigma ** 2)
        k = 4 * self.alpha * self.nu / self.sigma ** 2
        lam = 2 * c * yt * np.exp(-self.alpha * dt)
        ncsq = noncentral_chisquare(df=k, nonc=lam)
        vol = ncsq / (2 * c)
        return vol, np.sqrt(vol)


class FourOverTwo(Process):
    """4/2 model for stock price with stochastic volatility."""

    def __init__(self, x0, y0, r, alpha, nu, sigma, a, b):
        super().__init__(x0, y0, r)
        self.alpha = alpha
        self.nu = nu
        self.sigma = sigma
        self.a = a
        self.b = b

    def sample_vol(self, yt, dt):
        """Sample non-central chi-squared distribution."""
        c = 2 * self.alpha / ((1 - np.exp(-self.alpha * dt)) * self.sigma ** 2)
        k = 4 * self.alpha * self.nu / self.sigma ** 2
        lam = 2 * c * yt * np.exp(-self.alpha * dt)
        ncsq = noncentral_chisquare(df=k, nonc=lam)
        vol = ncsq / (2 * c)
        diff_coef = self.a * np.sqrt(vol) + self.b / np.sqrt(vol)
        return vol, diff_coef
