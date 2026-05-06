import numpy as np
from scipy.special import beta, betainc
import torch

class DirichletConditionalFlow:
    def __init__(self, K, T, alpha_min=1, alpha_spacing=0.01, scale=1.0):
        self.alpha_max = 1 - ((np.log(np.array([1.0]) - (T-1)/T)) * scale)
        self.alpha_spacing = alpha_spacing
        self.alphas = np.arange(alpha_min, self.alpha_max + self.alpha_spacing, self.alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / self.alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha, eps=10e-12):
        out1 = beta(alpha, self.K - 1)
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), 0)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1) + eps), 0)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))-1]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return final