import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb
from typing import Sequence, Tuple
from mpmath import beta as mp_beta, digamma as mp_digamma
from scipy.special import softmax
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sympy import EulerGamma

DT = torch.float64
torch.set_default_dtype(DT)

def get_bernstein_to_standard_matrix(degree):
    bernstein_to_standard_matrix = np.zeros((degree+1, degree+1))
    for v in range(degree+1):
        for l in range(v, degree+1):
            bernstein_to_standard_matrix[l, v] = comb(degree, l) * comb(l, v) * (-1)**(l-v)
    return bernstein_to_standard_matrix

class SkewedBLogistic:
    """
    Modify BLogistic use Generalized logistic distribution type I to make it skewed.
    """

    def __init__(self, degree: int, device: torch.device = None):
        self.degree = degree
        self.device = device if device is not None else torch.device('cpu')
        self.bernstein_to_standard_matrix_torch = torch.tensor(get_bernstein_to_standard_matrix(degree), dtype=DT, device=self.device)
        self.euler_gamma = torch.tensor(EulerGamma, dtype=DT, device=self.device)
    
    def get_mus(self, skewness):
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        mus = (self.euler_gamma + torch.digamma(1 + (skewness * (powers + 1)))) / (powers + 1) - 1 / (skewness * (powers + 1) ** 2)
        return mus

    def _process_input(self, xs, coeffs, raw_scale, raw_skewness):
        normalized_coeffs = torch.softmax(coeffs, dim=0) * (self.degree + 1)
        standard_coeffs = self.bernstein_to_standard_matrix_torch @ normalized_coeffs
        scale = torch.nn.functional.softplus(raw_scale)
        skewness = torch.nn.functional.softplus(raw_skewness)

        mus = self.get_mus(skewness)
        mean = torch.dot(mus, standard_coeffs)

        shifted_xs = (xs + mean) / scale
        Fx = (1.0 + torch.exp(-shifted_xs)) ** -skewness
        return shifted_xs, standard_coeffs, Fx, scale, skewness
    
    def logpdf(self, xs, coeffs, raw_scale, raw_skewness):
        shifted_xs, standard_coeffs, Fx, scale, skewness = self._process_input(xs, coeffs, raw_scale, raw_skewness)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)

        log_fprime = torch.log(skewness) - shifted_xs - (skewness + 1) * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        u_p = torch.pow(Fx.unsqueeze(-1), powers)
        poly = torch.sum(u_p * standard_coeffs, dim=-1)
        return torch.log(poly) + log_fprime

    def _process_input_vectorized(self, xs, coeffs, raw_scale, raw_skewness):
        normalized_coeffs = torch.softmax(coeffs, dim=1) * (self.degree + 1)
        standard_coeffs = (self.bernstein_to_standard_matrix_torch @ normalized_coeffs.T).T
        scale = torch.nn.functional.softplus(raw_scale).reshape(-1, 1)
        skewness = torch.nn.functional.softplus(raw_skewness).reshape(-1, 1)

        mus = self.get_mus(skewness)
        mean = torch.sum(standard_coeffs * mus, dim=1)
        mean = mean.reshape(-1, 1)

        shifted_xs = (xs + mean) / scale
        Fx = (1.0 + torch.exp(-shifted_xs)) ** -skewness
        return shifted_xs, standard_coeffs, Fx, scale, skewness

    def logpdf_vectorized(self, xs, coeffs, raw_scale, raw_skewness):
        shifted_xs, standard_coeffs, Fx, scale, skewness = self._process_input_vectorized(xs, coeffs, raw_scale, raw_skewness)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        log_fprime = torch.log(skewness) - shifted_xs - (skewness + 1) * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        u_p = torch.pow(Fx, powers.reshape(1, -1))
        poly = torch.sum(u_p * standard_coeffs, dim=-1).reshape(-1, 1)
        return torch.log(poly) + log_fprime
    
    def pdf(self, xs, coeffs, raw_scale, raw_skewness):
        return torch.exp(self.logpdf(xs, coeffs, raw_scale, raw_skewness))

    def cdf(self, xs, coeffs, raw_scale, raw_skewness):
        shifted_xs, standard_coeffs, Fx, scale, skewness = self._process_input(xs, coeffs, raw_scale, raw_skewness)

        powers = torch.arange(0, self.degree + 1, dtype=DT, device=self.device) + 1
        cdf_terms = torch.pow(Fx.unsqueeze(-1), powers) / powers
        cdf_val = torch.sum(cdf_terms * standard_coeffs, dim=-1)
        return cdf_val

class BLogistic:
    """
    Bernstein Logistic distribution class that delegates to SkewedBLogistic with skew parameter set to 0.
    """
    def __init__(self, degree: int, device: torch.device = None):
        self.degree = degree
        self.device = device if device is not None else torch.device('cpu')
        self._skewed = SkewedBLogistic(degree=degree, device=self.device)
        # For compatibility, store a dummy scale and skew parameter (skew=0)
        self._raw_skew = torch.tensor(0.0, dtype=DT, device=self.device)

    def logpdf(self, xs, coeffs, raw_scale):
        # skewness=0, so SkewedBLogistic reduces to symmetric
        return self._skewed.logpdf(xs, coeffs, raw_scale, self._raw_skew)

    def pdf(self, xs, coeffs, raw_scale):
        return self._skewed.pdf(xs, coeffs, raw_scale, self._raw_skew)

    def cdf(self, xs, coeffs, raw_scale):
        return self._skewed.cdf(xs, coeffs, raw_scale, self._raw_skew)

def get_ppf(blogistic, params, ps, max_scale=20, num_steps=100):
    scale = torch.nn.functional.softplus(params[1])
    left = -max_scale * scale
    right = max_scale * scale
    if blogistic.cdf(left, *params) > torch.min(ps):
        raise ValueError("The minimum value of ps is too small")
    if blogistic.cdf(right, *params) < torch.max(ps):
        raise ValueError("The maximum value of ps is too large")

    # implement vectorized binary search
    lefts = torch.ones_like(ps) * left
    rights = torch.ones_like(ps) * right

    for i in range(num_steps):
        mids = (lefts + rights) / 2
        cdfs = blogistic.cdf(mids, *params)
        lefts = torch.where(cdfs < ps, mids, lefts)
        rights = torch.where(cdfs > ps, mids, rights)

    return mids

def train_blogistic(xs, dof, lr, num_steps, allow_skew, device: torch.device = None):
    if allow_skew:
        degree = dof - 3
        blogistic = SkewedBLogistic(degree=degree, device=device)
        skew_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
        scale_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
        raw_coeffs = torch.nn.Parameter(torch.randn(degree + 1, device=device))
        params = [raw_coeffs, scale_param, skew_param]
    else:
        degree = dof - 2
        blogistic = BLogistic(degree=degree, device=device)
        scale_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
        raw_coeffs = torch.nn.Parameter(torch.randn(degree + 1, device=device))
        params = [raw_coeffs, scale_param]
    
    nll = lambda xs_batch: -torch.mean(blogistic.logpdf(xs_batch, *params))
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = nll(xs)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print(f"Step {step}, Final Loss: {loss.item():.4f}")
    
    return blogistic, params

if __name__ == "__main__":
    # example usage
    degree = 3
    coefs = torch.normal(0, 1, size=(degree + 1,))
    plot_xs = np.linspace(-10, 10, 10000)
    torch_plot_xs = torch.tensor(plot_xs, dtype=DT)

    blogistic = BLogistic(degree = degree)

    plt.plot(plot_xs, blogistic.pdf(torch_plot_xs, coefs).numpy())
    plt.show()

    print("ev", softmax(blogistic.logpdf(torch_plot_xs, coefs).numpy()) @ plot_xs)