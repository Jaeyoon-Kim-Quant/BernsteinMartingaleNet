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
from scipy.special import comb
from lib.DistHead import DistHead, StudentTHead, SkewedStudentTHead
from torch.distributions import StudentT
DT = torch.float64
torch.set_default_dtype(DT)

class BLogistic(DistHead):
    """
    Bernstein Logistic distribution class that uses the Bernstein polynomial to perterb the logistic distribution.
    """

    def __init__(self, degree: int, device: torch.device = None):
        self.degree = degree
        self.device = device if device is not None else torch.device('cpu')
        self.comb = torch.tensor([comb(self.degree, i) for i in range(self.degree + 1)], dtype=DT, device=self.device).reshape(1, -1)
        self.mus = torch.tensor(self._init_mus(), dtype=DT, device=self.device)
        self.to_variance = torch.tensor(self._init_variance(), dtype=DT, device=self.device)
    
    def num_params(self):
        return self.degree + 2
    
    def _get_harmonic_numbers(self, k):
        return np.concatenate([[0], np.cumsum(1 / np.arange(1, self.degree+1) ** k)])

    def _init_mus(self):
        harmonic_numbers = self._get_harmonic_numbers(1)
        mus = (harmonic_numbers - harmonic_numbers[::-1]) / (self.degree + 1)
        return mus
    
    def _init_variance(self):
        harmonic_numbers = self._get_harmonic_numbers(1)
        harmonic_numbers2 = self._get_harmonic_numbers(2)
        to_variance = (harmonic_numbers - harmonic_numbers[::-1]) ** 2 + np.pi ** 2 / 3 - (harmonic_numbers2 + harmonic_numbers2[::-1])
        to_variance /= (self.degree + 1)
        return to_variance

    def _process_input(self, xs, coeffs, raw_scale):
        normalized_coeffs = torch.softmax(coeffs, dim=-1) * (self.degree + 1)
        scale = torch.nn.functional.softplus(raw_scale).reshape(-1, 1)

        mean = normalized_coeffs @ self.mus

        shifted_xs = xs.reshape(-1, 1) / scale + mean.reshape(-1, 1)
        Fx = (1.0 + torch.exp(-shifted_xs)) ** -1
        return shifted_xs, normalized_coeffs, Fx, scale
    
    def _split_combined_coeffs(self, combined_coeffs):
        if combined_coeffs.ndim == 2:
            coeffs = combined_coeffs[:, :-1]
            raw_scale = combined_coeffs[:, -1]
        else:
            coeffs = combined_coeffs[:-1]
            raw_scale = combined_coeffs[-1]
        return coeffs, raw_scale

    def logpdf(self, xs, combined_coeffs):
        coeffs, raw_scale = self._split_combined_coeffs(combined_coeffs)

        shifted_xs, normalized_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        reversed_powers = torch.arange(self.degree, -1, -1, dtype=DT, device=self.device)
        log_fprime = - shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        log_u_p = -torch.nn.functional.softplus(-shifted_xs) * powers.reshape(1, -1)
        log_u_m = -torch.nn.functional.softplus(shifted_xs) * reversed_powers.reshape(1, -1)
        log_bernstein_poly = log_u_p + log_u_m + torch.log(self.comb)
        log_poly = torch.logsumexp(torch.log(normalized_coeffs) + log_bernstein_poly, dim=-1).reshape(-1, 1)
        log_poly = torch.logsumexp(coeffs + log_bernstein_poly, dim=-1).reshape(-1, 1) - torch.logsumexp(coeffs, dim=-1).reshape(-1, 1) + torch.log(torch.tensor(self.degree + 1, dtype=DT, device=self.device))
        return log_poly + log_fprime

    def naive_pdf(self, xs, combined_coeffs):
        coeffs, raw_scale = self._split_combined_coeffs(combined_coeffs)
        shifted_xs, normalized_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        reversed_powers = torch.arange(self.degree, -1, -1, dtype=DT, device=self.device)
        u_p = torch.pow(Fx, powers.reshape(1, -1))
        u_m = torch.pow(1 - Fx, reversed_powers.reshape(1, -1))
        bernstein_poly = u_p * u_m * self.comb
        poly = torch.sum(bernstein_poly * normalized_coeffs, dim=-1).reshape(-1, 1)
        return poly * torch.exp(-shifted_xs) / (1 + torch.exp(-shifted_xs)) ** 2 / scale

    def naive_logpdf(self, xs, combined_coeffs):
        coeffs, raw_scale = self._split_combined_coeffs(combined_coeffs)
        shifted_xs, normalized_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        reversed_powers = torch.arange(self.degree, -1, -1, dtype=DT, device=self.device)
        u_p = torch.pow(Fx, powers.reshape(1, -1))
        u_m = torch.pow(1 - Fx, reversed_powers.reshape(1, -1))
        bernstein_poly = u_p * u_m * self.comb
        poly = torch.sum(bernstein_poly * normalized_coeffs, dim=-1).reshape(-1, 1)
        log_fprime = - shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        return torch.log(poly) + log_fprime
    
    def pdf(self, xs, combined_coeffs):
        return torch.exp(self.logpdf(xs, combined_coeffs))

    def cdf(self, xs, combined_coeffs):
        raise NotImplementedError("CDF not implemented for BLogistic")

    def get_variance(self, combined_coeffs):
        coeffs, raw_scale = self._split_combined_coeffs(combined_coeffs)
        normalized_coeffs = torch.softmax(coeffs, dim=-1) * (self.degree + 1)
        scale = torch.nn.functional.softplus(raw_scale).reshape(-1, 1)
        return scale ** 2 * ((normalized_coeffs @ self.to_variance).reshape(-1, 1) - (normalized_coeffs @ self.mus).reshape(-1, 1) ** 2)

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

def train_blogistic(xs, dof, lr, num_steps, device: torch.device = None):
    degree = dof - 2
    blogistic = BLogistic(degree=degree, device=device)
    param = torch.randn(degree + 2, device=device)
    param[-1] = 0
    param = torch.nn.Parameter(param)
    
    nll = lambda xs_batch: -torch.mean(blogistic.logpdf(xs_batch, param))
    optimizer = torch.optim.Adam([param], lr=lr)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = nll(xs)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print(f"Step {step}, Final Loss: {loss.item():.4f}")
    
    return blogistic, param

class MixedBLogistic(DistHead):
    def __init__(self, dof, device: torch.device = None):
        self.dof = dof
        self.device = device if device is not None else torch.device('cpu')
        self.blogistic = BLogistic(dof - 6, device)
        self.skewed_studentt = SkewedStudentTHead(device)
    
    def num_params(self):
        return self.dof
    
    def logpdf(self, xs, params):
        blogistic_params = params[:, :self.blogistic.num_params()]
        blogistic_logpdf = self.blogistic.logpdf(xs, blogistic_params)
        #skewness_param = torch.nn.functional.softplus(params[:, -4]).reshape(-1, 1)
        #std = torch.nn.functional.softplus(params[:, -3]).reshape(-1, 1)
        #df = torch.nn.functional.softplus(params[:, -2]).reshape(-1, 1)
        mix_param = torch.nn.functional.sigmoid(params[:, -1]).reshape(-1, 1)
        studentt_logpdf = self.skewed_studentt.logpdf(xs, params[:, -4:-1])
        #return studentt_logpdf
        return torch.logsumexp(torch.stack([blogistic_logpdf + torch.log(mix_param), studentt_logpdf + torch.log(1 - mix_param)]), dim=0)

    def pdf(self, xs, params):
        return torch.exp(self.logpdf(xs, params))