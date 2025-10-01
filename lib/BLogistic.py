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

class BLogistic:
    """
    Bernstein Logistic distribution class that allows evaluation of logpdf_x for different degrees.
    """
    
    def __init__(self, degree: int, device: torch.device = None):
        """
        Initialize the BLogistic class with a specific degree.
        
        Args:
            degree: The degree of the Bernstein polynomial
            device: The device to place tensors on (defaults to CPU)
        """
        self.degree = degree
        self.device = device if device is not None else torch.device('cpu')
        self.mus, self.bernstein_to_standard_matrix_torch = self._initialize_consts(degree)
    
    def _initialize_consts(self, degree):
        """Initialize constants for the given degree."""
        mus = np.cumsum(1 / np.arange(1, degree+1)) / (np.arange(1, degree+1) + 1)
        mus = np.insert(mus, 0, 0)
        mus = torch.tensor(mus, dtype=DT, device=self.device)  # integral of lg(x/(1-x)) x ^ n

        return mus, get_bernstein_to_standard_matrix(degree)
    
    def logpdf(self, xs, coeffs):
        """
        Compute the log probability density function for given x values and coefficients.
        
        Args:
            xs: torch.Tensor, shape (batch,) or similar - input values
            coeffs: torch.Tensor, shape (degree+1,) - Bernstein polynomial coefficients
            
        Returns:
            torch.Tensor: log probability density values
        """
        # coeffs: torch.Tensor, shape (n+1,)
        # xs: torch.Tensor, shape (batch,) or similar
        normalized_coeffs = torch.softmax(coeffs, dim=0) * (self.degree + 1)
        standard_coeffs = self.bernstein_to_standard_matrix_torch @ normalized_coeffs
        mean = torch.dot(self.mus, standard_coeffs)
        shifted_xs = xs + mean

        log_fprime = -shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs)
        Fx = 1.0 / (1.0 + torch.exp(-shifted_xs))
        u_p = torch.pow(Fx.unsqueeze(-1), torch.arange(0, self.degree+1, dtype=DT, device=self.device))
        poly = torch.sum(u_p * standard_coeffs, dim=-1)
        return torch.log(poly) + log_fprime

    def pdf(self, xs, coeffs):
        """
        Compute the probability density function for given x values and coefficients.
        """
        return torch.exp(self.logpdf(xs, coeffs))
    
    def cdf(self, xs, coeffs):
        """
        Compute the cumulative distribution function for given x values and coefficients.
        """
        normalized_coeffs = torch.softmax(coeffs, dim=0) * (self.degree + 1)
        standard_coeffs = self.bernstein_to_standard_matrix_torch @ normalized_coeffs
        mean = torch.dot(self.mus, standard_coeffs)
        shifted_xs = xs + mean
        Fx = 1.0 / (1.0 + torch.exp(-shifted_xs))
        powers = torch.arange(0, self.degree + 1, dtype=DT, device=self.device) + 1
        cdf_terms = torch.pow(Fx.unsqueeze(-1), powers) / powers
        cdf_val = torch.sum(cdf_terms * standard_coeffs, dim=-1)
        return cdf_val

class SkewedBLogistic:
    """
    Modify BLogistic use Generalized logistic distribution type I to make it skewed.
    """

    def __init__(self, degree: int, device: torch.device = None):
        self.degree = degree
        self.device = device if device is not None else torch.device('cpu')
        self.bernstein_to_standard_matrix_torch = get_bernstein_to_standard_matrix(degree)
        self.euler_gamma = torch.tensor(EulerGamma, dtype=DT, device=self.device)
    
    def get_mus(self, skewness):
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        mus = (self.euler_gamma + torch.digamma(1 + (skewness * (powers + 1)))) / (powers + 1) - 1 / (skewness * (powers + 1) ** 2)
        return mus

    def _process_input(self, xs, coeffs, raw_scale, raw_scaleskewness):
        normalized_coeffs = torch.softmax(coeffs, dim=0) * (self.degree + 1)
        standard_coeffs = self.bernstein_to_standard_matrix_torch @ normalized_coeffs
        scale = torch.softplus(raw_scale)
        skewness = torch.softplus(raw_scaleskewness)

        mus = self.get_mus(skewness)
        mean = torch.dot(mus, standard_coeffs)

        shifted_xs = (xs + mean) / scale
        Fx = (1.0 + torch.exp(-shifted_xs)) ** -skewness
        return shifted_xs, standard_coeffs, Fx, scale

    def logpdf(self, xs, coeffs, raw_scale, raw_scaleskewness):
        shifted_xs, standard_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale, raw_scaleskewness)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)

        log_fprime = -shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        u_p = torch.pow(Fx.unsqueeze(-1), powers)
        poly = torch.sum(u_p * standard_coeffs, dim=-1)
        return torch.log(poly) + log_fprime
    
    def cdf(self, xs, coeffs, raw_scale, raw_scaleskewness):
        shifted_xs, standard_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale, raw_scaleskewness)

        powers = torch.arange(0, self.degree + 1, dtype=DT, device=self.device) + 1
        cdf_terms = torch.pow(Fx.unsqueeze(-1), powers) / powers
        cdf_val = torch.sum(cdf_terms * standard_coeffs, dim=-1)
        return cdf_val


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