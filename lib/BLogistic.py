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

DT = torch.float64
torch.set_default_dtype(DT)

class BLogistic:
    """
    Bernstein Logistic distribution class that allows evaluation of logpdf_x for different degrees.
    """
    
    def __init__(self, degree: int):
        """
        Initialize the BLogistic class with a specific degree.
        
        Args:
            degree: The degree of the Bernstein polynomial
        """
        self.degree = degree
        self.mus, self.bernstein_to_standard_matrix_torch = self._initialize_consts(degree)
    
    def _initialize_consts(self, degree):
        """Initialize constants for the given degree."""
        mus = np.cumsum(1 / np.arange(1, degree+1)) / (np.arange(1, degree+1) + 1)
        mus = np.insert(mus, 0, 0)
        mus = torch.tensor(mus, dtype=torch.float64)  # integral of lg(x/(1-x)) x ^ n

        bernstein_to_standard_matrix = np.zeros((degree+1, degree+1))
        for v in range(degree+1):
            for l in range(v, degree+1):
                bernstein_to_standard_matrix[l, v] = comb(degree, l) * comb(l, v) * (-1)**(l-v)
        bernstein_to_standard_matrix_torch = torch.tensor(bernstein_to_standard_matrix, dtype=DT)

        return mus, bernstein_to_standard_matrix_torch
    
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
        u_p = torch.pow(Fx.unsqueeze(-1), torch.arange(0, self.degree+1, dtype=DT, device=Fx.device))
        poly = torch.sum(u_p * standard_coeffs, dim=-1)
        return torch.log(poly) + log_fprime

    def pdf(self, xs, coeffs):
        """
        Compute the probability density function for given x values and coefficients.
        """
        return torch.exp(self.logpdf(xs, coeffs))

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