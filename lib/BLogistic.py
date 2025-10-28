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
DT = torch.float64
torch.set_default_dtype(DT)

class BLogistic:
    """
    Bernstein Logistic distribution class that uses the Bernstein polynomial to perterb the logistic distribution.
    """

    def __init__(self, degree: int, device: torch.device = None):
        self.degree = degree
        self.device = device if device is not None else torch.device('cpu')
        self.comb = torch.tensor([comb(self.degree, i) for i in range(self.degree + 1)], dtype=DT, device=self.device).reshape(1, -1)
        self.mus = torch.tensor(self.get_mus(), dtype=DT, device=self.device)
    
    def get_mus(self):
        harmonic_numbers = np.cumsum(1 / np.arange(1, self.degree+1))
        harmonic_numbers = np.concatenate([[0], harmonic_numbers])
        mus = np.zeros(self.degree+1)
        for i in range(self.degree+1):
            mus[i] = (harmonic_numbers[i] - harmonic_numbers[self.degree-i]) / (self.degree + 1)
        return mus

    def _process_input(self, xs, coeffs, raw_scale):
        normalized_coeffs = torch.softmax(coeffs, dim=-1) * (self.degree + 1)
        scale = torch.nn.functional.softplus(raw_scale).reshape(-1, 1)

        mean = normalized_coeffs @ self.mus

        shifted_xs = xs.reshape(-1, 1) / scale + mean.reshape(-1, 1)
        Fx = (1.0 + torch.exp(-shifted_xs)) ** -1
        return shifted_xs, normalized_coeffs, Fx, scale

    def logpdf(self, xs, coeffs, raw_scale):
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

    def naive_pdf(self, xs, coeffs, raw_scale):
        shifted_xs, normalized_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        reversed_powers = torch.arange(self.degree, -1, -1, dtype=DT, device=self.device)
        u_p = torch.pow(Fx, powers.reshape(1, -1))
        u_m = torch.pow(1 - Fx, reversed_powers.reshape(1, -1))
        bernstein_poly = u_p * u_m * self.comb
        poly = torch.sum(bernstein_poly * normalized_coeffs, dim=-1).reshape(-1, 1)
        return poly * torch.exp(-shifted_xs) / (1 + torch.exp(-shifted_xs)) ** 2 / scale

    def naive_logpdf(self, xs, coeffs, raw_scale):
        shifted_xs, normalized_coeffs, Fx, scale = self._process_input(xs, coeffs, raw_scale)
        powers = torch.arange(0, self.degree+1, dtype=DT, device=self.device)
        reversed_powers = torch.arange(self.degree, -1, -1, dtype=DT, device=self.device)
        u_p = torch.pow(Fx, powers.reshape(1, -1))
        u_m = torch.pow(1 - Fx, reversed_powers.reshape(1, -1))
        bernstein_poly = u_p * u_m * self.comb
        poly = torch.sum(bernstein_poly * normalized_coeffs, dim=-1).reshape(-1, 1)
        log_fprime = - shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        return torch.log(poly) + log_fprime
    
    def pdf(self, xs, coeffs, raw_scale):
        return torch.exp(self.logpdf(xs, coeffs, raw_scale))

    def cdf(self, xs, coeffs, raw_scale):
        raise NotImplementedError("CDF not implemented for BLogistic")

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

def open_uniform_knots(nbasis: int, degree: int = 3, *, device=None, dtype=None):
    """
    Open-uniform knot vector with multiplicity degree+1 at the ends over [0,1].
    For degree=3 (cubic), need nbasis >= 4.
    """
    assert nbasis >= degree + 1, "nbasis must be >= degree+1"
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    n_interior = nbasis - degree - 1  # number of interior knots

    # interior knots uniformly in (0,1), empty if n_interior == 0
    if n_interior > 0:
        interior = torch.linspace(0, 1, n_interior + 2, device=device, dtype=dtype)[1:-1]
    else:
        interior = torch.empty(0, device=device, dtype=dtype)

    t0 = torch.zeros(degree + 1, device=device, dtype=dtype)
    t1 = torch.ones (degree + 1, device=device, dtype=dtype)
    return torch.cat([t0, interior, t1])  # shape: (nbasis + degree + 1,)

class SplineLogistic:
    """
    Spline Logistic distribution class that delegates to BLogistic.
    """

    def __init__(self, dof: int, device: torch.device = None):
        self.dof = dof
        self.nbasis = dof - 1
        self.device = device if device is not None else torch.device('cpu')
        self.spline_degree = 3
        self.knots = open_uniform_knots(self.nbasis, self.spline_degree, device=self.device)
        self.normalization, self.mus = self._calc_mus()
        print("finished initializing spline logistic model")
    
    def _bspline_basis(self, x: torch.Tensor):
        """
        Evaluate B-spline basis of given degree at points x in [0,1] (broadcasts over x's shape).
        Returns B with shape (x.numel(), nbasis). Differentiable in x.
        """
        # Flatten x for computation; remember shape to reshape back if needed
        x_flat = x.reshape(-1, 1)  # (N, 1)
        N = x_flat.shape[0]

        # Handle x==1.0 corner so the last basis gets weight 1 there
        # (Cox–de Boor uses half-open intervals [t_i, t_{i+1}))
        eps_back = torch.nextafter(torch.tensor(1.0, device=x.device, dtype=x.dtype),
                                torch.tensor(0.0, device=x.device, dtype=x.dtype)) - 1.0
        x_work = torch.minimum(x_flat, (1.0 + eps_back).to(x.device).to(x.dtype))

        # Zeroth-degree indicator functions: N_i^0(x)
        # For each i, active on [t_i, t_{i+1})
        ti   = self.knots[:-1]        # (nbasis+degree,)
        tip1 = self.knots[1:]         # (nbasis+degree,)
        left  = (x_work >= ti)   # broadcast (N, nbasis+degree)
        right = (x_work <  tip1) # broadcast
        N0 = (left & right).to(x.dtype)

        # Keep only the first nbasis columns (there are len(knots)-1 intervals; #basis = len(knots)-degree-1)
        # Build recursively up to degree p
        Ni = N0[:, :self.nbasis]  # shape (N, nbasis) at p=0

        for p in range(1, self.spline_degree + 1):
            # Denominators (with safe zeros)
            denom1 = (self.knots[p: p + self.nbasis] - self.knots[:self.nbasis])            # t_{i+p} - t_i, shape (nbasis,)
            denom2 = (self.knots[p + 1: p + 1 + self.nbasis] - self.knots[1: self.nbasis+1])# t_{i+p+1} - t_{i+1}

            # Shifted basis from previous degree
            Ni_left  = Ni                                             # N_i^{p-1}
            Ni_right = torch.zeros_like(Ni)
            Ni_right[:, :-1] = Ni[:, 1:]                              # N_{i+1}^{p-1}; last column stays zero

            # Terms with safe division (define 0/0 as 0)
            with torch.no_grad():
                safe1 = denom1 != 0
                safe2 = denom2 != 0
            term1 = torch.zeros_like(Ni)
            term2 = torch.zeros_like(Ni)

            if safe1.any():
                w1 = torch.zeros_like(denom1)
                w1[safe1] = 1.0 / denom1[safe1]
                term1 = (x_work - self.knots[:self.nbasis]) * w1  # broadcasts (N, nbasis)
                term1 = term1 * Ni_left

            if safe2.any():
                w2 = torch.zeros_like(denom2)
                w2[safe2] = 1.0 / denom2[safe2]
                term2 = (self.knots[p + 1: p + 1 + self.nbasis] - x_work) * w2
                term2 = term2 * Ni_right

            Ni = term1 + term2  # N_i^p

        # Ni sums to 1 across basis for x in [0,1]; nonnegative.
        return Ni  # shape (N, nbasis)
    
    def _calc_mus(self):
        integral_xs = torch.linspace(0, 1, 2 ** 15, device=self.device)[1:-1]
        integral_bspline_basis = self._bspline_basis(integral_xs)
        Finv = torch.log(integral_xs / (1 - integral_xs))
        normalization = 1 / torch.trapz(integral_bspline_basis, integral_xs, dim=0)
        mus = normalization * torch.trapz(Finv.reshape(-1, 1) * integral_bspline_basis, integral_xs, dim=0)
        return normalization, mus
        
    def _process_input(self, xs, coeffs, raw_scale):
        normalized_coeffs = torch.softmax(coeffs, dim=0)
        scale = torch.nn.functional.softplus(raw_scale)

        mean = self.mus @ normalized_coeffs.reshape(-1, 1)
        shifted_xs = xs.reshape(-1, 1) / scale + mean.reshape(-1, 1)
        Fx = 1.0 / (1.0 + torch.exp(-shifted_xs))  # logistic CDF
        spline_basis = self._bspline_basis(Fx) * self.normalization
        spline_combined = (spline_basis @ normalized_coeffs).reshape(-1, 1)
        return shifted_xs, spline_combined, Fx, scale
    
    def logpdf(self, xs, coeffs, raw_scale):
        shifted_xs, spline_combined, Fx, scale = self._process_input(xs, coeffs, raw_scale)
        log_fprime = - shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        return torch.log(spline_combined) + log_fprime

    def _process_input_vectorized(self, xs, coeffs, raw_scale):
        normalized_coeffs = torch.softmax(coeffs, dim=1)
        scale = torch.nn.functional.softplus(raw_scale).reshape(-1, 1)

        mean = normalized_coeffs @ self.mus
        mean = mean.reshape(-1, 1)

        shifted_xs = (xs + mean) / scale
        Fx = (1.0 + torch.exp(-shifted_xs)) ** -1
        spline_basis = self._bspline_basis(Fx) * self.normalization
        spline_combined = (spline_basis * normalized_coeffs).sum(dim=1).reshape(-1, 1)
        return shifted_xs, spline_combined, Fx, scale

    def logpdf_vectorized(self, xs, coeffs, raw_scale):
        shifted_xs, spline_combined, Fx, scale = self._process_input_vectorized(xs, coeffs, raw_scale)
        log_fprime = - shifted_xs - 2 * torch.nn.functional.softplus(-shifted_xs) - torch.log(scale)
        return torch.log(spline_combined) + log_fprime
    
    def pdf(self, xs, coeffs, raw_scale):
        return torch.exp(self.logpdf(xs, coeffs, raw_scale))

    def cdf(self, xs, coeffs):
        shifted_xs, normalized_coeffs, Fx = self._process_input(xs, coeffs)
        spline_basis = (self._bspline_basis(Fx) @ normalized_coeffs).reshape(-1, 1)
        return spline_basis

def train_spline_logistic(xs, dof, lr, num_steps, device: torch.device = None):
    model = SplineLogistic(dof=dof, device=device)
    scale_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
    raw_coeffs = torch.nn.Parameter(torch.randn(model.nbasis, device=device))
    params = [raw_coeffs, scale_param]
    
    nll = lambda xs_batch: -torch.mean(model.logpdf(xs_batch, *params))
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = nll(xs)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print(f"Step {step}, Final Loss: {loss.item():.4f}")
    
    return model, params

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