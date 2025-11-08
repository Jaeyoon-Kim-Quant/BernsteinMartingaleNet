import sys
import os
import numpy as np
import torch
import pytest

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic, get_ppf, train_blogistic
from lib.utils import load_data


@pytest.fixture(scope="module")
def device():
    """Fixture to provide the device (GPU if available, otherwise CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="module")
def model(device):
    """Fixture to provide a BLogistic model instance."""
    degree = 3
    return BLogistic(degree=degree, device=device)


@pytest.fixture(scope="module")
def test_xs(device):
    """Fixture to provide test x values."""
    return torch.linspace(-20, 20, 100000, device=device)


@pytest.fixture(scope="module")
def scale(device):
    """Fixture to provide the scale parameter."""
    return torch.tensor(np.log(np.exp(1) - 1), device=device) * 2


@pytest.fixture(scope="module")
def degree():
    """Fixture to provide the degree parameter."""
    return 3


@pytest.fixture(scope="module")
def eps():
    """Fixture to provide the epsilon tolerance for assertions."""
    return 1e-5


class TestBLogisticMean:
    """Test class for BLogistic mean calculations and PDF properties."""
    
    @pytest.mark.parametrize("param_index", range(4))  # degree + 1 = 4
    def test_mean_consistency(self, model, test_xs, scale, param_index, device, eps):
        """Test that mean calculation is consistent between pdf and vectorized logpdf."""
        degree = 3
        params = np.ones(degree + 2) * -1e5
        params[param_index] = 1e5
        params = torch.tensor(params, device=device)
        params[-1] = scale
        
        pdf = model.pdf(test_xs, params)
        mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
        
        vectorized_pdf = torch.exp(model.logpdf(test_xs.reshape(-1, 1), params.reshape(1, -1)))
        mean_vectorized = (test_xs @ vectorized_pdf).item() * torch.diff(test_xs).mean().item()
        
        assert abs(mean - mean_vectorized) < eps, \
            f"Mean mismatch for param_index={param_index}: {mean} vs {mean_vectorized}"
    
    @pytest.mark.parametrize("param_index", range(4))  # degree + 1 = 4
    def test_pdf_integral(self, model, test_xs, scale, param_index, device, eps):
        """Test that PDF integrates to 1."""
        degree = 3
        params = np.ones(degree + 2) * -1e5
        params[param_index] = 1e5
        params = torch.tensor(params, device=device)
        params[-1] = scale
        
        pdf = model.pdf(test_xs, params)
        integral = pdf.sum().item() * torch.diff(test_xs).mean().item()
        
        assert abs(integral - 1.0) < eps, \
            f"PDF does not integrate to 1 for param_index={param_index}: {integral}"
    
    @pytest.mark.parametrize("param_index", range(4))  # degree + 1 = 4
    def test_naive_pdf_consistency(self, model, test_xs, scale, param_index, device, eps):
        """Test that naive_pdf matches pdf."""
        degree = 3
        params = np.ones(degree + 2) * -1e5
        params[param_index] = 1e5
        params = torch.tensor(params, device=device)
        params[-1] = scale
        
        pdf = model.pdf(test_xs, params)
        naive_pdf = model.naive_pdf(test_xs, params)
        
        max_diff = abs(naive_pdf - pdf).max().item()
        assert max_diff < eps, \
            f"naive_pdf does not match pdf for param_index={param_index}: max_diff={max_diff}"