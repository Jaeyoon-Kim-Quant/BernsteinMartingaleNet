import sys
import os
import numpy as np
import torch
import pytest
from scipy.integrate import quad

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic, get_ppf, train_blogistic, MixedBLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
from lib.utils import load_data


@pytest.fixture(scope="module")
def device():
    """Fixture to provide the device (GPU if available, otherwise CPU)."""
    return torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="module")
def model(device):
    """Fixture to provide a BLogistic model instance."""
    degree = 3
    return BLogistic(degree=degree, device=device)

@pytest.fixture(scope="module")
def mixed_model(device):
    """Fixture to provide a MixedBLogistic model instance."""
    dof = 16
    return MixedBLogistic(dof=dof, device=device)

@pytest.fixture(scope="module")
def normal_head(device):
    """Fixture to provide a NormalHead instance."""
    return NormalHead(device=device)

@pytest.fixture(scope="module")
def studentt_head(device):
    """Fixture to provide a StudentTHead instance."""
    return StudentTHead(device=device)

@pytest.fixture(scope="module")
def skewed_studentt_head(device):
    """Fixture to provide a SkewedStudentTHead instance."""
    return SkewedStudentTHead(device=device)


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
    
    @pytest.mark.parametrize("param_index", range(4))  # degree + 1 = 4
    def test_variance_calculation(self, model, test_xs, scale, param_index, device, eps):
        """Test that get_variance matches numerical variance calculation."""
        degree = 3
        params = np.ones(degree + 2) * -1e5
        params[param_index] = 1e5
        params = torch.tensor(params, device=device)
        params[-1] = scale
        
        numerical_variance, precision = quad(lambda x: x**2 * model.pdf(torch.tensor(x), params).item(), -np.inf, np.inf)
        
        # Analytical variance from get_variance
        analytical_variance = model.get_variance(params).item()
        
        assert abs(numerical_variance - analytical_variance) < eps, \
            f"Variance mismatch for BLogistic: numerical={numerical_variance}, analytical={analytical_variance}"

class TestNormalHead:
    """Test class for NormalHead PDF properties."""
    
    def test_pdf_integral(self, normal_head, test_xs, scale, device, eps):
        """Test that PDF integrates to 1."""
        # NormalHead has 1 param: std
        params = torch.tensor([[scale]], device=device)  # shape: (1, 1)
        
        pdf = normal_head.pdf(test_xs.reshape(-1, 1), params)
        integral = pdf.sum().item() * torch.diff(test_xs).mean().item()
        
        assert abs(integral - 1.0) < eps, \
            f"PDF does not integrate to 1: {integral}"
    
    def test_mean_calculation(self, normal_head, test_xs, scale, device, eps):
        """Test that mean calculation produces expected value (should be 0 for NormalHead)."""
        params = torch.tensor([[scale]], device=device)  # shape: (1, 1)
        
        pdf = normal_head.pdf(test_xs.reshape(-1, 1), params)
        mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
        
        # NormalHead has mean 0
        assert abs(mean) < eps, \
            f"Mean should be 0 for NormalHead, got: {mean}"
    
    def test_variance_calculation(self, normal_head, test_xs, scale, device, eps):
        """Test that get_variance matches numerical variance calculation."""
        params = torch.tensor([[scale]], device=device)  # shape: (1, 1)
        
        #pdf = normal_head.pdf(test_xs.reshape(-1, 1), params)
        #dx = torch.diff(test_xs).mean().item()
        
        ## Numerical variance: Var = E[X^2] - E[X]^2
        #mean = (test_xs @ pdf).item() * dx
        #mean_squared = (test_xs ** 2 @ pdf).item() * dx
        #numerical_variance = mean_squared - mean ** 2
        numerical_variance, _ = quad(lambda x: x**2 * normal_head.pdf(torch.tensor(x), params).item(), -np.inf, np.inf)
        
        # Analytical variance from get_variance
        analytical_variance = normal_head.get_variance(params).item()
        
        assert abs(numerical_variance - analytical_variance) < eps, \
            f"Variance mismatch for NormalHead: numerical={numerical_variance}, analytical={analytical_variance}"


class TestStudentTHead:
    """Test class for StudentTHead PDF properties."""
    
    def test_pdf_integral(self, studentt_head, test_xs, scale, device, eps):
        """Test that PDF integrates to 1."""
        # StudentTHead has 2 params: std, df
        params = torch.tensor([[scale, 10.0]], device=device)  # shape: (1, 2)
        
        pdf = studentt_head.pdf(test_xs.reshape(-1, 1), params)
        integral = pdf.sum().item() * torch.diff(test_xs).mean().item()
        
        assert abs(integral - 1.0) < eps, \
            f"PDF does not integrate to 1: {integral}"
    
    def test_mean_calculation(self, studentt_head, test_xs, scale, device, eps):
        """Test that mean calculation produces expected value (should be 0 for StudentTHead)."""
        params = torch.tensor([[scale, 10.0]], device=device)  # shape: (1, 2)
        
        pdf = studentt_head.pdf(test_xs.reshape(-1, 1), params)
        mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
        
        # StudentTHead has mean 0
        assert abs(mean) < eps, \
            f"Mean should be 0 for StudentTHead, got: {mean}"
    
    def test_variance_calculation(self, studentt_head, test_xs, scale, device, eps):
        """Test that get_variance matches numerical variance calculation."""
        params = torch.tensor([[scale, 10.0]], device=device)  # shape: (1, 2)
        numerical_variance, _ = quad(lambda x: x**2 * studentt_head.pdf(torch.tensor(x), params).item(), -np.inf, np.inf)
        
        # Analytical variance from get_variance
        analytical_variance = studentt_head.get_variance(params).item()
        
        assert abs(numerical_variance - analytical_variance) < eps, \
            f"Variance mismatch for StudentTHead: numerical={numerical_variance}, analytical={analytical_variance}"


class TestSkewedStudentTHead:
    """Test class for SkewedStudentTHead PDF properties."""
    
    def test_pdf_integral(self, skewed_studentt_head, test_xs, scale, device, eps):
        """Test that PDF integrates to 1."""
        # SkewedStudentTHead has 4 params: std, df, skewness, and one more
        # Based on _get_dist, it uses params[:, 0], params[:, 1], params[:, 2]
        # To get skewness=1.0 (no skew), we need softplus(x) = 1.0, so x = log(exp(1) - 1) ≈ 0.541
        # We'll set the first 3 params and leave the 4th as 0
        #skewness_param = np.log(np.exp(1) - 1)  # gives skewness ≈ 1.0
        skewness_param = 0.0  # gives skewness ≈ 1.0
        params = torch.tensor([[scale, 10.0, skewness_param, 0.0]], device=device)  # shape: (1, 4)
        
        pdf = skewed_studentt_head.pdf(test_xs.reshape(-1, 1), params)
        mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
        integral = pdf.sum().item() * torch.diff(test_xs).mean().item()
        
        assert abs(integral - 1.0) < eps, \
            f"PDF does not integrate to 1: {integral}"
    
    def test_mean_calculation(self, skewed_studentt_head, test_xs, scale, device, eps):
        """Test that mean calculation produces finite and reasonable values."""
        # SkewedStudentTHead has mean correction, so mean may not be exactly 0
        # To get skewness=1.0 (no skew), we need softplus(x) = 1.0, so x = log(exp(1) - 1) ≈ 0.541
        skewness_param = np.log(np.exp(1) - 1)  # gives skewness ≈ 1.0
        params = torch.tensor([[scale, 10.0, skewness_param, 0.0]], device=device)  # shape: (1, 4)
        
        pdf = skewed_studentt_head.pdf(test_xs.reshape(-1, 1), params)
        mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
        print("mean", mean)
        
        # Check that mean is finite and reasonable
        assert np.isfinite(mean), \
            f"Mean is not finite: {mean}"
        assert -25 < mean < 25, \
            f"Mean is out of reasonable range: {mean}"
    
    def test_variance_calculation(self, skewed_studentt_head, test_xs, scale, device, eps):
        """Test that get_variance matches numerical variance calculation."""
        # Test with different skewness values
        skewness_params = [0.0, np.log(np.exp(1) - 1), 1.0, 2.0]
        
        for skewness_param in skewness_params:
            params = torch.tensor([[scale, 10.0, skewness_param]], device=device)  # shape: (1, 4)
            
            # Numerical variance: Var = E[X^2]
            numerical_variance, _ = quad(lambda x: x**2 * skewed_studentt_head.pdf(x, params).item(), -np.inf, np.inf)
            
            # Analytical variance from get_variance
            analytical_variance = skewed_studentt_head.get_variance(params).item()
            
            assert abs(numerical_variance - analytical_variance) < eps, \
                f"Variance mismatch for SkewedStudentTHead (skewness_param={skewness_param}): " \
                f"numerical={numerical_variance}, analytical={analytical_variance}"