import sys
import os
import numpy as np
import torch
import pytest

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic, get_ppf, train_blogistic, MixedBLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
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


class TestMixedBLogisticMean:
    """Test class for MixedBLogistic mean calculations and PDF properties."""
    
    @pytest.mark.parametrize("param_index", range(13))  # blogistic.num_params() = 13 for dof=16
    def test_mean_calculation(self, mixed_model, test_xs, scale, param_index, device, eps):
        """Test that mean calculation produces finite and reasonable values."""
        dof = 16
        params = np.ones(dof) * -1e5
        params[param_index] = 1e5
        params = torch.tensor(params, device=device)
        # Set scale parameter (last param of blogistic, which is at index 12)
        params[12] = scale
        # Set reasonable values for StudentT params
        params[-3] = scale  # std (softplus(0) = log(2))
        params[-2] = 0.0  # df (softplus(0) = log(2))
        params[-1] = 0.0  # mix_param (sigmoid(0) = 0.5)
        
        # Reshape params to (1, dof) for logpdf
        params_reshaped = params.reshape(1, -1)
        
        pdf = torch.exp(mixed_model.logpdf(test_xs.reshape(-1, 1), params_reshaped))
        mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
        
        # Check that mean is finite and reasonable (within test_xs range)
        assert np.isfinite(mean), \
            f"Mean is not finite for param_index={param_index}: {mean}"
        assert -25 < mean < 25, \
            f"Mean is out of reasonable range for param_index={param_index}: {mean}"
    
    @pytest.mark.parametrize("param_index", range(13))  # blogistic.num_params() = 13 for dof=16
    def test_pdf_integral(self, mixed_model, test_xs, scale, param_index, device, eps):
        """Test that PDF integrates to 1."""
        dof = 16
        params = np.ones(dof) * -1e5
        params[param_index] = 1e5
        params = torch.tensor(params, device=device)
        # Set scale parameter (last param of blogistic, which is at index 12)
        params[12] = scale
        # Set reasonable values for StudentT params
        params[-3] = 0.0  # std (softplus(0) = log(2))
        params[-2] = 10.0  # df (softplus(0) = log(2))
        params[-1] = 0.0  # mix_param (sigmoid(0) = 0.5)
        
        # Reshape params to (1, dof) for logpdf
        params_reshaped = params.reshape(1, -1)
        
        pdf = torch.exp(mixed_model.logpdf(test_xs.reshape(-1, 1), params_reshaped))
        integral = pdf.sum().item() * torch.diff(test_xs).mean().item()
        
        assert abs(integral - 1.0) < eps, \
            f"PDF does not integrate to 1 for param_index={param_index}: {integral}"


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