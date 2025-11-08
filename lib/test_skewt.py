import numpy as np
import torch
import pytest
from lib.DistHead import SkewedStudentTHead


@pytest.fixture(scope="module")
def device():
    """Fixture to provide the device (CPU for this test)."""
    return torch.device("cpu")


@pytest.fixture(scope="module")
def dist(device):
    """Fixture to provide a SkewedStudentTHead instance."""
    return SkewedStudentTHead(device=device)


@pytest.fixture(scope="module")
def test_params(device):
    """Fixture to provide test parameters for SkewedStudentTHead."""
    df = 10
    mean = 0
    std = 1
    skewness = 2
    return torch.tensor([mean, std, df, skewness], dtype=torch.float64, device=device)


@pytest.fixture(scope="module")
def test_x(device):
    """Fixture to provide test x values."""
    return torch.linspace(-100, 100, 100000, device=device)


@pytest.fixture(scope="module")
def eps():
    """Fixture to provide the epsilon tolerance for assertions."""
    return 1e-5


class TestSkewedStudentTHead:
    """Test class for SkewedStudentTHead PDF properties."""
    
    def test_pdf_integral(self, dist, test_x, test_params, eps):
        """Test that PDF integrates to 1."""
        pdf = dist.pdf(test_x, test_params)
        integral = (pdf.sum() * torch.diff(test_x).mean()).item()
        
        assert abs(integral - 1.0) < eps, \
            f"PDF does not integrate to 1: {integral}"
