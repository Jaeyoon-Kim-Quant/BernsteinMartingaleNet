import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic, get_ppf, train_blogistic
from lib.utils import load_data

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

degree = 3
scale = torch.tensor(np.log(np.exp(1) - 1), device=device) * 2
model = BLogistic(degree=degree, device=device)


test_xs = torch.linspace(-20, 20, 100000, device=device)
cdf = 1.0 / (1.0 + torch.exp(-test_xs))

eps = 1e-5
for i in range(degree + 1):
    params = np.ones(degree + 1) * -1e5
    params[i] = 1e5
    params = torch.tensor(params, device=device)
    pdf = model.pdf(test_xs, params, scale)
    mean = (test_xs @ pdf).item() * torch.diff(test_xs).mean().item()
    vectorized_pdf = torch.exp(model.logpdf(test_xs.reshape(-1, 1), params.reshape(1, -1), scale))
    mean_vectorized = (test_xs @ vectorized_pdf).item() * torch.diff(test_xs).mean().item()
    assert abs(mean - mean_vectorized) < eps

print("All tests passed")