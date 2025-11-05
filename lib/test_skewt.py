import numpy as np
import matplotlib.pyplot as plt
import torch
from DistHead import SkewedStudentTHead

df = 10
mean = 0
std = 1
skewness = 2
dist = SkewedStudentTHead(device = "cpu")
params = torch.tensor([mean, std, df, skewness], dtype = torch.float64, device = "cpu")
x = torch.linspace(-100, 100, 100000, device = "cpu")
pdf = dist.pdf(x, params)

integral = (pdf.sum() * torch.diff(x).mean()).item()
assert abs(integral - 1.0) < 1e-5
