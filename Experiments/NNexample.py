import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from zoneinfo import ZoneInfo
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils import get_sequence_data
from lib.BLogistic import SkewedBLogistic

folder_path = r"../MarketData/historical_data"
context_window = 10
X, Y = get_sequence_data(folder_path, context_window, force_recompute=False)
print("data shape", X.shape, Y.shape)

dof = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(context_window, context_window)
        self.fc2 = nn.Linear(context_window, dof)
        self.blogistic = SkewedBLogistic(dof - 3, device=device)

    def forward(self, x, y):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return -self.blogistic.logpdf_vectorized(y, params[:, :-2], params[:, -2], params[:, -1]).mean()
    
def train_simple_nn(X, Y, lr, num_steps, device: torch.device = None):
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = model(X, Y)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    return model, model.parameters()


simple_X = torch.tensor(X[:, :, 0], device=device)
Y = torch.tensor(Y, device=device).reshape(-1, 1)

lr = 0.001
num_steps = 1000
model, params = train_simple_nn(simple_X, Y, lr, num_steps, device)

#test_sample_size = 5
#test_X = torch.randn(test_sample_size, 1, device=device)
#test_Y = torch.randn(test_sample_size, 1, device=device)
#test_param = torch.randn(test_sample_size, dof, device=device)
#
#model = SkewedBLogistic(dof - 3, device=device)
#print(model.logpdf_vectorized(test_Y, test_param[:, :-2], test_param[:, -2], test_param[:, -1]))
#
#
#parallel_y = torch.randn(100, device=device)
#single_param = torch.randn(dof, device=device)
#print(model.logpdf(parallel_y, single_param[:-2], single_param[-2], single_param[-1]))
#