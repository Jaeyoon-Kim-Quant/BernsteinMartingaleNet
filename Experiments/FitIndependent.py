import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic

df = pd.read_csv("../MarketData/spy_historical_data_20250929.csv")
df['date'] = pd.to_datetime(df['timestamp']).dt.date
df_close = [np.array(group["close"])[:-1] for _, group in df.groupby('date') if len(group) == 391]
minutely_returns = np.concatenate([np.diff(arr) / arr[:-1] for arr in df_close])
std = np.sqrt(np.mean(minutely_returns**2))
xs = minutely_returns / std
print(len(xs))


degree = 15
lr = 0.01
num_steps = 1000
batch_size = 4096
# fit a BLogistic distribution to the data
blogistic = BLogistic(degree=degree)

raw_coeffs = torch.nn.Parameter(torch.normal(0, 1, size=(degree + 1,)))

def nll(xs_batch):
    """Negative log-likelihood for a batch."""
    return -torch.mean(blogistic.logpdf(xs_batch, raw_coeffs))

optimizer = torch.optim.Adam([raw_coeffs], lr=lr)

# Create dataset with batching
dataset = TensorDataset(torch.tensor(xs))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for step in range(1, num_steps + 1):
    total_loss = 0.0
    for batch_idx, (batch_xs,) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = nll(batch_xs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}")

print(f"Step {step}, Final Loss: {total_loss:.4f}")