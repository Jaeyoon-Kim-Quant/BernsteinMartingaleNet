import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic, SkewedBLogistic

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("../MarketData/spy_historical_data_20250929.csv")
df['date'] = pd.to_datetime(df['timestamp']).dt.date
df_close = [np.array(group["close"])[:-1] for _, group in df.groupby('date') if len(group) == 391]
#minutely_returns = np.concatenate([np.diff(arr) / arr[:-1] for arr in df_close])
minutely_returns = np.concatenate([np.diff(np.log(arr)) for arr in df_close])
std = np.sqrt(np.mean(minutely_returns**2))
xs = minutely_returns / std * (np.pi / np.sqrt(3))
torch_xs = torch.tensor(xs, device=device)

def train_blogistic(xs, dof, lr, num_steps, allow_skew):
    if allow_skew:
        degree = dof - 3
        blogistic = SkewedBLogistic(degree=degree, device=device)
        skew_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
        scale_param = torch.nn.Parameter(torch.tensor(0.0, device=device))
        raw_coeffs = torch.nn.Parameter(torch.zeros(degree + 1, device=device))
        params = [raw_coeffs, scale_param, skew_param]
    else:
        degree = dof - 1
        blogistic = BLogistic(degree=degree, device=device)
        raw_coeffs = torch.nn.Parameter(torch.zeros(degree + 1, device=device))
        params = [raw_coeffs]
    
    nll = lambda xs_batch: -torch.mean(blogistic.logpdf(xs_batch, *params))
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss = nll(torch_xs)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    print(f"Step {step}, Final Loss: {loss.item():.4f}")
    
    return blogistic, params

dof = 32
lr = 0.01
num_steps = 3000
allow_skew = True
# fit a BLogistic distribution to the data
blogistic, params = train_blogistic(xs, dof, lr, num_steps, allow_skew=allow_skew)

plot_xs = torch.linspace(-12, 12, 10000)
fig, ax = plt.subplots(1, 2)
ax[0].plot(plot_xs.numpy(), blogistic.pdf(plot_xs.to(device), *params).cpu().detach().numpy(), label="BLogistic")
ax[0].hist(xs, bins=512, density=True, label="Data")
ax[0].set_title("PDF")
ax[0].legend()
def logistic_cdf(xs):
    return 1.0 / (1.0 + np.exp(-xs))
ax[1].plot(logistic_cdf(plot_xs.numpy()), blogistic.cdf(plot_xs.to(device), *params).cpu().detach().numpy(), label="BLogistic")
ax[1].plot(logistic_cdf(np.sort(xs)), np.linspace(0, 1, len(xs)), label="Data")
ax[1].legend()
ax[1].set_title("CDF")
plt.show()