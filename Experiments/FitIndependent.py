import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import BLogistic from lib/BLogistic.py
from lib.BLogistic import BLogistic, SkewedBLogistic, get_ppf, train_blogistic
from lib.utils import load_data

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xs = load_data("../MarketData/spy_historical_data_20250929.csv")
torch_xs = torch.tensor(xs, device=device)

dof = 16
lr = 0.05
num_steps = 2000
blogistic, params = train_blogistic(torch_xs, dof, lr, num_steps, allow_skew=True, device=device)

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

    


