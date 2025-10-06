import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.BLogistic import get_ppf, train_blogistic
from lib.utils import load_data

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xs = load_data("../MarketData/spy_historical_data_20250929.csv")
torch_xs = torch.tensor(xs, device=device)

dof = 16
lr = 0.05
num_steps = 2000
# fit a BLogistic distribution to the data
ps = torch.linspace(0, 1, 1000, device=device)[1:-1]
skewed_ppfs = get_ppf(*train_blogistic(torch_xs, dof, lr, num_steps, allow_skew=True, device=device), ps)
symmetric_ppfs = get_ppf(*train_blogistic(torch_xs, dof, lr, num_steps, allow_skew=False, device=device), ps)

data_ppfs = np.percentile(xs, 100 * ps.cpu().numpy())
plt.plot(data_ppfs, data_ppfs, label="data")
plt.plot(skewed_ppfs.cpu().detach().numpy(), data_ppfs, label="Skewed")
plt.plot(symmetric_ppfs.cpu().detach().numpy(), data_ppfs, label="Symmetric")
plt.xlabel("Fitted PPF")
plt.ylabel("Data PPF")
plt.title("Q-Q Plot")
plt.legend()

plt.show()