import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

cwd = os.getcwd()
root = cwd.split("BernsteinMartingaleNet")[0] + "BernsteinMartingaleNet"
if root not in sys.path:
    sys.path.append(root)

from lib.utils import train_model, get_normalized_data, train_dist
from lib.BLogistic import BLogistic, MixedBLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
from lib.Michenkow import Michenkow, AdjustedMichenkow
device = "cpu"

folder_path = root + "/MarketData/historical_data"
train_xs, train_ys, train_rv, dev_xs, dev_ys, dev_rv, test_xs, test_ys, test_rv = get_normalized_data(folder_path, 1, device, 0.2, 0.2)
xs = torch.concat([train_xs, dev_xs, test_xs], dim=0).cpu().numpy().flatten()

lim = 7
truncated_xs = xs
truncated_xs[truncated_xs > lim] = lim
truncated_xs[truncated_xs < -lim] = -lim
plt.hist(truncated_xs, bins=1000, density=True, label="Data")
plot_xs= torch.linspace(-lim, lim, 1000).reshape(-1, 1)
plt.xlim(-lim, lim)
#plt.hist(xs, bins=10000, density=True)
#dist_heads = [NormalHead(device=device), StudentTHead(device=device), SkewedStudentTHead(device=device), BLogistic(64 - 2, device=device)]
dist_heads = [NormalHead(device=device), BLogistic(64 - 2, device=device)]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_colors = [colors[0], colors[3]]
# save model parameters to file
for i, dist_head in enumerate(dist_heads):
    if os.path.exists(dist_head.__class__.__name__ + ".npy"):
        print("Loading", dist_head.__class__.__name__)
        param = np.load(dist_head.__class__.__name__ + ".npy")
        param = torch.tensor(param, device=device)
    else:
        print("Training", dist_head.__class__.__name__)
        param = train_dist(dist_head, train_xs, 1e-2, 5000, device=device)
        np.save(dist_head.__class__.__name__ + ".npy", param.detach().cpu().numpy())
    pdf = dist_head.pdf(plot_xs, param).detach().cpu().numpy().flatten()
    plt.plot(plot_xs.detach().cpu().numpy().flatten(), pdf, label=dist_head.__class__.__name__, color = plot_colors[i])
plt.rcParams.update({'font.size': 14})
#fig = plt.figure(figsize=(6, 6))
plt.title("PDF of Model Distributions\n in SPY Minutely Returns")
plt.xlabel("Return")
plt.ylabel("PDF")
plt.legend()
plt.savefig("PDFs.png")