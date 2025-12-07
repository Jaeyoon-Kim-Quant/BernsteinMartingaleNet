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
lim = 12
integral_xs= torch.linspace(-lim * 2, lim * 2, 1000000).reshape(-1, 1)
plt.xlim(-9, 11)
cdf_to_plot = np.linspace(0, 1, 10000)[1:-1]
data_quantiles = np.quantile(xs, cdf_to_plot)
#plt.hist(xs, bins=10000, density=True)
dist_heads = [NormalHead(device=device), StudentTHead(device=device), SkewedStudentTHead(device=device), BLogistic(64 - 2, device=device)]
# save model parameters to file
plt.plot(data_quantiles, data_quantiles, label="Data")
for dist_head in dist_heads:
    if os.path.exists(dist_head.__class__.__name__ + ".npy"):
        print("Loading", dist_head.__class__.__name__)
        param = np.load(dist_head.__class__.__name__ + ".npy")
        param = torch.tensor(param, device=device)
    else:
        print("Training", dist_head.__class__.__name__)
        param = train_dist(dist_head, train_xs, 1e-2, 5000, device=device)
        np.save(dist_head.__class__.__name__ + ".npy", param.detach().cpu().numpy())
    pdf = torch.softmax(dist_head.logpdf(integral_xs, param), dim=0)
    cdf = torch.cumsum(pdf, dim=0).detach().cpu().numpy().flatten()
    f = interp1d(cdf, integral_xs.detach().cpu().numpy().flatten())
    plt.plot(data_quantiles, f(cdf_to_plot), label=dist_head.__class__.__name__)


    #plt.plot(plot_xs.detach().cpu().numpy().flatten(), cdf.detach().cpu().numpy().flatten(), label=dist_head.__class__.__name__)
plt.rcParams.update({'font.size': 14})
#fig = plt.figure(figsize=(6, 6))
plt.title("QQ Plot of Model Quantiles vs Data Quantiles\n in SPY Minutely Returns")
plt.xlabel("Data Quantiles")
plt.ylabel("Model Quantiles")
plt.legend()
plt.savefig("QQPlot.png")