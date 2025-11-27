import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

cwd = os.getcwd()
root = cwd.split("BernsteinMartingaleNet")[0] + "BernsteinMartingaleNet"
if root not in sys.path:
    sys.path.append(root)

from lib.utils import train_model, get_normalized_data, train_dist
from lib.BLogistic import BLogistic, MixedBLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
from lib.Michenkow import MichenkowManytoMany, AdjustedMichenkow

device = torch.device('cuda')
feature_size = 3

folder_path = root + "/MarketData/historical_data"
train_xs, train_ys, train_rv, dev_xs, dev_ys, dev_rv, test_xs, test_ys, test_rv = get_normalized_data(folder_path, feature_size, device, 0.2, 0.2)
print(train_rv.mean(), dev_rv.mean(), test_rv.mean())
print(train_rv.median(), dev_rv.median(), test_rv.median())

architecture = AdjustedMichenkow
dist_head = BLogistic(16 - 2, device)
state_dict = torch.load(root + "/MichenkowResults/BLogistic16Adjust/final_model.pth")
#dist_head = SkewedStudentTHead(device)
#state_dict = torch.load(root + "/MichenkowResults/SkewedStudentTAdjust/final_model.pth")
#dist_head = NormalHead(device)
#state_dict = torch.load(root + "/MichenkowResults/NormalAdjust/final_model.pth")
model = architecture(dist_head, device, feature_size=3)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

xs =train_xs.clone()
rv =train_rv.clone()

with torch.no_grad():
    params = model.get_params(xs)
    pred_variance = [model.dist_head.get_variance(params[i]).detach().cpu().numpy() for i in range(train_xs.shape[0])]
pred_variance = np.array(pred_variance).reshape(xs.shape[0], xs.shape[1])
rv = rv.detach().cpu().numpy()

pred_vol = np.sqrt(pred_variance)
rv_vol = np.sqrt(rv)
mrse = np.sqrt(np.mean((pred_vol - rv_vol) ** 2))
print(f"MRSE: {mrse}")

pred_mean = np.mean(pred_variance, axis=0)
rv_mean = np.mean(rv, axis=0)
plt.plot(pred_mean, label=f"{dist_head.__class__.__name__} Predicted")
plt.plot(rv_mean, label="Actual")
plt.legend()
plt.title("Predicted vs Actual Mean")
plt.show()