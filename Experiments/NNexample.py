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
from lib.utils import get_sequence_data, train_model
from lib.BLogistic import BLogistic, train_blogistic, SplineLogistic, train_spline_logistic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get data
folder_path = r"../MarketData/historical_data"
context_window = 10
X, Y = get_sequence_data(folder_path, context_window, force_recompute=False)
dof = 16
simple_X = torch.tensor(X[:, :, 0], device=device)
simple_Y = torch.tensor(Y, device=device).reshape(-1, 1)

dev_size = 10000
np.random.seed(0)
indices = np.random.permutation(simple_X.shape[0])

dev_indices = indices[:dev_size]
train_indices = indices[dev_size:]
train_X = simple_X[train_indices]
train_Y = simple_Y[train_indices]
dev_X = simple_X[dev_indices, :]
dev_Y = simple_Y[dev_indices]

std = train_Y.std()
train_X = train_X / std
train_Y = train_Y / std
dev_X = dev_X / std
dev_Y = dev_Y / std
print("train_X", train_X.shape, "train_Y", train_Y.shape, "dev_X", dev_X.shape, "dev_Y", dev_Y.shape)
print("std", std, train_X.std())

class SimpleNN(nn.Module):
    def __init__(self, context_window=context_window, dof=dof, use_naive_pdf = False):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(context_window, context_window)
        self.fc2 = nn.Linear(context_window, dof)
        self.use_naive_pdf = use_naive_pdf
        self.blogistic = BLogistic(dof - 2, device=device)

    def forward(self, x, y):
        params = self.get_params(x)
        if self.use_naive_pdf:
            return -self.blogistic.naive_logpdf(y, params[:, :-1], params[:, -1]).mean()
        else:
            return -self.blogistic.logpdf(y, params[:, :-1], params[:, -1]).mean()

    def get_params(self, x):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return params
    
    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.blogistic.logpdf(sample_xs, params[:, :-1].reshape(1, -1), params[:, -1])

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))
    
    def transfer_learn(self, offset):
        with torch.no_grad():
            self.fc2.bias[:-1].copy_(offset[0])
            self.fc2.bias[-1].copy_(offset[1])

lr = 0.1
weight_decay = 0.0
num_steps = 1000
use_naive_pdf = False
transfer_learn = False

torch.manual_seed(0)
model = SimpleNN(use_naive_pdf=use_naive_pdf).to(device)
if transfer_learn:
    iid_train_steps = 300
    iid_lr = 0.05
    _, offset = train_blogistic(train_Y, dof, iid_lr, iid_train_steps, device=device)
    model.transfer_learn(offset)

model, train_losses, dev_losses = train_model(model, train_X, train_Y, dev_X, dev_Y, lr, weight_decay, num_steps, device=device)
torch.save(model.state_dict(), "simple_nn_model_spline16_context10.pth")
model = SimpleNN().to(device)
model.load_state_dict(torch.load("simple_nn_model_spline16_context10.pth"))

plot_xs = torch.linspace(-20, 20, 100000, device=device)
nplots = 10

for idx in range(nplots):
    # get color wheel
    color = plt.cm.viridis(idx / nplots)
    pdf = model.get_pdf(dev_X[idx, :].reshape(1, -1), plot_xs)
    plt.plot(plot_xs.cpu().numpy(), pdf.detach().cpu().numpy(), color=color)
    plt.scatter(dev_Y[idx, :].cpu().numpy(), model.get_pdf(dev_X[idx, :].reshape(1, -1), dev_Y[idx, :].reshape(1, 1)).detach().cpu().numpy(), color=color)

plt.xlabel("Return")
plt.ylabel("PDF")
plt.show()