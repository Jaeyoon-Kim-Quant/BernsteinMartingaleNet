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
from lib.BLogistic import SkewedBLogistic, train_blogistic, SplineLogistic, train_spline_logistic

folder_path = r"../MarketData/historical_data"
context_window = 60
X, Y = get_sequence_data(folder_path, context_window, force_recompute=False)
print("data shape", X.shape, Y.shape)

dof = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(context_window, context_window)
        self.fc2 = nn.Linear(context_window, dof)
        with torch.no_grad():
            self.fc1.weight *= 0.01
            self.fc2.weight *= 0.01
            self.fc1.bias *= 0
            self.fc2.bias *= 0
        self.blogistic = SkewedBLogistic(dof - 3, device=device)

    def forward(self, x, y):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return -self.blogistic.logpdf_vectorized(y, params[:, :-2], params[:, -2], params[:, -1]).mean()
     
    def print_nan_params(self, x, y):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        where_nan = params.isnan().any(dim=1)
        print("where_nan", where_nan.shape)
        if where_nan.any():
            print("params", params[where_nan, :-2])
            print("params", params[where_nan, -2])
            print("params", params[where_nan, -1])

    def get_params(self, x):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return params
    
    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.blogistic.logpdf(sample_xs, params[:, :-2].flatten(), params[:, -2], params[:, -1])

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))

class SplineNN(nn.Module):
    def __init__(self):
        super(SplineNN, self).__init__()
    def __init__(self):
        super(SplineNN, self).__init__()
        self.fc1 = nn.Linear(context_window, context_window)
        self.fc2 = nn.Linear(context_window, dof)
        with torch.no_grad():
            self.fc1.weight *= 0.001
            self.fc2.weight *= 0.001
            self.fc1.bias *= 0
            self.fc2.bias *= 0
        self.dist = SplineLogistic(dof, device=device)

    def forward(self, x, y):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return -self.dist.logpdf_vectorized(y, params[:, :-1], params[:, -1]).mean()

    def get_params(self, x):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return params
    
    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.dist.logpdf(sample_xs, params[:, :-1].flatten(), params[:, -1])

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))

class ScaleNN(nn.Module):
    def __init__(self, offset):
        super(ScaleNN, self).__init__()
        self.fc1 = nn.Linear(context_window, context_window)
        self.fc2 = nn.Linear(context_window, 1)
        with torch.no_grad():
            self.fc1.weight *= 0.001
            self.fc2.weight *= 0.001
            self.fc1.bias *= 0
            self.fc2.bias *= 0
        self.blogistic = SkewedBLogistic(dof - 3, device=device)
        self.offset = offset

    def forward(self, x, y):
        layer1 = torch.relu(self.fc1(x))
        scale = self.fc2(layer1)
        return -self.blogistic.logpdf_vectorized(y, self.offset[0].reshape(1, -1), scale, self.offset[2].reshape(1, -1)).mean()

    def get_params(self, x):
        layer1 = torch.relu(self.fc1(x))
        params = self.fc2(layer1)
        return params
    
    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.blogistic.logpdf(sample_xs, params[:, :-2].flatten(), params[:, -2], params[:, -1])

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))

def train_spline_nn(train_X, train_Y, dev_X, dev_Y, lr, num_steps, device: torch.device = None):
    iid_train_steps = 500
    iid_lr = 0.05
    _, offset = train_spline_logistic(train_Y.flatten(), dof, iid_lr, iid_train_steps, device=device)
    model = SplineNN().to(device)
    with torch.no_grad():
        model.fc2.bias[:-1].copy_(offset[0])
        model.fc2.bias[-1].copy_(offset[1])

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = train_X.shape[0] // 10
    print("num batches", train_X.shape[0] // batch_size)
    train_losses = []
    dev_losses = []
    for step in range(num_steps):
        # run mini-batch training
        for i in range(0, train_X.shape[0], batch_size):
            batch_X = train_X[i:i+batch_size, :]
            batch_Y = train_Y[i:i+batch_size, :]
            loss = model(batch_X, batch_Y)
            if loss.isnan():
                #model.print_nan_params(batch_X, batch_Y)
                exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 100 == 0:
            train_loss = model(train_X, train_Y)
            dev_loss = model(dev_X, dev_Y)
            train_losses.append(train_loss.item())
            dev_losses.append(dev_loss.item())

            print(f"Step {step}, Train Loss: {train_loss.item():.4f}, Dev Loss: {dev_loss.item():.4f}")
    return model, train_losses, dev_losses

def train_scale_nn(train_X, train_Y, dev_X, dev_Y, lr, num_steps, device: torch.device = None):
    iid_train_steps = 1000
    iid_lr = 0.05
    _, offset = train_blogistic(train_Y.flatten(), dof, iid_lr, iid_train_steps, allow_skew=True, device=device)
    model = ScaleNN(offset).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = train_X.shape[0]
    print("num batches", train_X.shape[0] // batch_size)
    train_losses = []
    dev_losses = []
    for step in range(num_steps):
        # run mini-batch training
        for i in range(0, train_X.shape[0], batch_size):
            batch_X = train_X[i:i+batch_size, :]
            batch_Y = train_Y[i:i+batch_size, :]
            loss = model(batch_X, batch_Y)
            if loss.isnan():
                #model.print_nan_params(batch_X, batch_Y)
                exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 100 == 0:
            train_loss = model(train_X, train_Y)
            dev_loss = model(dev_X, dev_Y)
            train_losses.append(train_loss.item())
            dev_losses.append(dev_loss.item())

            print(f"Step {step}, Train Loss: {train_loss.item():.4f}, Dev Loss: {dev_loss.item():.4f}")
    return model, train_losses, dev_losses

def train_simple_nn(train_X, train_Y, dev_X, dev_Y, lr, num_steps, device: torch.device = None, transfer_learn = True):
    model = SimpleNN().to(device)
    if transfer_learn:
        iid_train_steps = 500
        iid_lr = 0.05
        _, offset = train_blogistic(train_Y.flatten(), dof, iid_lr, iid_train_steps, allow_skew=True, device=device)
        with torch.no_grad():
            model.fc2.bias[:-2].copy_(offset[0])
            model.fc2.bias[-2].copy_(offset[1])
            model.fc2.bias[-1].copy_(offset[2])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = train_X.shape[0]
    print("num batches", train_X.shape[0] // batch_size)
    train_losses = []
    dev_losses = []
    for step in range(num_steps):
        # run mini-batch training
        for i in range(0, train_X.shape[0], batch_size):
            batch_X = train_X[i:i+batch_size, :]
            batch_Y = train_Y[i:i+batch_size, :]
            loss = model(batch_X, batch_Y)
            if loss.isnan():
                #model.print_nan_params(batch_X, batch_Y)
                exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 100 == 0:
            train_loss = model(train_X, train_Y)
            dev_loss = model(dev_X, dev_Y)
            train_losses.append(train_loss.item())
            dev_losses.append(dev_loss.item())

            print(f"Step {step}, Train Loss: {train_loss.item():.4f}, Dev Loss: {dev_loss.item():.4f}")
    return model, train_losses, dev_losses



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

#lr = 1e-3
#num_steps = 5000
#scale_model, _, _ = train_scale_nn(train_X, train_Y, dev_X, dev_Y, lr, num_steps, device=device)
#torch.save(scale_model.state_dict(), "scale_nn_model16.pth")

lr = 1e-3
num_steps = 5000
#model, train_losses, dev_losses = train_spline_nn(train_X, train_Y, dev_X, dev_Y, lr, num_steps, device=device)
model, train_losses, dev_losses = train_simple_nn(train_X, train_Y, dev_X, dev_Y, lr, num_steps, device=device)
torch.save(model.state_dict(), "simple_nn_model_spline16_context60.pth")
#
#plt.plot(train_losses, label="Train Loss")
#plt.plot(dev_losses, label="Dev Loss")
#plt.legend()
#plt.show()
# load the model
model = SimpleNN().to(device)
model.load_state_dict(torch.load("simple_nn_model_spline16_context60.pth"))

plot_xs = torch.linspace(-8, 8, 10000, device=device)
nplots = 10
#ntest = 1000
## calculate kl div for each example
#params = model.get_params(dev_X[:ntest, :])
#params_mean = params.mean(dim=0)
#params_diff = (params[:, :-2] - params_mean[:-2]).norm(dim=1)
#top_idx = np.argsort(params_diff.detach().cpu().numpy())[-nplots:]
blogistic = SkewedBLogistic(dof - 3, device=device)
for idx in range(nplots):
    # get color wheel
    color = plt.cm.viridis(idx / nplots)
    param = model.get_params(dev_X[idx, :].reshape(1, -1))
    original_scale = torch.nn.functional.softplus(param[:, -2]).item()
    plot_ys = model.get_pdf(dev_X[idx, :].reshape(1, -1), plot_xs) * original_scale
    plt.plot(plot_xs.cpu().numpy() / original_scale, plot_ys.detach().cpu().numpy(), color=color)
    plt.scatter(dev_Y[idx, :].cpu().numpy() / original_scale, model.get_pdf(dev_X[idx, :].reshape(1, -1), dev_Y[idx, :].reshape(1, 1)).detach().cpu().numpy() * original_scale, color=color)
plt.show()

for idx in range(nplots):
    # get color wheel
    color = plt.cm.viridis(idx / nplots)
    plot_ys = model.get_pdf(dev_X[idx, :].reshape(1, -1), plot_xs)
    plt.plot(plot_xs.cpu().numpy(), plot_ys.detach().cpu().numpy(), color=color)
    plt.scatter(dev_Y[idx, :].cpu().numpy(), model.get_pdf(dev_X[idx, :].reshape(1, -1), dev_Y[idx, :].reshape(1, 1)).detach().cpu().numpy(), color=color)

plt.xlabel("Return")
plt.ylabel("PDF")
plt.show()