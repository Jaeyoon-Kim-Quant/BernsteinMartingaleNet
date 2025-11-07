# take command line arguments
# output folder, dist type, optional parameter for dist type

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

cwd = os.getcwd()
root = cwd.split("BernsteinMartingaleNet")[0] + "BernsteinMartingaleNet"
if root not in sys.path:
    sys.path.append(root)

from lib.utils import train_model, get_sequence_data_by_month
from lib.BLogistic import BLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
from lib.Michenkow import Michenkow

# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o","--output_folder", type=str)
parser.add_argument("-d","--dist_type", type=str)
parser.add_argument("-p","--dist_param", type=str, default="")
args = parser.parse_args()

assert torch.cuda.is_available()
device = torch.device('cuda')

context_window = 60

dist_head = None
if args.dist_type == "Normal":
    dist_head = NormalHead(device)
elif args.dist_type == "StudentT":
    dist_head = StudentTHead(device)
elif args.dist_type == "SkewedStudentT":
    dist_head = SkewedStudentTHead(device)
elif args.dist_type == "BLogistic":
    dof = int(args.dist_param)
    dist_head = BLogistic(dof - 2, device)

folder_path = root + "/MarketData/historical_data"
context_window = 60
train_xs, train_ys, dev_xs, dev_ys, test_xs, test_ys = get_sequence_data_by_month(folder_path, context_window, 8, 8)

train_xs = torch.tensor(train_xs[:, :, 0], device=device)
train_ys = torch.tensor(train_ys, device=device).reshape(-1, 1)
dev_xs = torch.tensor(dev_xs[:, :, 0], device=device)
dev_ys = torch.tensor(dev_ys, device=device).reshape(-1, 1)
test_xs = torch.tensor(test_xs[:, :, 0], device=device)
test_ys = torch.tensor(test_ys, device=device).reshape(-1, 1)

std = torch.sqrt((train_ys**2).mean())
train_xs = train_xs / std
dev_xs = dev_xs / std
test_xs = test_xs / std
train_ys = train_ys / std
dev_ys = dev_ys / std
test_ys = test_ys / std


lr = 0.002
decay_step = 50
decay_gamma = 0.5
weight_decay = 0
num_steps = 1
batch_size = 512 * 8

model = Michenkow(context_window, dist_head, device)

model, train_losses, dev_losses = train_model(model, train_xs, train_ys, dev_xs, dev_ys,
    lr, weight_decay, num_steps, batch_size=batch_size, device=device, output_folder=args.output_folder, lr_decay_step=decay_step, lr_decay_gamma=decay_gamma)