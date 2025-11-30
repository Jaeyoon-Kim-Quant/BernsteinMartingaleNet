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

from lib.utils import train_model, get_normalized_data, train_dist
from lib.BLogistic import BLogistic, MixedBLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
from lib.Michenkow import Michenkow, AdjustedMichenkow
# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o","--output_folder", type=str)
parser.add_argument("-d","--dist_type", type=str)
parser.add_argument("-p","--dist_param", type=str, default="")
parser.add_argument("-f","--num_features", type=int, default=3)
parser.add_argument("-a","--architecture", type=str, default="AdjustedMichenkow")
args = parser.parse_args()

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

assert torch.cuda.is_available()
device = torch.device('cuda')

dist_head = None
if args.dist_type == "Normal":
    dist_head = NormalHead(device)
elif args.dist_type == "StudentT":
    dist_head = StudentTHead(device)
elif args.dist_type == "SkewedStudentT":
    dist_head = SkewedStudentTHead(device)
elif args.dist_type == "BLogistic":
    if args.dist_param == "":
        raise ValueError("BLogistic dist type requires a degree parameter")
    dof = int(args.dist_param)
    dist_head = BLogistic(dof - 2, device)
elif args.dist_type == "MixedBLogistic":
    if args.dist_param == "":
        raise ValueError("MixedBLogistic dist type requires a degree parameter")
    dof = int(args.dist_param)
    dist_head = MixedBLogistic(dof, device)
else:
    raise ValueError(f"Invalid dist type: {args.dist_type}")

feature_size = args.num_features # features in order: returns, realized variance, time
print(f"Feature size: {feature_size}")
assert feature_size >= 1 and feature_size <= 3

if args.architecture == "AdjustedMichenkow":
    architecture = AdjustedMichenkow
elif args.architecture == "Michenkow":
    architecture = Michenkow
else:
    raise ValueError(f"Invalid architecture: {args.architecture}")

folder_path = root + "/MarketData/historical_data"
train_xs, train_ys, train_rv, dev_xs, dev_ys, dev_rv, test_xs, test_ys, test_rv = get_normalized_data(folder_path, feature_size, device, 0.2, 0.2)

lr = 0.002
weight_decay = 0.000
num_steps = 800 + 1
batch_size = 1024

model = architecture(dist_head, device, feature_size=feature_size)
model, train_losses, dev_losses = train_model(model, train_xs, train_ys, dev_xs, dev_ys, test_xs, test_ys,
                                              lr, weight_decay, num_steps, batch_size=batch_size, device=device,
                                              output_folder=args.output_folder, verbose=False)