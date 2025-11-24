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
from lib.Michenkow import Michenkow, MichenkowManytoMany
from lib.AttnLSTM import AttnLSTM
# get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o","--output_folder", type=str)
parser.add_argument("-d","--dist_type", type=str)
parser.add_argument("-p","--dist_param", type=str, default="")
parser.add_argument("-f","--num_features", type=int, default=3)
args = parser.parse_args()

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
assert feature_size >= 1 and feature_size <= 3

folder_path = root + "/MarketData/historical_data"
train_xs, train_ys, dev_xs, dev_ys, test_xs, test_ys = get_normalized_data(folder_path, feature_size, device, 0.2, 0.2)

lr = 0.001
decay_step = 200
decay_gamma = 0.5
weight_decay = 0
weight_decay = 0.000
num_steps = 400 + 1
batch_size = 32

#model = MichenkowManytoMany(dist_head, device, feature_size=feature_size)
architecture = AttnLSTM
model = architecture(dist_head, device, feature_size=feature_size)
transfer_learning = False
if transfer_learning:
    model = architecture(BLogistic(dof-2, device), device, feature_size=feature_size)
    state_dict = torch.load(root + "/MichenkowResults/BLogisticAttention/final_model.pth")
    model.load_state_dict(state_dict)

    # transfer learning from base model
    #with torch.no_grad():
    #    base_params = base_model.fc.weight.data
    #    #model.fc.weight.data = base_params
    #    #model.fc.bias.data = base_model.fc.bias.data
    #    model.lstm1.weight_hh_l0.data = base_model.lstm1.weight_hh_l0.data
    #    model.lstm1.weight_ih_l0.data = base_model.lstm1.weight_ih_l0.data
    #    model.lstm2.weight_hh_l0.data = base_model.lstm2.weight_hh_l0.data
    #    model.lstm2.weight_ih_l0.data = base_model.lstm2.weight_ih_l0.data
    #    model.lstm3.weight_hh_l0.data = base_model.lstm3.weight_hh_l0.data
    #    model.lstm3.weight_ih_l0.data = base_model.lstm3.weight_ih_l0.data
    #freeze lstm layers
    #for param in model.lstm1.parameters():
    #    param.requires_grad = False
    #for param in model.lstm2.parameters():
    #    param.requires_grad = False
    #for param in model.lstm3.parameters():
    #    param.requires_grad = False

    #iid_xs = train_xs[:, :, 0].flatten()
    #new_params = train_dist(dist_head, iid_xs, 0.1, 300, device)
    #with torch.no_grad():
    #    model.fc.bias.copy_(new_params)

model, train_losses, dev_losses = train_model(model, train_xs, train_ys, dev_xs, dev_ys, test_xs, test_ys,
                                              lr, weight_decay, num_steps, batch_size=batch_size, device=device,
                                              output_folder=args.output_folder, lr_decay_step=decay_step, lr_decay_gamma=decay_gamma)