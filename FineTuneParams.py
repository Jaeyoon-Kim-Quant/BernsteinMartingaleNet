# output folder, dist type, optional parameter for dist type

import sys
import os
import numpy as np
import torch
import argparse
import cProfile
import pstats

cwd = os.getcwd()
root = cwd.split("BernsteinMartingaleNet")[0] + "BernsteinMartingaleNet"
if root not in sys.path:
    sys.path.append(root)

from lib.utils import train_model, get_normalized_data
from lib.BLogistic import BLogistic, MixedBLogistic
from lib.DistHead import NormalHead, StudentTHead, SkewedStudentTHead
from lib.Michenkow import AdjustedMichenkow
from lib.AttnLSTM import AttnLSTM
# get command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-o","--output_folder", type=str)
# parser.add_argument("-d","--dist_type", type=str)
# parser.add_argument("-p","--dist_param", type=str, default="")
# parser.add_argument("-f","--num_features", type=int, default=3)
# args = parser.parse_args()

def _output_folder_name(num_features):
    if num_features == 1:
        return "NoReg"
    elif num_features == 2:
        return "NoReg_RV"
    elif num_features == 3:
        return "NoReg_RV_Decay"

    raise NotImplementedError

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
# override manually

args.dist_type = "BLogistic" #"SkewedStudentT" #"BLogistic" #"SkewedStudentT" #StudentT Normal #SkewedStudentT #BLogistic
args.dist_param = "64"


def set_seed(seed=42):

    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before anything else
set_seed(42)  # or any other seed value

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

#feature_size = args.num_features # features in order: returns, realized variance, time
#assert feature_size >= 1 and feature_size <= 3

folder_path = root + "/MarketData/historical_data"

candidate_layer_sizes = [

    [128, 64, 32],

    [256, 256, 128, 64],
    [256, 128, 64, 32],
    [128, 64, 32, 16],
    [64, 32, 16, 8],

    [256, 128, 64],
    [64, 32, 16],

    [128, 64],
    [64, 32],

]


lr = 0.002
weight_decay = 0.000
num_steps = 800 + 1
batch_size = 1024
dropout_ratio = 0.2

for layer_size in candidate_layer_sizes:

    for feature_size in [3, 2, 1]:

        train_xs, train_ys, train_rv, dev_xs, dev_ys, dev_rv, test_xs, test_ys, test_rv = get_normalized_data(folder_path,
                                                                                                          feature_size,
                                                                                                          device, 0.2,
                                                                                                          0.2,
                                                                                                          force_recompute=True)

        # add data perturbation
        noise = 0.001
        noise = torch.tensor(noise, device=device)
        num_perturbations = 4
        synthetic_train_xs = []
        for i in range(num_perturbations):
            new_xs = train_xs.clone()
            new_xs[:, :, :2] = new_xs[:, :, :2] * torch.sqrt(1 - noise) + torch.randn_like(
                new_xs[:, :, :2]) * torch.sqrt(noise)
            synthetic_train_xs.append(new_xs)
        synthetic_train_xs = torch.concat(synthetic_train_xs, dim=0)

        synthetic_train_ys = torch.concat([train_ys.clone() for i in range(num_perturbations)], dim=0)
        total_train_xs = torch.concat([train_xs, synthetic_train_xs], dim=0)
        total_train_ys = torch.concat([train_ys, synthetic_train_ys], dim=0)
        # shuffle total_train_xs and total_train_ys
        indices = torch.randperm(total_train_xs.shape[0])
        total_train_xs = total_train_xs[indices]
        total_train_ys = total_train_ys[indices]

        print("total_train_xs.shape", total_train_xs.shape)

        architecture = AdjustedMichenkow
        model = architecture(dist_head, device, feature_size=feature_size,
                         layer_sizes=layer_size, dropout_ratio=dropout_ratio)

        layer_size_str = "-".join(list(map(lambda x: str(x), layer_size)))
        output_folder = (f"Results/LayerSizesCompare/SyntheticData/{args.dist_type}"
                         f"{_output_folder_name(feature_size)}_{layer_size_str}_{args.dist_param}")

        _, train_losses, dev_losses = train_model(model, total_train_xs, total_train_ys, dev_xs, dev_ys,
                                                  test_xs, test_ys,
                               lr, weight_decay, num_steps, batch_size=batch_size, device=device,
                               output_folder=output_folder)

