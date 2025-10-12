import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from math import comb
import scipy.interpolate as intrp
from scipy.stats import norm
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
import torch

xs = np.linspace(0, 1, 10000)
nbasis = 8
bpolys = np.array([comb(nbasis, i) * xs**i * (1 - xs)**(nbasis - i) for i in range(nbasis + 1)]).T

#plt.plot(xs, bpolys, color="black")

these_knots = np.linspace(0,1,nbasis - 2)

numpyknots = np.concatenate(([0,0,0],these_knots,[1,1,1]))
y_py = np.zeros((xs.shape[0], len(these_knots)+2))
for i in range(len(these_knots)+2):
    y_py[:,i] = intrp.BSpline(numpyknots, (np.arange(len(these_knots)+2)==i).astype(float), 3, extrapolate=False)(xs)

# PyTorch version
these_knots_torch = torch.linspace(0, 1, nbasis - 2, dtype=torch.float64)
torch_knots = torch.cat([torch.zeros(3, dtype=torch.float64), these_knots_torch, torch.ones(3, dtype=torch.float64)])
y_py_torch = torch.zeros(xs.shape[0], len(these_knots_torch) + 2, dtype=torch.float64)
xs_torch = torch.from_numpy(xs).to(torch.float64)
for i in range(len(these_knots_torch) + 2):
    coeffs = torch.zeros(len(these_knots_torch) + 2, dtype=torch.float64)
    coeffs[i] = 1.0
    interp_points = torch.linspace(0, 1, len(these_knots_torch) + 2, dtype=torch.float64)
    coeffs_tuple = natural_cubic_spline_coeffs(interp_points, coeffs)
    spline = NaturalCubicSpline(*coeffs_tuple)
    y_py_torch[:, i] = spline(xs_torch)

# Plot both for comparison
plt.figure()
plt.plot(xs, y_py, color="red", label="scipy BSpline (numpy)")
plt.plot(xs, y_py_torch.numpy(), '--', color="blue", label="torchcubicspline (pytorch)")
plt.legend()
plt.title("Comparison of BSpline Basis: Numpy/Scipy vs PyTorch")
plt.show()


#plt.plot(xs,y_py, color="red")

#gaussian_partition = np.zeros((xs.shape[0], nbasis + 1))
#centers = np.linspace(0, 1, nbasis)
#scale = 1/(nbasis * 2)
#for i in range(nbasis):
#    gaussian_partition[:,i] = norm.pdf(xs, loc=centers[i], scale=scale)
#gaussian_partition = gaussian_partition / np.sum(gaussian_partition, axis=1, keepdims=True)
#plt.plot(xs, gaussian_partition, color="purple")
#
#plt.show()

#spline_total = np.sum(y_py, axis=1)
#bpoly_total = np.sum(bpolys, axis=1)
#print(np.mean(spline_total), np.mean(bpoly_total))
#plt.plot(xs, spline_total, color="blue")
#plt.plot(xs, bpoly_total, color="green")
#plt.show()