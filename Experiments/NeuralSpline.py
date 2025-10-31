import torch
from nflows.transforms.splines.rational_quadratic import rational_quadratic_spline
import sys
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.BLogistic import get_ppf, train_blogistic
from lib.utils import load_data
import pickle
# Set device (GPU if available, otherwise CPU)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xs = load_data("../MarketData/spy_historical_data_20250929.csv")
xs_torch = torch.tensor(xs, device=device)

class NeuralSpline:
    def __init__(self, num_bins, device: torch.device = None):
        self.num_bins = num_bins
        self.device = device
    
    def _run_spline(self, xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        scale = torch.nn.functional.softplus(raw_scale)
        normal_ps = torch.distributions.Normal(0, scale).cdf(xs)
        unnormalized_widths_expanded = unnormalized_widths.unsqueeze(0).expand(len(xs), -1)
        unnormalized_heights_expanded = unnormalized_heights.unsqueeze(0).expand(len(xs), -1)
        unnormalized_derivatives_expanded = unnormalized_derivatives.unsqueeze(0).expand(len(xs), -1)
        outputs, logabsdet = rational_quadratic_spline(
            inputs=normal_ps,
            unnormalized_widths=unnormalized_widths_expanded,
            unnormalized_heights=unnormalized_heights_expanded,
            unnormalized_derivatives=unnormalized_derivatives_expanded,
            inverse=False
        )
        return outputs, logabsdet
    
    def cdf(self, xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        outputs, logabsdet = self._run_spline(xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
        return outputs
    
    def logpdf(self, xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        outputs, logabsdet = self._run_spline(xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
        scale = torch.nn.functional.softplus(raw_scale)
        log_normal_pdf = torch.distributions.Normal(0, scale).log_prob(xs)
        return logabsdet + log_normal_pdf
    
    def pdf(self, xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        return torch.exp(self.logpdf(xs, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives))
    
    def get_knots(self, raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        knots = torch.cumsum(torch.softmax(unnormalized_widths, dim=0), dim=0)[:-1]
        print(knots)
        scale = torch.nn.functional.softplus(raw_scale)
        return torch.distributions.Normal(0, scale).icdf(knots)
    

def train_neural_spline(xs_torch, num_bins, lr, num_steps, device: torch.device = None):
    init_scale = 0.00
    raw_scale = torch.nn.Parameter(torch.randn(1, device=device) * init_scale)
    unnormalized_widths = torch.nn.Parameter(torch.randn(num_bins, device=device) * init_scale)
    unnormalized_heights = torch.nn.Parameter(torch.randn(num_bins, device=device) * init_scale)
    unnormalized_derivatives = torch.nn.Parameter(torch.randn(num_bins + 1, device=device) * init_scale)

    neural_spline = NeuralSpline(num_bins, device)
    params = [raw_scale, unnormalized_widths, unnormalized_heights, unnormalized_derivatives]

    def loss(xs_torch):
        return -torch.mean(neural_spline.logpdf(xs_torch, *params))

    optimizer = torch.optim.Adam(params, lr=lr)
    costs = []
    sample_xs = torch.linspace(-10, 10, 10000, device=device)
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        loss_value = loss(xs_torch)
        loss_value.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss_value.item():.4f}")
            costs.append(loss_value.item())
    print(f"Step {step}, Final Loss: {loss_value.item():.4f}")
    return neural_spline, params, costs

num_bins = 5
num_steps = 10000
lr = 0.1
#neural_spline, params, costs = train_neural_spline(xs_torch, num_bins, lr, num_steps, device)
#with open("neural_spline_params_test.pkl", "wb") as f:
#    pickle.dump((neural_spline, params), f)
#plt.plot(costs)
#plt.show()

neural_spline, params = pickle.load(open("neural_spline_params.pkl", "rb"))
neural_spline = NeuralSpline(num_bins, device)

plot_xs = torch.linspace(-12, 12, 10000)
fig, ax = plt.subplots(1, 2)
ax[0].plot(plot_xs.numpy(), neural_spline.pdf(plot_xs.to(device), *params).cpu().detach().numpy(), label="NeuralSpline")
knots = neural_spline.get_knots(*params)
ax[0].scatter(knots.cpu().detach().numpy(), neural_spline.pdf(knots, *params).cpu().detach().numpy(), color="red", label="Knots")
ax[0].hist(xs, bins=512, density=True, label="Data")
ax[0].legend()
ax[0].set_title("PDF")

ax[1].plot(plot_xs.numpy(), neural_spline.cdf(plot_xs.to(device), *params).cpu().detach().numpy(), label="NeuralSpline")
ax[1].plot(np.sort(xs), np.linspace(0, 1, len(xs)), label="Data")
ax[1].legend()
ax[1].set_title("CDF")
plt.show()



