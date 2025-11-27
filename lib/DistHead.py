from abc import abstractmethod
import torch
from torch.distributions import Normal, StudentT
import numpy as np

class DistHead:
    @abstractmethod
    def logpdf(self, xs, params):
        pass
    
    @abstractmethod
    def pdf(self, xs, params):
        pass

    @abstractmethod
    def num_params(self):
        pass

    @abstractmethod
    def get_variance(self, params):
        pass
    
class NormalHead(DistHead):
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
    
    def _get_dist(self, params):
        std = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        return Normal(0, std)
    
    def logpdf(self, xs, params):
        dist = self._get_dist(params)
        return dist.log_prob(xs)
    
    def pdf(self, xs, params):
        return torch.exp(self.logpdf(xs, params))
    
    def num_params(self):
        return 1

    def get_variance(self, params):
        std = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        return std ** 2

class StudentTHead(DistHead):
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
    
    def _get_dist(self, params):
        std = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        df = torch.nn.functional.softplus(params[:, 1]).reshape(-1, 1) + 1 # ensures well defined mean
        return StudentT(df, 0, std)
    
    def logpdf(self, xs, params):
        dist = self._get_dist(params)
        return dist.log_prob(xs)
    
    def pdf(self, xs, params):
        return torch.exp(self.logpdf(xs, params))
    
    def num_params(self):
        return 2

    def get_variance(self, params):
        df = torch.nn.functional.softplus(params[:, 1]).reshape(-1, 1) + 1 # ensures well defined mean
        std = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        return std ** 2 * df / (df - 2)

class SkewedStudentTHead(DistHead):
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
    
    def _process_params(self, params):
        std = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        df = 1 + torch.nn.functional.softplus(params[:, 1]).reshape(-1, 1) # ensures well defined mean
        skewness = torch.nn.functional.softplus(params[:, 2]).reshape(-1, 1)
        mean_correction = (skewness ** 2 - skewness ** -2) * 2 * df
        mean_correction /= (skewness + 1/skewness) * (df - 1) * torch.sqrt(torch.pi * df)
        mean_correction *= torch.exp(torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2))
        return std, df, skewness, mean_correction
    
    def logpdf(self, xs, params):
        std, df, skewness, mean_correction = self._process_params(params)
        z = xs / std + mean_correction

        # Piecewise rescale: positive side uses 1/g, negative side uses g
        scale_mult = torch.where(z >= 0, 1.0 / skewness, skewness)

        # Normalization constant: log C = log 2 - log(g + 1/g)
        logC = torch.log(2.0 / (skewness + 1.0 / skewness))

        base = StudentT(df=df, loc=0.0, scale=1.0)
        logpdf = base.log_prob(z * scale_mult) + logC - torch.log(std)
        return logpdf
    
    def pdf(self, xs, params):
        return torch.exp(self.logpdf(xs, params))
    
    def num_params(self):
        return 3

    def get_variance(self, params):
        std, df, skewness, mean_correction = self._process_params(params)
        return std ** 2 * (df / (df - 2) * (skewness ** 2 - 1 + skewness ** -2) - mean_correction ** 2)
