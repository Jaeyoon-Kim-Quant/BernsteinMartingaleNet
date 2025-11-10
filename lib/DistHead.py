from abc import abstractmethod
import torch
from torch.distributions import Normal, StudentT

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
    
class NormalHead(DistHead):
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
    
    def _get_dist(self, params):
        mean = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        std = torch.nn.functional.softplus(params[:, 1]).reshape(-1, 1)
        return Normal(mean, std)
    
    def logpdf(self, xs, params):
        dist = self._get_dist(params)
        return dist.log_prob(xs)
    
    def pdf(self, xs, params):
        dist = self._get_dist(params)
        return dist.prob(xs)
    
    def num_params(self):
        return 2

class StudentTHead(DistHead):
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
    
    def _get_dist(self, params):
        mean= torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        std = torch.nn.functional.softplus(params[:, 1]).reshape(-1, 1)
        df = torch.nn.functional.softplus(params[:, 2]).reshape(-1, 1)
        return StudentT(df, mean, std)
    
    def logpdf(self, xs, params):
        dist = self._get_dist(params)
        return dist.log_prob(xs)
    
    def pdf(self, xs, params):
        dist = self._get_dist(params)
        return dist.prob(xs)
    
    def num_params(self):
        return 3

class SkewedStudentTHead(DistHead):
    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')
    
    def _get_dist(self, params):
        mean = torch.nn.functional.softplus(params[:, 0]).reshape(-1, 1)
        std = torch.nn.functional.softplus(params[:, 1]).reshape(-1, 1)
        df = torch.nn.functional.softplus(params[:, 2]).reshape(-1, 1)
        skewness = torch.nn.functional.softplus(params[:, 3]).reshape(-1, 1)
        return mean, std, df, skewness
    
    def logpdf(self, xs, params):
        mean, std, df, skewness = self._get_dist(params)
        #zs = (xs - mean) / std
        #dist = StudentT(df, 0, 1)
        #skew_mult = skewness ** -torch.sign(zs)
        #return dist.logpdf(zs * skew_mult) + torch.log(2 / (skewness + skewness ** -1)) - torch.log(std)

        z = (xs - mean) / std

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
        return 4

