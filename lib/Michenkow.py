import torch
import torch.nn as nn

# implements the Michenkow model from the paper "Forecasting Probability Distributions of Financial Returns with Deep Neural"
# https://arxiv.org/pdf/2508.18921
class Michenkow(nn.Module):
    def __init__(self, context_window, dist_head, device):
        super().__init__()
        self.context_window = context_window
        self.dist_head      = dist_head
        self.device         = device
        self.layer_sizes    = [128, 64, 32]

        print(f"\nInitializing LSTM with context_window={context_window}, dist={dist_head.__class__.__name__}, dof={dist_head.num_params()}")

        # LSTM layers (3 layers with decreasing neurons: 128 -> 64 -> 32)
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=self.layer_sizes[0],
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        self.lstm2 = nn.LSTM(
            input_size=self.layer_sizes[0],
            hidden_size=self.layer_sizes[1],
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        self.lstm3 = nn.LSTM(
            input_size=self.layer_sizes[1],
            hidden_size=self.layer_sizes[2],
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        self.dropout = nn.Dropout(0.02)
        self.fc = nn.Linear(self.layer_sizes[2], dist_head.num_params())
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)


    def forward(self, x, y):
        params = self.get_params(x)
        logpdf = self.dist_head.logpdf(
            y, params
        )
        return -logpdf.mean()

    def get_params(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)

        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x, _ = self.lstm3(x)
        x = self.dropout(x)

        x = x[:, -1, :]

        params = self.fc(x)
        return params

    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.dist_head.logpdf(sample_xs, params)

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))

class MichenkowManytoMany(nn.Module):
    def __init__(self, dist_head, device, feature_size=1):
        super().__init__()
        self.dist_head      = dist_head
        self.device         = device
        self.feature_size   = feature_size
        self.layer_sizes    = [128, 64, 32]

        print(f"\nInitializing LSTM with dist={dist_head.__class__.__name__}, dof={dist_head.num_params()}")

        # LSTM layers (3 layers with decreasing neurons: 128 -> 64 -> 32)
        self.lstm1 = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.layer_sizes[0],
            num_layers=1,
            batch_first=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.layer_sizes[0],
            hidden_size=self.layer_sizes[1],
            num_layers=1,
            batch_first=True,
        )

        self.lstm3 = nn.LSTM(
            input_size=self.layer_sizes[1],
            hidden_size=self.layer_sizes[2],
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2) # should this be 0.2?

        self.fc = nn.Linear(self.layer_sizes[2], dist_head.num_params())
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, y):
        params = self.get_params(x)

        logpdf = self.dist_head.logpdf(
            y.reshape(-1, 1), params.reshape(-1, params.shape[-1])
        )
        return -logpdf.mean()

    def get_params(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)

        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x, _ = self.lstm3(x)
        x = self.dropout(x)

        #x = x[:, -1, :]

        params = self.fc(x)
        return params

    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.dist_head.logpdf(sample_xs, params)

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))