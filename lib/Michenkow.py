import torch
import torch.nn as nn

# implements the Michenkow model from the paper "Forecasting Probability Distributions of Financial Returns with Deep Neural"
# https://arxiv.org/pdf/2508.18921
class MichenkowManytoMany(nn.Module):
    def __init__(self, dist_head, device, feature_size=1):
        super().__init__()
        self.dist_head      = dist_head
        self.device         = device
        self.feature_size   = feature_size
        #self.layer_sizes    = [128, 64, 32]
        self.layer_sizes    = [256, 128, 64]

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

class AdjustedMichenkow(nn.Module):
    def __init__(self, dist_head, device, feature_size=1):
        super().__init__()
        self.dist_head      = dist_head
        self.device         = device
        self.feature_size   = feature_size
        self.layer_sizes    = [128, 64, 32]
        self.layer_norm = True

        print(f"\nInitializing LSTM with dist={dist_head.__class__.__name__}, dof={dist_head.num_params()}")
        self.lstms = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList() if self.layer_norm else None

        in_dim = self.feature_size
        for h_dim in self.layer_sizes:
            # single-layer LSTM at each depth
            self.lstms.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=h_dim,
                    num_layers=1,
                    batch_first=True,
                )
            )

            # projection for residual if needed
            if in_dim == h_dim:
                self.projs.append(nn.Identity())
            else:
                self.projs.append(nn.Linear(in_dim, h_dim))

            if self.layer_norm:
                self.norms.append(nn.LayerNorm(h_dim))

            in_dim = h_dim

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(self.layer_sizes[-1], dist_head.num_params())
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, y):
        params = self.get_params(x)

        logpdf = self.dist_head.logpdf(
            y.reshape(-1, 1), params.reshape(-1, params.shape[-1])
        )
        return -logpdf.mean()

    def get_params(self, x):
        out = x
        states = []

        for i, (lstm, proj) in enumerate(zip(self.lstms, self.projs)):
            y, (hT, cT) = lstm(out)

            y = self.dropout(y)

            # residual add (with projection)
            res = proj(out)
            out = y + res

            if self.layer_norm:
                out = self.norms[i](out)

        params = self.fc(out)
        return params

    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.dist_head.logpdf(sample_xs, params)

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))