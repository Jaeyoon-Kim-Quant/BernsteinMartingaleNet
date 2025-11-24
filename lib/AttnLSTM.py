import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnLSTM(nn.Module):
    def __init__(self, dist_head, device, feature_size=1):
        super().__init__()
        self.dist_head      = dist_head
        self.device         = device
        self.feature_size   = feature_size
        self.hidden_size    = 128

        print(f"\nInitializing AttnLSTM with dist={dist_head.__class__.__name__}, dof={dist_head.num_params()}")

        # Encoder LSTM (as in Bahdanau et al. 2014)
        self.encoder_lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.2)

        # Bahdanau attention mechanism (additive attention)
        # Alignment model: score(h_t, h_s) = v_a^T * tanh(W_a * h_s + U_a * h_t)
        self.decoder_attn = nn.Linear(self.hidden_size * 2, 1, bias=False)  # alignment scoring vector

        # Combine context vector with current hidden state
        self.fc = nn.Linear(self.hidden_size, dist_head.num_params())  # *2 because we concat context + hidden
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, y):
        params = self.get_params(x)

        logpdf = self.dist_head.logpdf(
            y.reshape(-1, 1), params.reshape(-1, params.shape[-1])
        )
        return -logpdf.mean()

    def get_params(self, x):
        # Encode sequence with encoder LSTM
        # x shape: (batch, seq_len, feature_size)
        batch_size, seq_len, _ = x.shape
        encoder_outputs, _ = self.encoder_lstm(x)
        # encoder_outputs shape: (batch, seq_len, hidden_size)
        encoder_outputs = self.dropout(encoder_outputs)
        
        # Initialize decoder hidden states: (num_layers, batch_size, hidden_size)
        s = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device)
        h = torch.zeros(1, batch_size, self.hidden_size, device=encoder_outputs.device)
        
        outputs = []
        for t in range(seq_len):
            # Get current decoder hidden state: (batch_size, hidden_size)
            decoder_state = s.squeeze(0)  # (batch_size, hidden_size)
            
            # Compute attention over encoder outputs up to timestep t
            num_encoder_steps = t + 1
            encoder_slice = encoder_outputs[:, :num_encoder_steps, :]  # (batch_size, t+1, hidden_size)
            # stack encoder slice and s
            stacked_s = torch.concat([s.reshape(batch_size, 1, self.hidden_size).expand(batch_size, num_encoder_steps, self.hidden_size), encoder_slice], dim=-1)
            energies = self.decoder_attn(stacked_s)  # (batch_size, t+1, 1)
            energies = energies.squeeze(-1)  # (batch_size, t+1)
            
            # Compute attention weights
            attention_weights = F.softmax(energies, dim=1)  # (batch_size, t+1)

            # Compute context vector as weighted sum (use bmm for memory efficiency)
            #print("t:", t, "attention_weights shape:", attention_weights.shape, "encoder_slice shape:", encoder_slice.shape)
            context = torch.sum(attention_weights.reshape(batch_size, num_encoder_steps, 1) * encoder_slice, dim=1)
            
            # Reshape context for LSTM: (batch_size, 1, hidden_size) for batch_first=True
            context_input = context.unsqueeze(1)  # (batch_size, 1, hidden_size)
            
            # Forward through decoder LSTM
            output, (s, h) = self.decoder_lstm(context_input, (s, h))
            # output: (batch_size, 1, hidden_size)
            # s, h: (1, batch_size, hidden_size)
            
            outputs.append(output.squeeze(1))  # (batch_size, hidden_size)
        
        # Stack outputs: (batch_size, seq_len, hidden_size)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        
        # Generate parameters
        params = self.fc(outputs)  # (batch_size, seq_len, num_params)
        return params

    def get_logpdf(self, x, sample_xs):
        params = self.get_params(x)
        return self.dist_head.logpdf(sample_xs, params)

    def get_pdf(self, x, sample_xs):
        return torch.exp(self.get_logpdf(x, sample_xs))