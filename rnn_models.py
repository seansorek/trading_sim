import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        self.hidden_size = hidden_size

    def forward(self, X):
        out, _ = self.rnn(X)
        last = out[:, -1, :]
        return self.fc(last)
