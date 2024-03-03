import torch
import torch.nn as nn
from torch.autograd import Variable


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        features = features.unsqueeze(-1).float()
        h0 = torch.zeros(self.num_layers, features.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, features.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(features, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out