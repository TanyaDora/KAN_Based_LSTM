import torch
import torch.nn as nn
from fastkan import FastKAN as KAN

class KanLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kan_hidden_size):
        super(KanLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define MLPs for input and hidden state transformations
        self.KAN_x = KAN([input_size, kan_hidden_size, hidden_size], num_grids=5)
        self.KAN_h = KAN([hidden_size, kan_hidden_size, hidden_size], num_grids=5)

        # Biases for the gates
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x, hidden):
        h, c = hidden

        # Compute gate values
        i = torch.sigmoid(self.KAN_x(x) + self.KAN_h(h))
        f = torch.sigmoid(self.KAN_x(x) + self.KAN_h(h))
        o = torch.sigmoid(self.KAN_x(x) + self.KAN_h(h))
        c_tilde = torch.tanh(self.KAN_x(x) + self.KAN_h(h))
        
        c_next = f * c + i * c_tilde
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class KanLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs, kan_hidden_size = 5, num_layers=1):
        super(KanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList(
            [KanLSTMCell(input_size if i == 0 else hidden_size, hidden_size, kan_hidden_size) for i in range(num_layers)]
        )
        self.fc = KAN([hidden_size, num_outputs], num_grids=5)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        else:
            h_0, c_0 = hidden

        h_n = []
        c_n = []

        for i, cell in enumerate(self.lstm_cells):
            h, c = h_0[i], c_0[i]
            outputs = []
            for t in range(seq_len):
                h, c = cell(x[:, t, :], (h, c))
                outputs.append(h)
            h_n.append(h)
            c_n.append(c)
            x = torch.stack(outputs, dim=1)

        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        x = self.fc(x)
        return x, (h_n, c_n)