import torch
import torch.nn as nn


class ReLUNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(ReLUNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        return self.layers(x)


class InvertProjector(nn.Module):
    def __init__(self, n_layers, n_in, n_hidden):
        super(InvertProjector, self).__init__()
        self.project_layers = nn.ModuleList([ReLUNet(n_in // 2, n_hidden, n_in // 2) for _ in range(n_layers)])

    def forward(self, x):
        """
        project x

        Args:
            x: [batch_size, seq_len, n_hidden * 2]

        Returns:

        """
        for i, project_layer in enumerate(self.project_layers):
            left, right = torch.chunk(x, chunks=2, dim=-1)
            if i % 2 == 0:
                x = torch.cat((left, right + project_layer(left)), dim=-1)
            else:
                x = torch.cat((left + project_layer(right), right), dim=-1)
        return x
