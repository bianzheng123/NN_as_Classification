import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['n_input']
        hidden_dim_1 = 512
        hidden_dim_2 = 512
        output_dim = config['n_output']
        dropout_probability = 0.1
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),

            nn.Linear(hidden_dim_2, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
