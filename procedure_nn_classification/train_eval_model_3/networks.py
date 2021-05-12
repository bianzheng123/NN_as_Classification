import torch.nn as nn
import torch

POOL = nn.AvgPool1d

channel = 6
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)


class TwoBlock512Dim(nn.Module):
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


class OneBlock2048Dim(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['n_input']
        hidden_dim_1 = 2048
        output_dim = config['n_output']
        dropout_probability = 0.1

        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),

            nn.Dropout(p=dropout_probability),

            nn.Linear(hidden_dim_1, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config['n_input']
        hidden_dim = self.input_dim
        output_dim = config['n_output']
        dropout_probability = 0.1
        self.input_channel = 2
        self.flat_size = self.input_dim // self.input_channel * channel

        self.pre_mutual_layer = nn.Sequential(
            nn.Conv1d(self.input_channel, channel, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
        )
        self.separate_layer = nn.Sequential(
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(channel),
        )
        self.mutual_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),

            nn.Linear(self.flat_size, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        N = len(x)
        x = x.view(-1, self.input_channel, self.input_dim).float()
        x = self.pre_mutual_layer(x)
        x = self.separate_layer(x) + x
        x = x.view(N, self.flat_size)

        x = self.mutual_layer(x)
        return x


class TwoBlock8192DimNoBnDropout(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['n_input']
        hidden_dim_1 = 8192
        hidden_dim_2 = 8192
        output_dim = config['n_output']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),

            nn.Linear(hidden_dim_2, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if 'n_character' not in config:
            raise Exception('n_character not in config')
        self.C = config['n_character']  # number of character
        self.M = config['n_input']  # dimension of the string
        self.embedding = config['n_output']
        self.input_channel = 1
        hidden_dim_1 = 512
        hidden_dim_2 = 512
        dropout_probability = 0.1

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.ReLU(),

            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.ReLU(),

            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.ReLU(),

            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            POOL(2),
        )

        # Size after pooling
        self.flat_size = self.M // 4096 * self.C // self.input_channel * channel
        print("# self.flat_size ", self.flat_size)

        self.layer = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),

            nn.Linear(hidden_dim_2, self.embedding),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        N = len(x)
        x = x.view(-1, self.input_channel, self.M).float()
        x = self.conv(x)
        x = x.view(N, self.flat_size)
        x = self.layer(x)

        return x
