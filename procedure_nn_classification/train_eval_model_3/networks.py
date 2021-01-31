import torch.nn as nn

POOL = nn.AvgPool1d

channel = 6


class NNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['n_input']
        hidden_dim_1 = config['n_hidden'] if 'n_hidden' in config else 512
        hidden_dim_2 = config['n_hidden'] if 'n_hidden' in config else 512
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


class UnirefCNN(nn.Module):
    def __init__(self, config):
        super(UnirefCNN, self).__init__()
        if 'n_character' not in config:
            raise Exception('character not in config')
        self.C = config['n_character']  # number of character
        self.M = config['n_input']  # dimension of the string
        self.embedding = config['n_output']
        self.input_channel = 1

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
        self.flat_size = self.M // 4096 * channel // self.input_channel
        print("# self.flat_size ", self.flat_size)
        self.fc1 = nn.Linear(self.flat_size, self.embedding)
        self.softmax = nn.Softmax(dim=-1)
        # self.fc2 = nn.Linear(self.flat_size, self.flat_size)

    def forward(self, x):
        N = len(x)
        x = x.view(-1, self.input_channel, self.M)
        x = self.conv(x)
        x = x.view(N, self.flat_size)
        # x = self.fc2(x)
        # x = torch.relu(x)
        x = self.fc1(x)
        x = self.softmax(x)

        return x
