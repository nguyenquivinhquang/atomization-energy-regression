import torch
import torch.nn as nn


# import List
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MLP, self).__init__()
        print(input_size)
        self.fc = nn.Sequential()
        self.fc.add_module("fc0", nn.Linear(input_size, hidden_sizes[0]))
        for idx, (in_size, out_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.fc.add_module("fc{}".format(idx + 1), nn.Linear(in_size, out_size))
            self.fc.add_module("relu{}".format(idx + 1), nn.ReLU())

        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        x = self.fc(x)
        # x = self.sigmoid(x)
        x = self.fc3(x)
        return x
