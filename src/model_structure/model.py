import torch.nn as nn
import torch


# Add Batch normalization and DropOuts

class NeuralNetwork(nn.Module):
    def __init__(self, in_channels=None, output=None) -> None:
        super(NeuralNetwork, self).__init__()
        # Input [1,256,256]
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=1, padding=(1, 1),
                      bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=(1, 1),
                      bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=1, padding=(1, 1),
                      bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Input [32, 32, 32]
        self.flatten = nn.Flatten()

        # Input [32*32*32]
        self.fully_connected_block = nn.Sequential(
            nn.Linear(in_features=32 * 32 * 32, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output, bias=True)
        )
        # output [3]

    def forward(self, x):
        features = self.conv_block(x)
        flatten = self.flatten(features)
        result = self.fully_connected_block(flatten)
        return result
