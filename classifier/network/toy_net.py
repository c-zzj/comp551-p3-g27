"""
used for debugging
"""

from classifier import *


class ToyNetwork(Network):
    def __init__(self):
        super(ToyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),

            Network.Flatten(),

            nn.Linear(16 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(16 * 14 * 14, 256),
        )

    def forward(self, x):
        return self.net(x)
