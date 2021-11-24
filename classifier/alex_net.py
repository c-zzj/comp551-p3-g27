from classifier import *

class AlexNet(Network):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            Network.flatten(),

            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 36)
        )

    def forward(self, x):
        return self.net(x)


class AlexNetClassifier(NNClassifier):
    def __init__(self,
                 network: Callable[[], Network],
                 training_l: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 val_proportion: int = 0.1,):
        """

        :param network: a function that gives a Network
        :param training_l: the labeled dataset
        :param val_proportion: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(AlexNetClassifier, self).__init__(AlexNet, training_l, training_ul, val_proportion)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

