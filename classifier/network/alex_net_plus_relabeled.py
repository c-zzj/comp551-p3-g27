from classifier import *


class AlexNetPlus(Module):
    def __init__(self):
        super(AlexNetPlus, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )
        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 36)
        )

    def forward(self, x):
        x = self.conv(x)
        x = Function.flatten(x)
        return self.dense(x)


class AlexNetPlusClassifier(NNClassifier):
    def __init__(self,
                 network: Module,
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None):
        """
        :param training_l: the labeled dataset
        :param validation: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(AlexNetPlusClassifier, self).__init__(AlexNetPlus, training_l, validation, training_ul)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def _predict(self, x: Tensor):
        return self._original_36_argmax(x)

