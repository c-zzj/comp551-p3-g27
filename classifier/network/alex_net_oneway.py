from classifier import *


class AlexNetOneWay(Module):
    def get_conv(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, (3, 3), padding='same'),
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

    def get_dense(self):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
        )

    def __init__(self):
        super(AlexNetOneWay, self).__init__()
        self.conv1 = self.get_conv()
        self.dense1 = self.get_dense()
        self.last_layer = nn.Linear(1024, 36)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = Function.flatten(x1)
        x1 = self.dense1(x1)
        return self.last_layer(x1)


class AlexNetOneWayClassifier(NNClassifier):
    def __init__(self, training_l: LabeledDataset, validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None):
        """
        :param training_l: the labeled dataset
        :param validation: the validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(AlexNetOneWayClassifier, self).__init__(AlexNetOneWay, training_l, validation, training_ul)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def _predict(self, x: Tensor):
        return self._original_36_argmax(x)

