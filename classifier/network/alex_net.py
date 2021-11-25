from classifier import *


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
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


class AlexNetClassifier(NNClassifier):
    def __init__(self,
                 training_l: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 val_proportion: float = 0.1,):
        """
        :param training_l: the labeled dataset
        :param val_proportion: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(AlexNetClassifier, self).__init__(AlexNet, training_l, training_ul, val_proportion)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def _predict(self, x: Tensor):
        return self._argmax_prediction(x)

    def _argmax_prediction(self, x: Tensor):
        """
            argmax prediction - use the highest value as prediction
            :param x:
            :return:
        """
        self.network.eval()
        with torch.no_grad():
            pred = self.network(x[:, None, :].float())
        self.network.train()

        data_size = len(x)
        output = torch.zeros((data_size, 36), dtype=torch.int, device=self.device)
        numbers = pred[:, :10]
        letters = pred[:, 10:]
        num_pred = torch.argmax(numbers, dim=1)
        letter_pred = torch.argmax(letters, dim=1) + 10
        output[range(data_size), num_pred] = 1
        output[range(data_size), letter_pred] = 1
        return output
