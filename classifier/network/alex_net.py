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
                 val_proportion: int = 0.1,):
        """
        :param training_l: the labeled dataset
        :param val_proportion: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(AlexNetClassifier, self).__init__(AlexNet, training_l, training_ul, val_proportion)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def _predict(self, x: Tensor):
        """
        COPIED FROM ING TIAN
        !!!!!!!!!!!!!!!!!!!! TO BE REIMPLEMENTED !!!!!!!!!!!!!!!!!!!!!!!!!!!
        :param x:
        :return:
        """
        number_of_instances = len(x)
        self.network.eval()
        with torch.no_grad():
            output = self.network(x[:, None, :].float())
        self.network.train()
        number_predicts = output[:, :10]
        alphabet_predicts = output[:, 10:]

        number_predicts_mask = torch.zeros((number_of_instances, 10), dtype=torch.bool)
        alphabet_predicts_mask = torch.zeros((number_of_instances, 26), dtype=torch.bool)
        max_number_indices = torch.argmax(number_predicts, dim=1)
        max_alphabet_indices = torch.argmax(alphabet_predicts, dim=1)
        number_predicts_mask[range(number_of_instances), max_number_indices] = True
        alphabet_predicts_mask[range(number_of_instances), max_alphabet_indices] = True

        number_predicts[number_predicts_mask] = 1
        number_predicts[~number_predicts_mask] = 0
        alphabet_predicts[alphabet_predicts_mask] = 1
        alphabet_predicts[~alphabet_predicts_mask] = 0
        return output


