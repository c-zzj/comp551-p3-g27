"""
A toy network used for debugging
feel free to modify
"""

from classifier import *


class ToyNet(Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding='same'),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 36)
        )

    def forward(self, x):
        x = self.conv(x)
        x = Function.flatten(x)
        return self.dense(x)


class ToyClassifier(NNClassifier):
    def __init__(self,
                 training_l: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 val_proportion: float = 0.1,):
        """
        :param training_l: the labeled dataset
        :param val_proportion: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(ToyClassifier, self).__init__(ToyNet, training_l, training_ul, val_proportion)
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
