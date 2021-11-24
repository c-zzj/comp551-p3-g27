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
                 val_proportion: int = 0.1,):
        """
        :param training_l: the labeled dataset
        :param val_proportion: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(ToyClassifier, self).__init__(ToyNet, training_l, training_ul, val_proportion)
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