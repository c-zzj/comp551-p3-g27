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
    def __init__(self, training_l: LabeledDataset, validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None):
        """
        :param training_l: the labeled dataset
        :param validation: the validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(ToyClassifier, self).__init__(ToyNet(), training_l, validation, training_ul)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def predict(self, x: Tensor):
        return self._original_36_argmax(x)

