from classifier import *


class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = 0
        self.dense = 0

    def forward(self, x):
        x = self.conv(x)
        x = Function.flatten(x)
        return self.dense(x)


class ResNetClassifier(NNClassifier):
    def __init__(self, training_l: LabeledDataset, validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None):
        """
        :param training_l: the labeled dataset
        :param validation: the validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(ResNetClassifier, self).__init__(ResNet, training_l, validation, training_ul)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def _predict(self, x: Tensor):
        return self._original_36_argmax(x)

