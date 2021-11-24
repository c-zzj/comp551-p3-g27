from dataset import *
import torch.nn as nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch


class Network(Module):
    @staticmethod
    def flatten():
        """
        :return: A function x -> x flattened
        """
        def num_flat_features(x):
            """
            flatten the
            :param x:
            :return:
            """
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return x.view(-1, num_flat_features(x))
        return num_flat_features


class OptimizerProfile:
    def __init__(self, optimizer: Callable[..., Optimizer],
                      parameters: Dict[str, Any]):
        self.optim = optimizer
        self.params = parameters


class Classifier:
    def __init__(self,
                 training_l: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 val_proportion: int = 0.1,):
        self.device = device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.training_ul = training_ul
        self.val_proportion = val_proportion
        partition = training_l.partition([int(val_proportion * len(training_l))])
        self.validation = partition[0]
        self.training_l = partition[1]

    def train_performance(self, metric):
        raise NotImplementedError

    def val_performance(self, metric):
        raise NotImplementedError

    def predict(self, x: Tensor):
        raise NotImplementedError


TrainingPlugin = Callable[[Any, int], None]


class NNClassifier(Classifier):
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
        super(NNClassifier, self).__init__(training_l, training_ul, val_proportion)
        self.network = network()
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def set_optimizer(self, optimizer: OptimizerProfile):
        self.optim = optimizer.optim(self.network.parameters(), **optimizer.params)

    def set_loss(self, loss: Callable[..., Module]):
        self.loss = loss

    def reset_train_val(self, training_l: LabeledDataset, val_proportion: int):
        partition = training_l.partition([int(val_proportion * len(training_l))])
        self.validation = partition[0]
        self.training_l = partition[1]

    def train(self,
              epochs: int,
              batch_size: int,
              shuffle = True,
              plugins: Optional[List[TrainingPlugin]] = None)\
            -> None:
        """

        :param epochs: number of epochs
        :param batch_size: batch size for training
        :param shuffle: whether or not to shuffle the training data
        :param plugins: training plugin that is run after each epoch
        :return:
        """
        train_loader = DataLoader(self.training_l, batch_size=batch_size, shuffle=shuffle)
        # the following code adopted from the tutorial notebook
        for epoch in range(1, epochs+1):  # loop over the dataset multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                outputs = self.network(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optim.step()
            if plugins:
                for plugin in plugins:
                    plugin(self, epoch)
        return

    def train_performance(self, metric):
        
        raise NotImplementedError

    def val_performance(self, metric):
        raise NotImplementedError

