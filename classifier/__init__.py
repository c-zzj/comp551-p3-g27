from dataset import *
import torch.nn as nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch
from pathlib import Path
import pandas as pd
import numpy as np

class Function:
    @staticmethod
    def flatten(x: Tensor):
        """
        flatten the tensor
        :param x:
        :return:
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return x.view(-1, num_features)


class Classifier:
    """
    Abstract Classifier
    """
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
        """
        to be implemented by concrete classifiers
        :param metric:
        :return:
        """
        raise NotImplementedError

    def val_performance(self, metric):
        """
        to be implemented by concrete classifiers
        :param metric:
        :return:
        """
        raise NotImplementedError

    def _predict(self, x: Tensor):
        """
        to be implemented by concrete classifiers
        :param x:
        :return:
        """
        raise NotImplementedError

    def save_test_result(self,
                         test_set: UnlabeledDataset,
                         folder_path: Path,
                         submission_number: int,
                         batch_size=300) -> None:
        # get predictions
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        pred = Tensor([]).to(self.device)
        for i, x in enumerate(loader, 0):
            x = x.to(self.device)
            pred = torch.cat((pred, self._predict(x)), dim=0)

        # type cast to concatenated string
        pred = pred.detach().to('cpu').numpy().astype(int).astype(bytearray)
        result = []
        for i, row in enumerate(pred):
            result.append([i, ''.join(row.astype(str))])
        result = pd.DataFrame(result)
        result.columns = ["# ID", "Category"]

        # save predictions
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        result.to_csv(folder_path / (str(submission_number) + '.csv'), index=False)
        return


class OptimizerProfile:
    def __init__(self, optimizer: Callable[..., Optimizer],
                      parameters: Dict[str, Any] = {}):
        self.optim = optimizer
        self.params = parameters


TrainingPlugin = Callable[[Any, int], None]
Metric = Callable[[Tensor, Tensor], float] # pred, true -> result. The higher the better


class NNClassifier(Classifier):
    """
    Abstract Network Classifier
    """
    def __init__(self,
                 network: Callable[[], Module],
                 training_l: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 val_proportion: int = 0.1,
                 ):
        """
        :param network: a function that gives a Network
        :param training_l: the labeled dataset
        :param val_proportion: proportion of validation set
        :param training_ul: (optional) the unlabeled dataset
        """
        super(NNClassifier, self).__init__(training_l, training_ul, val_proportion)
        self.network = network().to(self.device)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def load_network(self,
                     folder_path: Path,
                     epoch: int):
        """
        :param network:
        :param folder_path:
        :param epoch:
        :return: a network callable that can be passed to the NNClassifier constructor
        """
        self.network.load_state_dict(torch.load(folder_path / str(epoch)))

    def set_optimizer(self, optimizer: OptimizerProfile):
        self.optim = optimizer.optim(self.network.parameters(), **optimizer.params)

    def set_loss(self, loss: Callable):
        self.loss = loss

    def reset_train_val(self, training_l: LabeledDataset, val_proportion: int):
        partition = training_l.partition([int(val_proportion * len(training_l))])
        self.validation = partition[0]
        self.training_l = partition[1]

    def train(self,
              epochs: int,
              batch_size: int,
              shuffle: bool = True,
              plugins: Optional[List[TrainingPlugin]] = None,
              verbose: bool = True)\
            -> None:
        """
        Train the model up to the epochs given.
        There is no return value. Plugins are used to save model and record performances.
        :param epochs: number of epochs
        :param batch_size: batch size for training
        :param shuffle: whether or not to shuffle the training data
        :param plugins: training plugin that is run after each epoch
        :return: None
        """
        if verbose:
            pass

        train_loader = DataLoader(self.training_l, batch_size=batch_size, shuffle=shuffle)
        # the following code adopted from the tutorial notebook
        for epoch in range(1, epochs+1):  # loop over the dataset multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # turn input into format required by neural network forward pass
                inputs = inputs[:, None, :].float()
                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                self.loss.zero_grad()
                outputs = self.network(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optim.step()
            if plugins:
                for plugin in plugins:
                    plugin(self, epoch)
        if verbose:
            pass
        return

    def train_performance(self, metric: Metric, batch_size=300):
        loader = DataLoader(self.training_l, batch_size=batch_size, shuffle=False)
        pred = Tensor([]).to(self.device)
        for i, data in enumerate(loader, 0):
            x = data[0].to(self.device)
            pred = torch.cat((pred, self._predict(x)), dim=0)
        return metric(pred, self.training_l.y.to(self.device))

    def val_performance(self, metric: Metric, batch_size=300):
        loader = DataLoader(self.validation, batch_size=batch_size, shuffle=False)
        pred = Tensor([]).to(self.device)
        for i, data in enumerate(loader, 0):
            x = data[0].to(self.device)
            pred = torch.cat((pred, self._predict(x)), dim=0)
        return metric(pred, self.validation.y.to(self.device))

    def _predict(self, x: Tensor):
        """
        to be implemented by concrete networks
        :param x:
        :return:
        """
        raise NotImplementedError


