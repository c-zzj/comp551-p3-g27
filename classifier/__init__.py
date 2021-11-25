from dataset import *
import torch.nn as nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2

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
                 val_proportion: float = 0.1,):
        self.device = device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.training_ul = training_ul
        self.val_proportion = val_proportion
        p = partition(training_l, [int(val_proportion * len(training_l))])
        self.validation = p[0]
        self.training_l = p[1]

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

    def extract_wrong_pred_entries(self, folder_path: Path, batch_size: int = 300):
        """

        :param folder_path:
        :param batch_size:
        :return:
        """
        loader = DataLoader(self.validation, batch_size=batch_size, shuffle=False)
        pred = Tensor([]).to(self.device)
        for i, data in enumerate(loader, 0):
            x = data[0].to(self.device)
            pred = torch.cat((pred, self._predict(x)), dim=0)
        true = self.validation.y.to(self.device)
        indices = torch.nonzero(torch.any(pred != true, dim=1))[:,0]

        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        for i in indices:
            entry = self.validation.x[i].numpy()
            torch.save(entry, Path(folder_path / f"{i}-wrong.jpg"))
            cv2.imwrite(str(Path(folder_path / f"{i}-wrong.jpg")), entry)


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
        self.training_message = 'No training message.'
        # temporary variable used for plugins to communicate
        self._tmp = {}

    def load_network(self,
                     folder_path: Path,
                     epoch: int):
        """
        :param network:
        :param folder_path:
        :param epoch:
        :return: a network callable that can be passed to the NNClassifier constructor
        """
        self.network.load_state_dict(torch.load(folder_path / f"{epoch}.params"))

    def set_optimizer(self, optimizer: OptimizerProfile):
        self.optim = optimizer.optim(self.network.parameters(), **optimizer.params)

    def set_loss(self, loss: Callable):
        self.loss = loss

    def reset_train_val(self, training_l: LabeledDataset, val_proportion: int):
        p = partition(training_l, [int(val_proportion * len(training_l))])
        self.validation = p[0]
        self.training_l = p[1]

    def train(self,
              epochs: int,
              batch_size: int,
              shuffle: bool = True,
              start_epoch = 1,
              plugins: Optional[List[TrainingPlugin]] = None,
              verbose: bool = True)\
            -> None:
        """
        Train the model up to the epochs given.
        There is no return value. Plugins are used to save model and record performances.
        :param verbose:
        :param start_epoch:
        :param epochs: number of epochs
        :param batch_size: batch size for training
        :param shuffle: whether or not to shuffle the training data
        :param plugins: training plugin that is run after each epoch
        :return: None
        """
        if verbose:
            s = ''
            s += "Model Summary:\n"
            s += repr(self.network) + '\n'
            s += f"Device used for training: {self.device}\n"
            s += f"Size of training set: {len(self.training_l)}\n"
            s += f"Size of validation set: {len(self.validation)}\n"
            self.training_message = s
            print(s)
        train_loader = DataLoader(self.training_l, batch_size=batch_size, shuffle=shuffle)
        # the following code adopted from the tutorial notebook
        for epoch in range(start_epoch, start_epoch + epochs):  # loop over the dataset multiple times
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
            if verbose:
                s = f"---{epoch} EPOCHS FINISHED---\n"
                self.training_message += s
                print(s, end='')
            if plugins:
                s = f"Plugin messages for epoch {epoch}:\n"
                self.training_message += s
                print(s, end='')
                for plugin in plugins:
                    plugin(self, epoch)
                self.training_message = '' # reset training message
        if verbose:
            s = f"Finished training all {epochs} epochs."
            self.training_message = s
            print(s)
        return

    def train_performance(self, metric: Metric, proportion: float = 0.025, batch_size=300):
        """
        Obtain the performance on a subset of the training set
        :param metric: the metric of performance
        :param proportion: the proportion of the subset to be checked
        :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
        :return:
        """
        loader = DataLoader(self.training_l, batch_size=batch_size, shuffle=True)
        pred = Tensor([]).to(self.device)
        true = Tensor([]).to(self.device)
        portion_to_check = int(proportion * len(self.training_l))
        for i, data in enumerate(loader, 0):
            if i * batch_size >= portion_to_check:
                break
            x = data[0].to(self.device)
            pred = torch.cat((pred, self._predict(x)), dim=0)
            true = torch.cat((true, data[1].to(self.device)), dim=0)
        return metric(pred, true)

    def val_performance(self, metric: Metric, batch_size=300):
        """
        Obtain the performance on a subset of the training set
        :param metric: the metric of performance
        :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
        :return:
        """
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


