from data import *
import torch.nn as nn
from torch.nn import Module, CrossEntropyLoss, functional
from torch.optim import SGD, Optimizer, Adam
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2


TrainingPlugin = Callable[[Any, int], None]
Metric = Callable[[Tensor, Tensor], float] # pred, true -> result. The higher the better


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

    @staticmethod
    def label_to_36(y: Tensor, device: device):
        """
        convert probabilities y from 260 labels to original 26 labels. If input in 36 labels, return it as is
        :param y:
        :param device:
        :return:
        """
        if y.size()[1] == 36:
            return y
        output = torch.zeros((len(y), 36), dtype=torch.float, device=device)
        for number in range(10):
            prob = torch.zeros(len(y), 10, dtype=torch.float, device=device)
            for letter in range(26):
                prob += y[range(len(y)), letter * 10 + number]
            output[:, number] = prob

        for letter in range(26):
            prob = torch.zeros(len(y), 10, dtype=torch.float, device=device)
            for number in range(10):
                prob += y[range(len(y)), letter * 10 + number]
            output[:, letter + 10] = prob
        return output

    @staticmethod
    def label_to_36_argmax(y: Tensor, device: device):
        """

        :param y:
        :param device:
        :return:
        """
        if y.size()[1] == 36:
            return y
        data_size = len(y)
        index = torch.argmax(y, dim=1)
        num = index % 10
        letter = torch.div(index, Tensor([10]).to(device), rounding_mode='floor') + 10
        output = torch.zeros((data_size, 36), dtype=torch.int, device=device)
        output[range(data_size), num.to(torch.long)] = 1
        output[range(data_size), letter.to(torch.long)] = 1
        return output

    @staticmethod
    def label_to_260(y: Tensor, device: device):
        """

        :param y:
        :param device:
        :return:
        """
        if y.size()[1] == 260:
            return y
        numbers = torch.softmax(y[:,:10], dim=1)
        letters = torch.softmax(y[:,10:], dim=1)
        output = torch.zeros((len(y), 260), dtype=torch.float, device=device)
        for num in range(10):
            for letter in range(26):
                output[range(len(y)),  letter * 10 + num] = numbers[range(len(y)), num] * letters[range(len(y)), letter]
        return output

class Classifier:
    """
    Abstract Classifier
    """
    def __init__(self,
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 ):
        self.device = device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.training_l = training_l
        self.validation = validation
        self.training_ul = training_ul

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
            pred = torch.cat((pred, self.predict(x)), dim=0)
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
            pred = torch.cat((pred, self.predict(x)), dim=0)
        return metric(pred, self.validation.y.to(self.device))

    def predict(self, x: Tensor):
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
            pred = torch.cat((pred, Function.label_to_36_argmax(self.predict(x), self.device)), dim=0)

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
        save wrongly labeled entries and a post-mortem summary to the folder
        :param folder_path:
        :param batch_size:
        :return:
        """
        loader = DataLoader(self.validation, batch_size=batch_size, shuffle=False)
        pred = Tensor([]).to(self.device)
        true = Tensor([]).to(self.device)
        for i, data in enumerate(loader, 0):
            x = data[0].to(self.device)
            y = data[1].to(self.device)
            pred = torch.cat((pred, Function.label_to_36_argmax(self.predict(x), device=self.device)), dim=0)
            true = torch.cat((true, Function.label_to_36_argmax(y,device=self.device)), dim=0)
        indices = torch.nonzero(torch.any(pred != true, dim=1))[:,0]

        if not folder_path.exists():
            folder_path.mkdir(parents=True)

        with open(str(folder_path / 'post-mortem.txt'), 'w+') as f:
            for i in indices:
                correct_num, correct_letter = torch.nonzero(true[i][:10])[0, 0], torch.nonzero(true[i][10:])[0, 0]
                predicted_num, predicted_letter = torch.nonzero(pred[i][:10])[0, 0], torch.nonzero(pred[i][10:])[0, 0]
                correct_letter, predicted_letter = chr(ord('a') + correct_letter), chr(ord('a') + predicted_letter)
                f.write(f'---{i}-th image---\n')
                f.write(f'correct: {correct_num}{correct_letter}\tpredicted: {predicted_num}{predicted_letter}\n')
                f.write(f'--------------------------\n')
                entry = self.validation.x[i].numpy()
                torch.save(entry, Path(folder_path / f"{i}-wrong.jpg"))
                cv2.imwrite(str(Path(folder_path / f"{i}-wrong.jpg")), entry)


class OptimizerProfile:
    def __init__(self, optimizer: Callable[..., Optimizer],
                      parameters: Dict[str, Any] = {}):
        self.optim = optimizer
        self.params = parameters


class NNClassifier(Classifier):
    """
    Abstract Network Classifier
    """
    def __init__(self,
                 network: Module,
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 ):
        """
        :param network: a function that gives a Network
        :param training_l: the labeled data
        :param validation: the validation set
        :param training_ul: (optional) the unlabeled data
        """
        super(NNClassifier, self).__init__(training_l, validation, training_ul)
        self.network = network.to(self.device)
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
        for epoch in range(start_epoch, start_epoch + epochs):  # loop over the data multiple times
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

    def train_semisupervised(self,
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
        mixed_dataset = MixedDataset(self.training_l, self.training_ul, epoch_ratio_over_unlabeled=2)
        train_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=shuffle)
        # the following code adopted from the tutorial notebook
        for epoch in range(start_epoch, start_epoch + epochs):  # loop over the data multiple times
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                labeled_x, labels, unlabeled_x, unlabeled_x_processed = \
                    data[0].to(self.device), data[1].to(self.device), data[2], data[3]

                unlabeled_x = torch.cat(unlabeled_x, dim=0).to(self.device)
                unlabeled_x_processed = torch.cat(unlabeled_x_processed, dim=0).to(self.device)

                pseudolabels = self.predict(unlabeled_x).to(torch.float)

                # turn input into format required by neural network forward pass
                labeled_x = labeled_x[:, None, :].float()
                unlabeled_x_processed = unlabeled_x_processed[:, None, :].float()
                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                self.loss.zero_grad()
                loss = self.loss(self.network(labeled_x), labels) + self.loss(self.network(unlabeled_x_processed), pseudolabels)
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

    def train_fixmatch(self,
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
        mixed_dataset = MixedDataset(self.training_l, self.training_ul, epoch_ratio_over_unlabeled=1)
        # the following code adopted from the tutorial notebook
        for epoch in range(start_epoch, start_epoch + epochs):  # loop over the data multiple times
            loader = DataLoader(self.training_ul, batch_size=batch_size, shuffle=False)
            pred = Tensor([]).to(self.device)
            for i, data in enumerate(loader, 0):
                x = data[0].to(self.device)
                pred = torch.cat((pred, self.predict(x)), dim=0)
            pred_set = UnlabeledDataset(pred)
            wrapper = WrapperDataset([pred_set, mixed_dataset], len(self.training_l))

            train_loader = DataLoader(wrapper, batch_size=batch_size, shuffle=shuffle)
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                pseudolabels = data[0].to(self.device)
                data = data[1]

                labeled_x, labels, unlabeled_x, unlabeled_x_processed = \
                    data[0].to(self.device), data[1].to(self.device), data[2], data[3]

                unlabeled_x_processed = torch.cat(unlabeled_x_processed, dim=0).to(self.device)


                # turn input into format required by neural network forward pass
                labeled_x = labeled_x[:, None, :].float()
                unlabeled_x_processed = unlabeled_x_processed[:, None, :].float()
                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                self.loss.zero_grad()
                loss = self.loss(self.network(labeled_x), labels) + self.loss(self.network(unlabeled_x_processed), pseudolabels)
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

    def predict(self, x: Tensor):
        """
        to be implemented by concrete networks
        :param x:
        :return:
        """
        raise NotImplementedError

    def _original_36_argmax(self, x: Tensor):
        """
        argmax prediction - use the highest value as prediction
        :param x:
        :return:
        """
        self.network.eval()
        with torch.no_grad():
            pred = self.network(x[:, None, :].float())
        self.network.train()

        if pred.size()[1] == 260:
            return self._260_argmax_to_36(pred)

        data_size = len(x)
        numbers = pred[:, :10]
        letters = pred[:, 10:]
        num_pred = torch.argmax(numbers, dim=1)
        letter_pred = torch.argmax(letters, dim=1) + 10

        output = torch.zeros((data_size, 36), dtype=torch.int, device=self.device)
        output[range(data_size), num_pred] = 1
        output[range(data_size), letter_pred] = 1
        return output

    def _260_argmax_to_36(self, pred: Tensor):
        """
        argmax prediction - use the highest value as prediction
        :param x:
        :return:
        """
        if pred.size()[1] != 260:
            return pred
        data_size = len(pred)
        index = torch.argmax(pred, dim=1)
        num = index % 10
        letter = (torch.div(index, Tensor([10]).to(self.device), rounding_mode='floor')) + 10
        output = torch.zeros((data_size, 36), dtype=torch.int, device=self.device)
        output[range(data_size), num.to(torch.long)] = 1
        output[range(data_size), letter.to(torch.long)] = 1
        return output

    def _260_argmax(self, x: Tensor):
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
        index = torch.argmax(pred, dim=1)

        output = torch.zeros((data_size, 260), dtype=torch.int, device=self.device)
        output[range(data_size), index] = 1
        return output