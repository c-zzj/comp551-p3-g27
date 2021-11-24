import pickle
import torch
from torch import Tensor
from typing import List, Tuple, Callable, Dict, Any, Union, Optional, Iterable
from torch.utils.data import Dataset

TRAINING_X = "images_l.pkl"
TRAINING_Y = "labels_l.pkl"
TRAINING_UL = "images_ul.pkl"
TEST = "images_test.pkl"

Preprocess = Callable[[Tensor, Tensor], Tensor]


class LabeledDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, name: Optional[str] = "Training"):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
                f"Number of entries: {len(self.x)}---"

    def partition(self, splits: List[int]) -> List[Dataset]:
        """
        partition the dataset given the indices
        :param splits: list of indices i_1, ..., i_n. i_n must be strictly smaller than len(self)
        :return: Partitioned subsets, e.g., the first set contains x_0,...,x_{i_1-1} for x
        """
        p = []
        start = 0
        for i, index in enumerate(splits):
            p.append(LabeledDataset(
                self.x[start:i],
                self.y[start:i],
                self.name + f'- {i}-th partition'
            ))
            start = i
        p.append(LabeledDataset(
            self.x[start:],
            self.y[start:],
            self.name + f'- last partition'
        ))
        return p


class UnlabeledDataset(Dataset):
    def __init__(self, x: Tensor, name: Optional[str] = None):
        self.x = x
        self.name = name

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.x)}---"

    def partition(self, splits: List[int]) -> List[Dataset]:
        """
        partition the dataset given the indices
        :param splits: list of indices i_1, ..., i_n. i_n must be strictly smaller than len(self)
        :return: Partitioned subsets, e.g., the first set contains x_0,...,x_{i_1-1} for x
        """
        p = []
        start = 0
        for i, index in enumerate(splits):
            p.append(UnlabeledDataset(
                self.x[start:i],
                self.name + f'- {i}-th partition'
            ))
            start = i
        p.append(UnlabeledDataset(
            self.x[start:],
            self.name + f'- last partition'
        ))
        return p


def read_train_labeled(folder_path: str) -> LabeledDataset:
    """
    read the labeled training set
    :param folder_path: path of the dataset folder
    :return: features, labels
    """
    with open(folder_path + '/' + TRAINING_X, 'rb') as f:
        x = pickle.load(f)
    with open(folder_path + '/' + TRAINING_Y, 'rb') as f:
        y = pickle.load(f)
    return LabeledDataset(x, y)


def read_train_unlabeled(folder_path: str) -> UnlabeledDataset:
    """
    read the unlabeled training set
    :param folder_path: path of the dataset folder
    :return: features
    """
    with open(folder_path + '/' + TRAINING_UL, 'rb') as f:
        x = pickle.load(f)
    return UnlabeledDataset(x)


def read_test(folder_path: str) -> UnlabeledDataset:
    """
    read the test set
    :param folder_path: path of the dataset folder
    :return: features
    """
    with open(folder_path + '/' + TEST, 'rb') as f:
        x = pickle.load(f)
    return UnlabeledDataset(x)


