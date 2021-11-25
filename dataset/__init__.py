import pickle
import torch
from torch import Tensor
from typing import List, Tuple, Callable, Dict, Any, Union, Optional, Iterable
from torch.utils.data import Dataset
from pathlib import Path

TRAINING_X = "images_l.pkl"
TRAINING_Y = "labels_l.pkl"
TRAINING_UL = "images_ul.pkl"
TEST = "images_test.pkl"


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


class UnlabeledDataset(Dataset):
    def __init__(self, x: Tensor, name: Optional[str] = "Unlabeled dataset"):
        self.x = x
        self.name = name

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.x)}---"


def partition(data: Union[LabeledDataset, UnlabeledDataset], splits: List[int]) \
    -> List[Union[LabeledDataset, UnlabeledDataset]]:
    """
    partition the dataset given the indices
    :param data:
    :param splits: list of indices i_1, ..., i_n. i_n must be strictly smaller than len(self)
    :return: Partitioned subsets, e.g., the first set contains x_0,...,x_{i_1-1} for x
    """
    if type(data) == LabeledDataset:
        p = []
        start = 0
        for i, index in enumerate(splits):
            p.append(LabeledDataset(
                data.x[start:index],
                data.y[start:index],
                data.name + f'- {i}-th partition'
            ))
            start = index
        p.append(LabeledDataset(
            data.x[start:],
            data.y[start:],
            data.name + f'- last partition'
        ))
        return p
    else:
        p = []
        start = 0
        for i, index in enumerate(splits):
            p.append(UnlabeledDataset(
                data.x[start:i],
                data.name + f'- {i}-th partition'
            ))
            start = i
        p.append(UnlabeledDataset(
            data.x[start:],
            data.name + f'- last partition'
        ))
        return p


def read_train_labeled(folder_path: Path) -> LabeledDataset:
    """
    read the labeled training set
    :param folder_path: path of the dataset folder
    :return: features, labels
    """
    with open(str(folder_path / TRAINING_X), 'rb') as f:
        x = torch.from_numpy(pickle.load(f))
    with open(str(folder_path / TRAINING_Y), 'rb') as f:
        y = torch.from_numpy(pickle.load(f))
    return LabeledDataset(x, y)


def read_train_unlabeled(folder_path: Path) -> UnlabeledDataset:
    """
    read the unlabeled training set
    :param folder_path: path of the dataset folder
    :return: features
    """
    with open(str(folder_path / TRAINING_UL), 'rb') as f:
        x = torch.from_numpy(pickle.load(f))
    return UnlabeledDataset(x, "Unlabeled Training")


def read_test(folder_path: Path) -> UnlabeledDataset:
    """
    read the test set
    :param folder_path: path of the dataset folder
    :return: features
    """
    with open(str(folder_path / TEST), 'rb') as f:
        x = torch.from_numpy(pickle.load(f))
    return UnlabeledDataset(x, "Test")


