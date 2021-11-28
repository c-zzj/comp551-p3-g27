import pickle
import torch
from torch import Tensor, device
from typing import List, Tuple, Callable, Dict, Any, Union, Optional, Iterable
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch import randperm

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
    def __init__(self, x: Tensor, x_processed: Optional[Any] = None, name: Optional[str] = "Unlabeled data"):
        self.x = x
        self.x_processed = x_processed
        self.name = name

    def __getitem__(self, index):
        if self.x_processed:
            return self.x[index], self.x_processed[index]
        return self.x[index]

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.x)}---"

class WrapperDataset(Dataset):
    def __init__(self, datasets: List[Dataset], length: int):
        self.datasets = datasets
        self.len = length

    def __getitem__(self, index):
        return [dataset[index] for dataset in self.datasets]

    def __len__(self):
        return self.len

class MixedDataset(Dataset):
    def __init__(self, labeled: LabeledDataset,
                 unlabeled: UnlabeledDataset,
                 epoch_ratio_over_unlabeled: int = 3,
                 name: Optional[str] = "data for semisupervised learning"):
        self.labeled_x = labeled.x
        self.labeled_y = labeled.y
        self.unlabeled_x = unlabeled
        self.name = name
        self.epoch_ratio_over_unlabeled = epoch_ratio_over_unlabeled

    def __getitem__(self, index):
        unlabeled_x_multiple = [self.unlabeled_x[(index * self.epoch_ratio_over_unlabeled + i) % len(self.unlabeled_x)][0] for i in
                    range(self.epoch_ratio_over_unlabeled)]
        unlabeled_x_processed_multiple = [self.unlabeled_x[(index * self.epoch_ratio_over_unlabeled + i) % len(self.unlabeled_x)][1] for i in
                    range(self.epoch_ratio_over_unlabeled)]

        return self.labeled_x[index], self.labeled_y[index], \
               unlabeled_x_multiple, unlabeled_x_processed_multiple


    def __len__(self):
        return len(self.labeled_x)

    def __str__(self):
        return f"---Dataset name: {self.name}---" \
               f"Number of entries: {len(self.labeled_x)}---"


def partition(data: Union[LabeledDataset, UnlabeledDataset], splits: List[int], shuffle=True) \
    -> List[Union[LabeledDataset, UnlabeledDataset]]:
    """
    partition the data given the indices
    :param shuffle:
    :param data:
    :param splits: list of indices i_1, ..., i_n. i_n must be strictly smaller than len(self)
    :return: Partitioned subsets, e.g., the first set contains x_0,...,x_{i_1-1} for x
    """
    if type(data) == LabeledDataset:
        p = []
        start = 0
        x = data.x
        y = data.y
        if shuffle:
            indices = randperm(len(data))
            x = x[indices]
            y = y[indices]
        for i, index in enumerate(splits):
            p.append(LabeledDataset(
                x[start:index],
                y[start:index],
                data.name + f'- {i}-th partition'
            ))
            start = index
        p.append(LabeledDataset(
            x[start:],
            y[start:],
            data.name + f'- last partition'
        ))
        return p
    else:
        p = []
        start = 0
        x = data.x
        if shuffle:
            indices = randperm(len(data))
            x = x[indices]
        for i, index in enumerate(splits):
            p.append(UnlabeledDataset(
                x[start:i],
                data.name + f'- {i}-th partition'
            ))
            start = i
        p.append(UnlabeledDataset(
            x[start:],
            data.name + f'- last partition'
        ))
        return p


def read_train_labeled(folder_path: Path) -> LabeledDataset:
    """
    read the labeled training set
    :param folder_path: path of the data folder
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
    :param folder_path: path of the data folder
    :return: features
    """
    with open(str(folder_path / TRAINING_UL), 'rb') as f:
        x = torch.from_numpy(pickle.load(f))
    return UnlabeledDataset(x, name="Unlabeled Training")


def read_test(folder_path: Path) -> UnlabeledDataset:
    """
    read the test set
    :param folder_path: path of the data folder
    :return: features
    """
    with open(str(folder_path / TEST), 'rb') as f:
        x = torch.from_numpy(pickle.load(f))
    return UnlabeledDataset(x, name="Test")


