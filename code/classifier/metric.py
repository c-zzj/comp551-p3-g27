from classifier import *


def accuracy(pred: Tensor, true: Tensor) -> float:
    N = true.size()[0]
    err = torch.count_nonzero(torch.any(pred != true, dim=1)).item()
    return (N-err) / N