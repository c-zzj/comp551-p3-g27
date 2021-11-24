from classifier import *


def save(folder_path: str, step: int = 1) -> TrainingPlugin:
    """
    :param folder_path: the path of the folder to save the model
    :param step: number of epochs to save the model
    :return: a plugin that saves the model after each step
    """
    def s(clf: NNClassifier, epoch: int) -> None:
        """
        save the model
        :param clf: the neural network classifier
        :param epoch: number of current epoch
        :return: None
        """
        if epoch % step == 0:
            torch.save(clf.network.state_dict(), folder_path)
    return s


