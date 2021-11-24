from classifier import *

def save(folder_path: str, step: int = 1) -> TrainingPlugin:
    """
    :param folder_path: the path of the folder to save the model
    :param step: number of epochs to save the model
    :return: a plugin that saves the model after each step
    """
    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            torch.save(clf.network.state_dict(), folder_path + '/' + str(epoch))
    return plugin


def print_training_performance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param metric:
    :param batch_size:
    :param step:
    :return: a plugin that prints training performance after each step
    """
    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            print(clf.train_performance(metric, batch_size))
    return plugin


def print_validation_performance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param metric:
    :param batch_size:
    :param step:
    :return: a plugin that prints validation performance after each step
    """
    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            print(clf.val_performance(metric, batch_size))
    return plugin
