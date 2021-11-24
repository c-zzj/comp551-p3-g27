from classifier import *


def save_model(folder_path: Path, step: int = 1) -> TrainingPlugin:
    """
    :param folder_path: the path of the folder to save the model
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves the model after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            torch.save(clf.network.state_dict(), str(folder_path / f"{epoch}.params"))
    return plugin


def save_training_message(folder_path: Path, step: int = 1, empty_previous: bool = True) -> TrainingPlugin:
    """
    :param empty_previous:
    :param folder_path: the path of the log to be saved
    :param step: step size of epochs to activate the plugin
    :return: a plugin that appends the training message to the log file after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        if empty_previous:
            open(str(folder_path / 'log.txt'), 'w').close()

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            with open(str(folder_path / 'log.txt'), 'a+') as f:
                f.write(clf.training_message)
    return plugin


def calc_train_val_performance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves training and validation performance to the temporary variable
    """

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            clf._tmp['performance'] = (
                clf.train_performance(metric, batch_size),
                clf.val_performance(metric, batch_size)
            )
    return plugin


def log_train_val_performance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """
    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that logs training and validation performance to training message after each step
    """
    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if type(clf._tmp) != dict or 'performance' not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp['performance'][0], clf._tmp['performance'][1]
            s = f"TRAIN: {train}\tVAL: {val}\n"
            clf.training_message += s
    return plugin


def print_train_val_performance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that prints training and validation performance after each step
    """

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if type(clf._tmp) != dict or 'performance' not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp['performance']
            s = f"TRAIN: {train}\tVAL: {val}"
            print(s)
    return plugin


def save_train_val_performance(folder_path: Path, metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param folder_path:
    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves training and validation performance after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if type(clf._tmp) != dict or 'performance' not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp['performance']
            if 'learning_path' not in clf._tmp:
                clf._tmp['learning_path'] = {'epochs': [], 'train': [], 'val': []}
            clf._tmp['learning_path']['epochs'].append(epoch)
            clf._tmp['learning_path']['train'].append(train)
            clf._tmp['learning_path']['val'].append(val)
            torch.save(clf._tmp['learning_path'], str(folder_path / 'performance.pt'))
    return plugin


def load_train_val_performance(folder_path: Path) -> Dict[str, list]:
    """
    load the saved learning path generated by save_train_val_performance
    :param folder_path:
    :return: {'epochs': List[float], 'train': List[float], 'val': List[float]}
    """
    if Path(folder_path / 'performance.pt').exists():
        return torch.load(folder_path / 'performance.pt')
    else:
        raise FileNotFoundError(f"The learning path for {folder_path} has not been saved!")


def plot_train_val_performance(folder_path: Path,
                               title: str,
                               metric: Metric,
                               show: bool = True,
                               save: bool = False,
                               batch_size: int = 300,
                               step: int = 1) -> TrainingPlugin:
    """

    :param save:
    :param title:
    :param folder_path:
    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves and/or shows the learning curve
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    epochs = []
    train_performances = []
    val_performances = []

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if type(clf._tmp) != dict or 'performance' not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp['performance']
            epochs.append(epoch)
            train_performances.append(train)
            val_performances.append(val)
            plt.plot(epochs, train_performances,
                     label="training", alpha=0.5)
            plt.plot(epochs, val_performances,
                     label="validation", alpha=0.5)
            ax = plt.gca()
            ax.set_ylim([0., 1.])
            plt.xlabel('Number of epochs')
            plt.ylabel('Accuracy')
            plt.title(title)
            plt.legend()
            if save:
                plt.savefig(folder_path / f'{epoch} epochs.jpg')
                # delete previous plot
                if Path((folder_path / f'{epoch-step} epochs.jpg')).exists():
                    Path.unlink(Path((folder_path / f'{epoch-step} epochs.jpg')))
            if show:
                plt.show()
    return plugin
