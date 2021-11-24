"""
This file should be migrated to a jupyter notebook.
"""
from classifier import *
from classifier.network.toy_net import *
from classifier.network.alex_net import *
from classifier.plugin import *
from classifier.metric import *
from dataset import *
from preprocess import *


DATASET_PATH = Path("../dataset")
SUBMISSION_PATH = Path("submissions")
TRAINED_MODELS_PATH = Path("trained_models")

trained_alex_net_PATH = Path(TRAINED_MODELS_PATH / "alex_net")
trained_toy_net_PATH = Path(TRAINED_MODELS_PATH / "toy_net")

def toy_demo():
    training_set = read_train_labeled(DATASET_PATH)
    print(training_set.y.size())
    toy_clf = ToyClassifier(training_set)
    toy_clf.train(epochs=5,
                  batch_size=100,
                  plugins=[
                      save_model(trained_toy_net_PATH),
                      calc_train_val_performance(accuracy),
                      print_train_val_performance(accuracy),
                      log_train_val_performance(accuracy),
                      save_training_message(trained_toy_net_PATH),
                      save_train_val_performance(trained_toy_net_PATH, accuracy),
                      plot_train_val_performance(trained_toy_net_PATH, 'Modified AlexNet', accuracy, True, True)
                  ]
                  )
    toy_clf = ToyClassifier(read_train_labeled(DATASET_PATH))
    toy_clf.load_network(trained_toy_net_PATH, 2)
    print(toy_clf.val_performance(accuracy))

def run_alex():
    training_set = read_train_labeled(DATASET_PATH)
    alex = AlexNetClassifier(training_set)
    adam = OptimizerProfile(Adam, {
            "lr": 0.0005,
            "betas": (0.9, 0.99),
            "eps": 1e-8
        })
    alex.set_optimizer(adam)
    alex.train(5,
               batch_size=50,
               plugins=[
                   save_model(trained_alex_net_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_training_message(trained_alex_net_PATH),
                   save_train_val_performance(trained_alex_net_PATH, accuracy),
                   plot_train_val_performance(trained_alex_net_PATH, 'Modified AlexNet', accuracy, True, True)
               ])


def load_and_test_NN(clf: Callable[..., NNClassifier], epochs: int, submission_number: int, data_folder: Path, model_path: Path):
    clf = clf(read_train_labeled(data_folder))
    clf.load_network(model_path, epochs)
    clf.save_test_result(read_test(data_folder), SUBMISSION_PATH, submission_number)
    print(f"Validation accuracy: {clf.val_performance(accuracy)}")

if __name__ == '__main__':
    #load_and_test_NN(AlexNetClassifier, 25, 3, DATASET_PATH, trained_alex_net_PATH)
    run_alex()
    #toy_demo()