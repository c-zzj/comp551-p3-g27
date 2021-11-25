"""
This file should be migrated to a jupyter notebook.
"""
from classifier.network.toy_net import *
from classifier.network.alex_net import *
from classifier.plugin import *
from classifier.metric import *
from preprocess import *


DATASET_PATH = Path("../dataset")
SUBMISSION_PATH = Path("submissions")
TRAINED_MODELS_PATH = Path("trained_models")
PROCESSED_DATA_PATH = Path("processed_data")

trained_alex_net_PATH = Path(TRAINED_MODELS_PATH / "alex_net")
trained_toy_net_PATH = Path(TRAINED_MODELS_PATH / "toy_net")
trained_alex_net_on_augmented_PATH = Path(TRAINED_MODELS_PATH / "alex_net_rotation_augmented")
rotation_augmented_data_PATH = Path(PROCESSED_DATA_PATH / "rotation_augmented_dataset.data")


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
    alex.train(30,
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


def run_alex_on_augmented():
    training_set = read_train_labeled(DATASET_PATH)
    process_data(training_set, rotation_augmented_data_PATH,
                 [preprocess_rotate], [{'rotations': [-30, -20, -10, 0, 10, 20, 30]}])
    training_set = torch.load(rotation_augmented_data_PATH)
    alex = AlexNetClassifier(training_set)
    adam = OptimizerProfile(Adam, {
        "lr": 0.0005,
        "betas": (0.9, 0.99),
        "eps": 1e-8
    })
    alex.set_optimizer(adam)
    alex.train(30,
               batch_size=50,
               plugins=[
                   save_model(trained_alex_net_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_train_val_performance(trained_alex_net_PATH, accuracy),
                   plot_train_val_performance(trained_alex_net_PATH, 'Modified AlexNet', accuracy, True, True),
                   elapsed_time(),
                   save_training_message(trained_alex_net_PATH),
               ])


def load_and_test_NN(clf: Callable[..., NNClassifier], epochs: int, submission_number: int, data_folder: Path, model_path: Path):
    clf = clf(read_train_labeled(data_folder))
    clf.load_network(model_path, epochs)
    clf.save_test_result(read_test(data_folder), SUBMISSION_PATH, submission_number)
    print(f"Validation accuracy: {clf.val_performance(accuracy)}")

if __name__ == '__main__':
    #load_and_test_NN(AlexNetClassifier, 25, 3, DATASET_PATH, trained_alex_net_PATH)
    run_alex_on_augmented()
    #toy_demo()