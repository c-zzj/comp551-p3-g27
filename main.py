"""
This file should be migrated to a jupyter notebook.
"""
from classifier.network.toy_net import *
from classifier.network.alex_net_plus import *
from classifier.network.alex_net import *
from classifier.network.alex_net_oneway import *
from classifier.plugin import *
from classifier.metric import *
from preprocess import *


DATASET_PATH = Path("../dataset")
SUBMISSION_PATH = Path("submissions")
TRAINED_MODELS_PATH = Path("trained-models")
PROCESSED_DATA_PATH = Path("processed-data")
WRONG_PRED_ENTRIES_PATH = Path("wrong-pred-entries")

ADAM_PROFILE = OptimizerProfile(Adam, {
            "lr": 0.0005,
            "betas": (0.9, 0.99),
            "eps": 1e-8
        })

SGD_PROFILE = OptimizerProfile(SGD, {
        'lr': 0.0005,
        'momentum': 0.99
})

trained_toy_net_PATH = Path(TRAINED_MODELS_PATH / "toy-net")
def toy_demo():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], True)
    toy_clf = ToyClassifier(train, validation)
    toy_clf.train(epochs=100,
                  batch_size=100,
                  plugins=[
                      #save_model(trained_toy_net_PATH),
                      calc_train_val_performance(accuracy),
                      print_train_val_performance(accuracy),
                      log_train_val_performance(accuracy),
                      #save_training_message(trained_toy_net_PATH),
                      #save_train_val_performance(trained_toy_net_PATH, accuracy),
                      plot_train_val_performance(trained_toy_net_PATH, 'Modified AlexNet', accuracy, True, True)
                  ]
                  )
    toy_clf = ToyClassifier(train, validation)
    toy_clf.load_network(trained_toy_net_PATH, 2)
    print(toy_clf.val_performance(accuracy))


trained_alex_net_PATH = Path(TRAINED_MODELS_PATH / "alex-net")
def run_alex():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], True)
    alex = AlexNetClassifier(train, validation)
    adam = OptimizerProfile(Adam, {
            "lr": 0.0005,
            "betas": (0.9, 0.99),
            "eps": 1e-8
        })
    alex.set_optimizer(adam)
    alex.train(30,
               batch_size=100,
               plugins=[
                   save_model(trained_alex_net_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_training_message(trained_alex_net_PATH),
                   plot_train_val_performance(trained_alex_net_PATH, 'Modified AlexNet', accuracy, True, True),
                   elapsed_time(),
                   save_train_val_performance(trained_alex_net_PATH, accuracy),
               ])


trained_alex_net_oneway_PATH = Path(TRAINED_MODELS_PATH / "alex-net-oneway")
def run_alex_oneway():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], True)
    alex = AlexNetOneWayClassifier(train, validation)
    adam = OptimizerProfile(Adam, {
            "lr": 0.0005,
            "betas": (0.9, 0.99),
            "eps": 1e-8
        })
    alex.set_optimizer(adam)
    alex.train(30,
               batch_size=100,
               plugins=[
                   save_model(trained_alex_net_oneway_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_training_message(trained_alex_net_oneway_PATH),
                   plot_train_val_performance(trained_alex_net_oneway_PATH, 'Modified AlexNet', accuracy, True, True),
                   elapsed_time(),
                   save_train_val_performance(trained_alex_net_oneway_PATH, accuracy),
               ])


trained_alex_net_plus_PATH = Path(TRAINED_MODELS_PATH / "alex-net-plus")
def run_alex_plus():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], True)
    alex = AlexNetPlusClassifier(train, validation)
    adam = OptimizerProfile(Adam, {
            "lr": 0.0005,
            "betas": (0.9, 0.99),
            "eps": 1e-8
        })
    alex.set_optimizer(adam)
    alex.train(30,
               batch_size=100,
               plugins=[
                   save_model(trained_alex_net_plus_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_training_message(trained_alex_net_plus_PATH),
                   plot_train_val_performance(trained_alex_net_plus_PATH, 'Modified AlexNet', accuracy, True, True),
                   elapsed_time(),
                   save_train_val_performance(trained_alex_net_plus_PATH, accuracy),
               ])


rotation_augmented_data_PATH = Path(PROCESSED_DATA_PATH / "rotation-augmented-dataset-last90percent.data")
trained_alex_net_plus_rotation_augmented_WRONG_PRED_PATH = \
    Path(WRONG_PRED_ENTRIES_PATH / "alex-net-plus-rotation-augmented-val-corrected")
trained_alex_net_plus_rotation_augmented_PATH = \
    Path(TRAINED_MODELS_PATH / "alex-net-plus-rotation-augmented-val-corrected")

def run_alex_plus_rotation_augmented():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    # augment the training set
    process_data(train, rotation_augmented_data_PATH,
                 [preprocess_rotate], [{'rotations': [-30, -20, -10, 0, 10, 20, 30]}])
    train = torch.load(rotation_augmented_data_PATH)
    alex_plus = AlexNetPlusClassifier(train, validation)
    alex_plus.set_optimizer(ADAM_PROFILE)
    alex_plus.train(30,
               batch_size=100,
               plugins=[
                   save_model(trained_alex_net_plus_rotation_augmented_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_train_val_performance(trained_alex_net_plus_rotation_augmented_PATH, accuracy),
                   plot_train_val_performance(trained_alex_net_plus_rotation_augmented_PATH, 'Modified AlexNet', accuracy, True, True),
                   elapsed_time(),
                   save_training_message(trained_alex_net_plus_rotation_augmented_PATH),
               ])
    alex_plus.extract_wrong_pred_entries(trained_alex_net_plus_rotation_augmented_WRONG_PRED_PATH)


rotation_augmented_ignorext_data_PATH = Path(PROCESSED_DATA_PATH / "rotation-augmented-ignorext-dataset-last90percent.data")
trained_alex_net_plus_rotation_augmented_ignorext_PATH = \
    Path(TRAINED_MODELS_PATH / "alex-net-plus-rotation-ignorext-augmented")
trained_alex_net_plus_rotation_augmented_ignorext_WRONG_PRED_PATH = \
    Path(WRONG_PRED_ENTRIES_PATH / "alex-net-plus-rotation-ignorext-augmented")

def run_alex_plus_rotation_augmented_ignorext():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    # augment the training set
    # process_data(train, rotation_augmented_ignorext_data_PATH,
    #              [preprocess_rotate_ignore_xt], [{'rotations': [-30, -20, -10, 0, 10, 20, 30]}])
    train = torch.load(rotation_augmented_ignorext_data_PATH)
    alex_plus = AlexNetPlusClassifier(train, validation)
    alex_plus.set_optimizer(ADAM_PROFILE)
    alex_plus.train(30,
               batch_size=100,
               plugins=[
                   save_model(trained_alex_net_plus_rotation_augmented_ignorext_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_train_val_performance(trained_alex_net_plus_rotation_augmented_ignorext_PATH, accuracy),
                   plot_train_val_performance(trained_alex_net_plus_rotation_augmented_ignorext_PATH, 'Modified AlexNet', accuracy, True, True),
                   elapsed_time(),
                   save_training_message(trained_alex_net_plus_rotation_augmented_ignorext_PATH),
               ])
    alex_plus.extract_wrong_pred_entries(trained_alex_net_plus_rotation_augmented_ignorext_WRONG_PRED_PATH)


def load_and_test_NN(clf: Callable[..., NNClassifier], epochs: int, submission_number: int, data_folder: Path, model_path: Path):
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    clf = clf(train, validation)
    clf.load_network(model_path, epochs)
    clf.save_test_result(read_test(data_folder), SUBMISSION_PATH, submission_number)
    print(f"Validation accuracy: {clf.val_performance(accuracy)}")


if __name__ == '__main__':
    #load_and_test_NN(AlexNetPlusClassifier, 25, 3, DATASET_PATH, trained_alex_net_PATH)
    #toy_demo()
    # training_set = torch.load(rotation_augmented_data_PATH)
    # alex = AlexNetPlusClassifier(training_set, val_proportion=0.025)
    # alex.load_network(trained_alex_net_on_augmented_PATH, 10)
    # print(alex.val_performance(accuracy))
    # alex.extract_wrong_pred_entries(trained_alex_net_rotation_augmented_WRONG_PRED_PATH)
    # load_and_test_NN(AlexNetPlusClassifier, 25, 7, DATASET_PATH, trained_alex_net_plus_rotation_augmented_PATH)
    #run_alex_plus_rotation_augmented_ignorext()
    #load_and_test_NN(AlexNetOneWayClassifier, 21, -1, DATASET_PATH, trained_alex_net_oneway_PATH)
    run_alex_oneway()
