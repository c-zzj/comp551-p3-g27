"""
This file should be migrated to a jupyter notebook.
"""
from classifier import *
from classifier.network.toy_net import *
from classifier.network.alex_net import *
from classifier.plugin import *
from classifier.metric import *
from dataset import *
from preprocess import  *
import os.path

DATASET_PATH = Path("../dataset")
SUBMISSION_PATH = Path("submissions")

trained_alex_net_PATH = Path("trained_models/alex_net")
trained_toy_net_PATH = Path("trained_models/toy_net")

def demo():
    training_set = read_train_labeled(DATASET_PATH)
    print(training_set.y.size())
    toy_clf = ToyClassifier(training_set)
    toy_clf.train(epochs=10,
                  batch_size=100,
                  plugins=[
                      #save_model(trained_toy_net),
                      print_training_performance(accuracy),
                      print_validation_performance(accuracy)
                  ]
                  )
    toy_clf = ToyClassifier(read_train_labeled(DATASET_PATH))
    toy_clf.load_network(trained_toy_net_PATH, 10)
    print(toy_clf.val_performance(accuracy))

def alex():
    training_set = read_train_labeled(DATASET_PATH)
    alex = AlexNetClassifier(training_set)
    adam = OptimizerProfile(Adam, {
            "lr": 0.0005,
            "betas": (0.9, 0.99),
            "eps": 1e-8
        })
    alex.set_optimizer(adam)
    alex.train(50,
               batch_size=50,
               plugins=[
                   save_model(trained_alex_net_PATH),
                   print_training_performance(accuracy),
                   print_validation_performance(accuracy)
               ])

if __name__ == '__main__':
    alex = AlexNetClassifier(read_train_labeled(DATASET_PATH))
    alex.load_network(trained_alex_net_PATH, 6)
    test_set = read_test(DATASET_PATH)
    alex.save_test_result(test_set, SUBMISSION_PATH, 2)