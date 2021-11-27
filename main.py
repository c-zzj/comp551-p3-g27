"""
This file should be migrated to a jupyter notebook.
"""
from classifier.network.toy_net import *
from classifier.network.alex_net import *
from classifier.network.committee_net import *
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
                      #calc_train_val_performance(accuracy),
                      print_train_val_performance(accuracy),
                      #log_train_val_performance(accuracy),
                      #save_training_message(trained_toy_net_PATH),
                      #save_train_val_performance(trained_toy_net_PATH, accuracy),
                      plot_train_val_performance(trained_toy_net_PATH, 'Modified AlexNet', accuracy, True, True)
                  ]
                  )
    toy_clf = ToyClassifier(train, validation)
    toy_clf.load_network(trained_toy_net_PATH, 2)
    print(toy_clf.val_performance(accuracy))


rotation_augmented_data_PATH = Path(PROCESSED_DATA_PATH / "rotation-augmented-dataset-last90percent.data")
rotation_augmented_ignorext_data_PATH = Path(PROCESSED_DATA_PATH / "rotation-augmented-ignorext-dataset-last90percent.data")

TRAINED_ALEX_NET_PATH = Path(TRAINED_MODELS_PATH / "alex-nets")
ALEX_NET_WRONG_PREDICTION_PATH = Path(WRONG_PRED_ENTRIES_PATH / "alex-nets")
def run_alex(n_way: int, depth: Tuple[int, int, int], scaled: bool, relabeled: bool, rotation_augment: False):
    print(f"n_way: {n_way}")
    print(f"depth: {depth}")
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    trained_alex_net_PATH = TRAINED_ALEX_NET_PATH
    alex_net_rotation_augmented_WRONG_PRED_PATH = ALEX_NET_WRONG_PREDICTION_PATH
    if rotation_augment:
        trained_alex_net_PATH = Path(Path(str(trained_alex_net_PATH) + "-rotation-augmented"))
        alex_net_rotation_augmented_WRONG_PRED_PATH = Path(Path(str(alex_net_rotation_augmented_WRONG_PRED_PATH) + "-rotation-augmented"))

        validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
        train = torch.load(rotation_augmented_data_PATH)
    else:
        validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    if scaled:
        trained_alex_net_PATH = Path(Path(str(trained_alex_net_PATH) + "-scaled"))
        alex_net_rotation_augmented_WRONG_PRED_PATH = Path(
            Path(str(alex_net_rotation_augmented_WRONG_PRED_PATH) + "-scaled"))
        train = preprocess_scale_image(train)
        validation = preprocess_scale_image(validation)
    if relabeled:
        trained_alex_net_PATH = Path(Path(Path(str(trained_alex_net_PATH) + "-relabeled")))
        alex_net_rotation_augmented_WRONG_PRED_PATH = Path(
            Path(str(alex_net_rotation_augmented_WRONG_PRED_PATH) + "-relabeled"))
        train = preprocess_260_labels(train)
        validation = preprocess_260_labels(validation)

    trained_alex_net_PATH = Path(trained_alex_net_PATH / f"{n_way}way-depth{depth}")
    alex_net_rotation_augmented_WRONG_PRED_PATH = Path(alex_net_rotation_augmented_WRONG_PRED_PATH / f"{n_way}way-depth{depth}")
    alex = AlexNetClassifier(train, validation, n_way=n_way, depth=depth, scaled=scaled, relabeled=relabeled)
    alex.set_optimizer(ADAM_PROFILE)
    alex.train(30,
               batch_size=100,
               plugins=[
                   save_model(trained_alex_net_PATH),
                   calc_train_val_performance(accuracy),
                   print_train_val_performance(accuracy),
                   log_train_val_performance(accuracy),
                   save_training_message(trained_alex_net_PATH),
                   plot_train_val_performance(trained_alex_net_PATH, 'Modified AlexNet', accuracy, show=False, save=True),
                   elapsed_time(),
                   save_train_val_performance(trained_alex_net_PATH, accuracy),
               ])
    alex.extract_wrong_pred_entries(alex_net_rotation_augmented_WRONG_PRED_PATH)

COMMITTEE_WRONG_PREDICTION_PATH = Path(WRONG_PRED_ENTRIES_PATH / "committee-")
def run_committee_clf(members: Dict[Path, Tuple[int, Dict[str, Any]]], committee_name: str):
    """

    :param members:
    :param committee_name:
    :return:
    """
    committee_name = "Committee Classifier - " + committee_name + "\n"
    committee_wrong_pred_path = Path(str(COMMITTEE_WRONG_PREDICTION_PATH)+committee_name)
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    models = []
    for path in members:
        epoch, params = members[path]
        alex = AlexNetClassifier(train, validation, **params)
        alex.load_network(path, epoch)
        models.append(alex)
    committee = CommitteeClassifier1(models, train, validation)
    with open(str(TRAINED_MODELS_PATH / 'committees.txt'), 'a+') as f:
        f.write(committee_name)
        f.write(f"Val acc: {committee.val_performance(accuracy)}\n\n")
    committee.extract_wrong_pred_entries(committee_wrong_pred_path)


def run_committee_val():
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    train_scaled = preprocess_scale_image(train)
    train_scaled_relabeled = preprocess_260_labels(train_scaled)
    train_relabeled = preprocess_260_labels(train)

    validation_scaled = preprocess_scale_image(validation)
    validation_scaled_relabeled = preprocess_260_labels(validation_scaled)
    validation_relabeled = preprocess_260_labels(validation)

    p1 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': False, 'relabeled': False}
    p2 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': True, 'relabeled': False}
    p3 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': False, 'relabeled': True}
    p4 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': True, 'relabeled': True}

    original = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented' / '1way-depth(3, 4, 4)')
    scaled = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented-scaled' / '1way-depth(3, 4, 4)')
    relabeled = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented-relabeled' / '1way-depth(3, 4, 4)')
    scaled_relabeled = Path(
        TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented-scaled-relabeled' / '1way-depth(3, 4, 4)')

    alex1 = AlexNetClassifier(train, validation, **p1)
    alex2 = AlexNetClassifier(train_scaled, validation_scaled, **p2)
    alex3 = AlexNetClassifier(train_relabeled, validation_relabeled, **p3)
    alex4 = AlexNetClassifier(train_scaled_relabeled, validation_scaled_relabeled, **p4)
    alex1.load_network(original, 28)
    alex2.load_network(scaled, 30)
    alex3.load_network(relabeled, 7)
    alex4.load_network(scaled_relabeled, 28)
    pred1 = Tensor([]).to(alex1.device)
    pred2 = Tensor([]).to(alex2.device)
    pred3 = Tensor([]).to(alex3.device)
    pred4 = Tensor([]).to(alex4.device)
    true = Tensor([]).to(alex1.device)
    loader = DataLoader(validation, batch_size=500, shuffle=False)
    for i, data in enumerate(loader, 0):
        x = data[0].to(alex1.device)
        y = data[1].to(alex1.device)
        pred1 = torch.cat((pred1, Function.label_to_36_argmax(alex1.predict(x), device=alex1.device)), dim=0)
        pred3 = torch.cat((pred3, Function.label_to_36_argmax(alex3.predict(x), device=alex3.device)), dim=0)
        true = torch.cat((true, y), dim=0)

    loader = DataLoader(validation_scaled, batch_size=500, shuffle=False)
    for i, data in enumerate(loader, 0):
        x = data[0].to(alex1.device)
        pred2 = torch.cat((pred2, Function.label_to_36_argmax(alex2.predict(x), device=alex1.device)), dim=0)
        pred4 = torch.cat((pred4, Function.label_to_36_argmax(alex4.predict(x), device=alex3.device)), dim=0)
    pred = pred1 * 1.2 + pred2 * 1.1 + pred3 + pred4
    data_size = len(pred)
    numbers = pred[:, :10]
    letters = pred[:, 10:]
    num_pred = torch.argmax(numbers, dim=1)
    letter_pred = torch.argmax(letters, dim=1) + 10

    output = torch.zeros((data_size, 36), dtype=torch.int, device=alex1.device)
    output[range(data_size), num_pred] = 1
    output[range(data_size), letter_pred] = 1
    print(accuracy(output, true))


def run_committee():
    """
    shit code
    don't do anything to this
    :return:
    """
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    train_scaled = preprocess_scale_image(train)
    train_scaled_relabeled = preprocess_260_labels(train_scaled)
    train_relabeled = preprocess_260_labels(train)

    validation = read_test(DATASET_PATH)
    validation_scaled = preprocess_scale_image(validation)
    validation_scaled_relabeled = preprocess_260_labels(validation_scaled)
    validation_relabeled = preprocess_260_labels(validation)


    p1 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': False, 'relabeled': False}
    p2 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': True, 'relabeled': False}
    p3 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': False, 'relabeled': True}
    p4 = {'n_way': 1, 'depth': (3, 4, 4), 'scaled': True, 'relabeled': True}

    original = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented' / '1way-depth(3, 4, 4)')
    scaled = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented-scaled' / '1way-depth(3, 4, 4)')
    relabeled = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented-relabeled' / '1way-depth(3, 4, 4)')
    scaled_relabeled = Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented-scaled-relabeled' / '1way-depth(3, 4, 4)')

    alex1 = AlexNetClassifier(train, validation, **p1)
    alex2 = AlexNetClassifier(train_scaled, validation_scaled, **p2)
    alex3 = AlexNetClassifier(train_relabeled, validation_relabeled, **p3)
    alex4 = AlexNetClassifier(train_scaled_relabeled, validation_scaled_relabeled, **p4)
    alex1.load_network(original, 28)
    alex2.load_network(scaled, 30)
    alex3.load_network(relabeled, 7)
    alex4.load_network(scaled_relabeled, 28)
    pred1 = Tensor([]).to(alex1.device)
    pred2 = Tensor([]).to(alex2.device)
    pred3 = Tensor([]).to(alex3.device)
    pred4 = Tensor([]).to(alex4.device)
    true = Tensor([]).to(alex1.device)
    loader = DataLoader(validation, batch_size=500, shuffle=False)
    for i, data in enumerate(loader, 0):
        x = data.to(alex1.device)
        #y = data[1].to(alex1.device)
        pred1 = torch.cat((pred1, Function.label_to_36_argmax(alex1.predict(x), device=alex1.device)), dim=0)
        pred3 = torch.cat((pred3, Function.label_to_36_argmax(alex3.predict(x), device=alex3.device)), dim=0)
        #true = torch.cat((true, y), dim=0)

    loader = DataLoader(validation_scaled, batch_size=500, shuffle=False)
    for i, data in enumerate(loader, 0):
        x = data.to(alex1.device)
        pred2 = torch.cat((pred2, Function.label_to_36_argmax(alex2.predict(x), device=alex1.device)), dim=0)
        pred4 = torch.cat((pred4, Function.label_to_36_argmax(alex4.predict(x), device=alex3.device)), dim=0)
    pred = pred1 * 1.2 + pred2 + pred3 + pred4 * 1.1
    data_size = len(pred)
    numbers = pred[:, :10]
    letters = pred[:, 10:]
    num_pred = torch.argmax(numbers, dim=1)
    letter_pred = torch.argmax(letters, dim=1) + 10

    output = torch.zeros((data_size, 36), dtype=torch.int, device=alex1.device)
    output[range(data_size), num_pred] = 1
    output[range(data_size), letter_pred] = 1
    #print(accuracy(output, true))
    # type cast to concatenated string
    pred = output
    pred = pred.detach().to('cpu').numpy().astype(int).astype(bytearray)
    result = []
    for i, row in enumerate(pred):
        result.append([i, ''.join(row.astype(str))])
    result = pd.DataFrame(result)
    result.columns = ["# ID", "Category"]

    # save predictions
    if not SUBMISSION_PATH.exists():
        SUBMISSION_PATH.mkdir(parents=True)
    result.to_csv(SUBMISSION_PATH / (str(12) + '.csv'), index=False)


def load_and_test_NN(clf: Callable[..., NNClassifier],
                     params: Dict[str, Any],
                     epochs: int,
                     submission_number: int,
                     data_folder: Path,
                     model_path: Path):
    """

    :param clf:
    :param params:
    :param epochs:
    :param submission_number:
    :param data_folder:
    :param model_path:
    :return:
    """
    training_set = read_train_labeled(DATASET_PATH)
    validation_proportion = 0.1
    validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    clf = clf(train, validation, **params)
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
    # load_and_test_NN(AlexNetPlusClassifier, 25, 7, DATASET_PATH, trained_alex_net_rotation_augmented_PATH)
    # load_and_test_NN(AlexNetOneWayClassifier, 21, -1, DATASET_PATH, trained_alex_net_oneway_PATH)
    #run_alex_plus_rotation_augmented_ignorext()
    # the following are to be tried

    #run_alex(1, (1,1,2)) #alex_net_oneway
    #run_alex(2, (1,1,2)) #alex_net original
    #run_alex(3, (1,1,2))
    #run_alex(4, (1,1,2))
    c1_name, c1 = '1', {
        # path: (epoch, {'n_way': 1, 'depth': (3,4,4), 'scaled': , 'relabeled': ,}
    }
    # depth exploration
    # run_alex(2, (1,1,3), scaled=False, relabeled=False, rotation_augment=False)
    # run_alex(2, (1,2,2), scaled=False, relabeled=False, rotation_augment=False)
    # run_alex(2, (1,2,3), scaled=False, relabeled=False, rotation_augment=False)
    # run_alex(2, (2,2,4), scaled=False, relabeled=False, rotation_augment=False) #alex_net_plus (best so far)
    # run_alex(2, (2,3,3), scaled=False, relabeled=False, rotation_augment=False)
    # run_alex(2, (2,3,4), scaled=False, relabeled=False, rotation_augment=False)
    # run_alex(2, (3,3,5), scaled=False, relabeled=False, rotation_augment=False)
    #run_alex(1, (3, 4, 4), scaled=False, relabeled=False, rotation_augment=False)
    # run_alex(1, (3, 4, 4), scaled=False, relabeled=False, rotation_augment=True)
    # run_alex(1, (3, 4, 4), scaled=True, relabeled=False, rotation_augment=True)
    # run_alex(1, (3, 4, 4), scaled=False, relabeled=True, rotation_augment=True)
    # run_alex(1, (3, 4, 4), scaled=True, relabeled=True, rotation_augment=True)
    # run_alex(2, (3,4,5), scaled=False, relabeled=False, rotation_augment=False)

    # load_and_test_NN(AlexNetClassifier, {'n_way':1, 'depth':(3,4,4), 'scaled':False,'relabeled':False},
    #                  epochs=28,
    #                  submission_number=9,
    #                  data_folder=DATASET_PATH,
    #                  model_path=Path(TRAINED_MODELS_PATH / 'alex-nets-rotation-augmented' / '1way-depth(3, 4, 4)'))

    #run_committee_val()
    run_committee()

    # training_set = read_train_labeled(DATASET_PATH)
    # validation_proportion = 0.1
    # validation, train = partition(training_set, [int(len(training_set) * validation_proportion)], False)
    # train = preprocess_scale_image(train)
    # #train = preprocess_260_labels(train)
    # validation = preprocess_scale_image(validation)
    # #validation = preprocess_260_labels(validation)
    # clf = AlexNetClassifier(train, validation, n_way=1, depth=(3,4,4), scaled=True, relabeled=False,)
    # path = Path(TRAINED_MODELS_PATH / 'alex-nets-scaled-relabeled' / '1way-depth(3, 4, 4)')
    # clf.load_network(path, 26)
    # path = Path(WRONG_PRED_ENTRIES_PATH / 'alex-nets-scaled-relabeled' / '1way-depth(3, 4, 4)')