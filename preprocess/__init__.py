from dataset import *
import torchvision.transforms.functional as TF

Preprocess = Callable[[Union[LabeledDataset, UnlabeledDataset], Any], Union[LabeledDataset, UnlabeledDataset]]
PreprocessPipeline = List[Preprocess]


def process_data(dataset: Union[LabeledDataset, UnlabeledDataset],
                 target: Path,
                 pipeline: PreprocessPipeline,
                 params: List[Dict[str, Any]]= []) -> Dataset:
    """

    :param target:
    :param dataset:
    :param pipeline:
    :param params:
    :return:
    """
    if len(pipeline) != len(params) and len(params) != 0:
        raise IndexError("Lengths of the pipeline and corresponding parameters must match")
    for i in range(len(pipeline)):
        if len(params) != 0:
            dataset = pipeline[i](dataset, **params[i])

    if not target.parent.exists():
        target.parent.mkdir(parents=True)
    torch.save(dataset, str(target))
    return dataset


def preprocess_260_labels(dataset: LabeledDataset, batch_size=1000) -> LabeledDataset:
    """

    :param dataset:
    :param rotations:
    :return:
    """
    if type(dataset) == UnlabeledDataset:
        return dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    d = device('cuda:0' if torch.cuda.is_available() else 'cpu')
    result_x = Tensor([])
    result_y = Tensor([])
    for i, data in enumerate(loader, 0):
        x, y = data[0].to(d), data[1].to(d)
        result_x = torch.cat((result_x, x.clone().detach().to('cpu')), dim=0)

        numbers = y[:, :10]
        letters = y[:, 10:]
        num = torch.argmax(numbers, dim=1)
        letter = torch.argmax(letters, dim=1)
        new_y = torch.zeros((len(y), 260), dtype=torch.int, device=d)
        new_y[range(len(y)), letter * 10 + num] = 1
        result_y = torch.cat((result_y, new_y.to('cpu')), dim=0)
    return LabeledDataset(result_x, result_y)


def preprocess_scale_image(dataset: Union[LabeledDataset, UnlabeledDataset], batch_size=1000) -> Union[LabeledDataset, UnlabeledDataset]:
    """

    :param dataset:
    :param scale:
    :param batch_size:
    :return:
    """
    if type(dataset) == UnlabeledDataset:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        d = device('cuda:0' if torch.cuda.is_available() else 'cpu')
        result_x = Tensor([])
        target_size = 56 * 2
        for i, data in enumerate(loader, 0):
            x = data.to(d)
            result_x = torch.cat((result_x, TF.resize(x, [target_size, target_size]).to('cpu')), dim=0)
        return UnlabeledDataset(result_x,)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    d = device('cuda:0' if torch.cuda.is_available() else 'cpu')
    result_x = Tensor([])
    result_y = Tensor([])
    target_size = 56*2
    for i, data in enumerate(loader, 0):
        x, y = data[0].to(d), data[1]
        result_x = torch.cat((result_x, TF.resize(x, [target_size, target_size]).to('cpu')), dim=0)
        result_y = torch.cat((result_y, y.clone().detach()), dim=0)
    return LabeledDataset(result_x, result_y)


def preprocess_rotate(dataset: LabeledDataset, rotations: List[int], batch_size=1000)-> LabeledDataset:
    """

    :param dataset:
    :param rotations:
    :return:
    """
    rotations = list(rotations)
    if 0 not in rotations:
        rotations.append(0)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    d = device('cuda:0' if torch.cuda.is_available() else 'cpu')
    result_x = Tensor([])
    result_y = Tensor([])
    for rotation in rotations:
        for i, data in enumerate(loader, 0):
            x, y = data[0].to(d), data[1]
            result_x = torch.cat((result_x, TF.rotate(x, rotation).to('cpu')), dim=0)
            result_y = torch.cat((result_y, y.clone().detach()), dim=0)
    return LabeledDataset(result_x, result_y)


def preprocess_rotate_ignore_xt(dataset: LabeledDataset, rotations: List[int], batch_size=1000) -> LabeledDataset:
    """

    :param dataset:
    :param rotations:
    :return:
    """
    rotations = list(rotations)
    if 0 not in rotations:
        rotations.append(0)

    letter_x = ord('x') - ord('a') + 10
    letter_t = ord('t') - ord('a') + 10
    result_x = Tensor([])
    result_y = Tensor([])
    no_tx_x = Tensor([])
    no_tx_y = Tensor([])
    # add images containing x or t
    for i in range(len(dataset)):
        x, y = dataset.x[i][None, :], dataset.y[i][None, :]
        if y[0, letter_x] == 1 or y[0, letter_t] == 1:
            x, y = x.clone().detach(), y.clone().detach()
            for _ in rotations:
                result_x = torch.cat((result_x, x), dim=0)
                result_y = torch.cat((result_y, y), dim=0)
        else:
            no_tx_x = torch.cat((no_tx_x, x), dim=0)
            no_tx_y = torch.cat((no_tx_y, y), dim=0)

    # add other images
    d = device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(LabeledDataset(no_tx_x, no_tx_y), batch_size=batch_size, shuffle=False)
    for rotation in rotations:
        for i, data in enumerate(loader, 0):
            x, y = data[0].clone().detach().to(d), data[1].clone().detach()
            result_x = torch.cat((result_x, TF.rotate(x, rotation).to('cpu')), dim=0)
            result_y = torch.cat((result_y, y.clone().detach()), dim=0)
    indices = randperm(len(result_x))
    result_x = result_x[indices]
    result_y = result_y[indices]
    return LabeledDataset(result_x, result_y)

