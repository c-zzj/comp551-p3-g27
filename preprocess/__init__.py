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
            result_y = torch.cat((result_y, torch.tensor(y)), dim=0)
    return LabeledDataset(result_x, result_y)