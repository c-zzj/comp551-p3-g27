from dataset import *

Preprocess = Callable[[Dataset, Any], Dataset]
PreprocessPipeline = List[Preprocess]


def process_data(dataset: Union[LabeledDataset, UnlabeledDataset],
                 pipeline: PreprocessPipeline,
                 params: List[Dict[str: Any]]= [{}]) -> Dataset:
    """

    :param dataset:
    :param pipeline:
    :param params:
    :return:
    """
    if len(pipeline) != len(params):
        raise IndexError("Lengths of the pipeline and corresponding parameters must match")
    for i in range(len(pipeline)):
        dataset = pipeline[i](dataset, **params[i])
    return dataset

