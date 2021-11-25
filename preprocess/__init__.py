from dataset import *

Preprocess = Callable[[Union[LabeledDataset, UnlabeledDataset], Any], Union[LabeledDataset, UnlabeledDataset]]
PreprocessPipeline = List[Preprocess]


def process_data(dataset: Union[LabeledDataset, UnlabeledDataset],
                 pipeline: PreprocessPipeline,
                 params: List[Dict[str, Any]]= []) -> Dataset:
    """

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
    return dataset

