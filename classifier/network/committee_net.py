from classifier import *

class CommitteModel(Module):
    def __init__(self, networks: List[Module]):
        super(CommitteModel, self).__init__()
        self.networks = networks
        for i in range(len(networks)):
            setattr(self, f'weight{i}', nn.Conv2d(1, 1, (1,1)))

    def forward(self, x):
        results = [Function.label_to_36(network(x), x.get_device()) for network in self.networks]
        sum = getattr(self,f'weight0')(results[0])
        for i in range(1, len(results)):
            sum += getattr(self,'weighti')(results[i])
        return sum


class CommitteeClassifier(NNClassifier):
    def __init__(self,
                 nn_clfs: List[NNClassifier],
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 ):
        networks = [clf.network for clf in nn_clfs]
        super(CommitteeClassifier, self).__init__(CommitteModel(networks), training_l, validation, training_ul)
        self.networks = networks

    def predict(self, x: Tensor):
        return self._original_36_argmax(x)


class CommitteeClassifier1(NNClassifier):
    def __init__(self,
                 nn_clfs: List[NNClassifier],
                 training_l: LabeledDataset,
                 validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 ):
        networks = [clf.network for clf in nn_clfs]
        super(CommitteeClassifier1, self).__init__(CommitteModel(networks), training_l, validation, training_ul)
        self.networks = networks

    def predict(self, x: Tensor):
        results = [network._260_argmax_to_36(network.predict(x)) for network in self.networks]
        vote = torch.zeros((len(x), 36), dtype=torch.int, device=self.device)
        for pred in results:
            vote += pred
        return self._original_36_argmax(x)

