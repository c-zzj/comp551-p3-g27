from classifier import *


class AlexNet(Module):
    def get_conv1(self, depth: Tuple[int, int, int]):
        d1, d2, d3 = depth
        layers = []
        if self.scaled_by_2:
            layers += [nn.Conv2d(1, 16, (5, 5), padding='same')]
            layers += [nn.MaxPool2d(2)]
            layers += [nn.LeakyReLU()]
            layers += [nn.BatchNorm2d(16)]
        else:
            layers += [nn.Conv2d(1, 16, (3, 3), padding='same')]
        if d1 == 1:
            layers += [nn.MaxPool2d(2)]
        layers += [nn.LeakyReLU()]
        layers += [nn.BatchNorm2d(16)]
        for i in range(d1-1):
            layers += [nn.Conv2d(16, 16, (3, 3), padding='same')]
            if i == d1-2:
                layers += [nn.MaxPool2d(2)]
            layers += [nn.LeakyReLU()]
            layers += [nn.BatchNorm2d(16)]

        layers += [nn.Conv2d(16, 64, (3, 3), padding='same')]
        if d2 == 1:
            layers += [nn.MaxPool2d(2)]
        layers += [nn.LeakyReLU()]
        layers += [nn.BatchNorm2d(64)]
        for i in range(d2 - 1):
            layers += [nn.Conv2d(64, 64, (3, 3), padding='same')]
            if i == d2 - 2:
                layers += [nn.MaxPool2d(2)]
            layers += [nn.LeakyReLU()]
            layers += [nn.BatchNorm2d(64)]
        return nn.Sequential(*layers)

    def get_conv2(self, n_way: int, depth: Tuple[int, int, int]):
        d1, d2, d3 = depth
        layers = []
        layers += [nn.Conv2d(64 * n_way, 256, (3, 3), padding='same')]
        layers += [nn.LeakyReLU()]
        layers += [nn.BatchNorm2d(256)]
        for i in range(d3 - 1):
            layers += [nn.Conv2d(256, 256, (3, 3), padding='same')]
            layers += [nn.LeakyReLU()]
            layers += [nn.BatchNorm2d(256)]
        layers += [nn.Conv2d(256, 64, (3, 3), padding='same'),
                   nn.MaxPool2d(2),
                   nn.LeakyReLU(),
                   nn.BatchNorm2d(64), ]
        return nn.Sequential(*layers)

    def get_dense1(self, n_way: int):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * n_way * 7 * 7, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
        )

    def get_dense2(self, n_way: int):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_way * 1024, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
        )

    def __init__(self,
                 n_way: int,
                 depth: Tuple[int, int, int],
                 scaled_by_2: bool,
                 label260: bool):
        super(AlexNet, self).__init__()
        self.n_way = n_way
        self.depth = depth
        self.scaled_by_2 = scaled_by_2
        self.label260 = label260
        conv1 = [self.get_conv1(depth) for _ in range(n_way)]
        conv2 = [self.get_conv2(n_way, depth) for _ in range(n_way)]
        dense1 = [self.get_dense1(n_way) for _ in range(n_way)]
        dense2 = [self.get_dense2(n_way) for _ in range(n_way)]
        self.last_layer = nn.Linear(n_way * 1024, 260) if label260 else nn.Linear(n_way * 1024, 36)
        for i in range(n_way):
            setattr(self, f'conv1_way{i}', conv1[i])
            setattr(self, f'conv2_way{i}', conv2[i])
            setattr(self, f'dense1_way{i}', dense1[i])
            setattr(self, f'dense2_way{i}', dense2[i])

    def forward(self, x):
        output = []
        for i in range(self.n_way):
            conv1 = getattr(self, f'conv1_way{i}')
            output.append(conv1(x))
        x = torch.cat(output, dim=1) # concatenate in the channel dimension
        output = []
        for i in range(self.n_way):
            conv2 = getattr(self, f'conv2_way{i}')
            output.append(conv2(x))
        x = torch.cat(output, dim=1)
        x = Function.flatten(x)
        output = []
        for i in range(self.n_way):
            dense1 = getattr(self, f'dense1_way{i}')
            output.append(dense1(x))
        x = torch.cat(output, dim=1)
        output = []
        for i in range(self.n_way):
            dense2 = getattr(self, f'dense2_way{i}')
            output.append(dense2(x))
        x = torch.cat(output, dim=1)
        return self.last_layer(x)


class AlexNetClassifier(NNClassifier):
    def __init__(self, training_l: LabeledDataset, validation: LabeledDataset,
                 training_ul: Optional[UnlabeledDataset] = None,
                 n_way: int = 1,
                 depth: Tuple[int, int, int] = (1, 1, 2),
                 scaled: bool = False,
                 relabeled: bool = False):
        """

        :param training_l:
        :param validation:
        :param training_ul:
        :param n_way: n ways of convolution layers
        :param depth: depth in each scale space. The length of the list must be 3
        """
        super(AlexNetClassifier, self).__init__(AlexNet(n_way, depth, scaled, relabeled), training_l, validation, training_ul)
        self.optim = SGD(self.network.parameters(), lr=1e-3, momentum=0.99)
        self.loss = CrossEntropyLoss()

    def predict(self, x: Tensor):
        if self.network.label260:
            return self._260_argmax(x)
        return self._original_36_argmax(x)

