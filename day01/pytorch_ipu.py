# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import poptorch
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(972, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))

        return x


class ModelwithLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x, labels=None):
        output = self.model(x)
        if labels is not None:
            loss = self.criterion(output, labels)
            return output, poptorch.identity_loss(loss, reduction='sum')
        return output


if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        "../datasets", transform=transform, download=True, train=True)

    test_dataset = torchvision.datasets.FashionMNIST(
        "../datasets", transform=transform, download=True, train=False)

    opts = poptorch.Options()
    opts.enableExecutableCaching('../cache')

    model = ClassificationModel()
    model.train()

    train_dataloader = poptorch.DataLoader(opts,
                                           train_dataset,
                                           batch_size=16,
                                           shuffle=True)

    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()

    model = ModelwithLoss(model, criterion)
    training_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)

    epochs = 5
    for epoch in range(1, epochs + 1):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description(f'[Epoch {epoch:02d}]')
        for data, labels in bar:
            output, loss = training_model(data, labels)
            bar.set_postfix({"Loss": torch.mean(loss).item()})

    training_model.detachFromDevice()

    model = model.eval()
    inference_model = poptorch.inferenceModel(model, options=opts)

    test_dataloader = poptorch.DataLoader(opts,
                                          test_dataset,
                                          batch_size=32)

    predictions, labels = [], []
    for data, label in test_dataloader:
        predictions += inference_model(data).data.max(dim=1).indices
        labels += label

    inference_model.detachFromDevice()

    print(f"Eval accuracy: {100 * accuracy_score(labels, predictions):.2f}%")
