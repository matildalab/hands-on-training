# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        "../datasets", transform=transform, download=True, train=True)

    test_dataset = torchvision.datasets.FashionMNIST(
        "../datasets", transform=transform, download=True, train=False)

    model = ClassificationModel()
    model.train()
    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=16,
                                             shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()

    epochs = 5
    for epoch in range(1, epochs + 1):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description(f'[Epoch {epoch:02d}]')
        for data, labels in bar:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            bar.set_postfix({"Loss": torch.mean(loss).item()})

    model = model.eval()

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    predictions, labels = [], []
    for data, label in test_dataloader:
        data = data.to(device)
        predictions += model(data).data.max(dim=1).indices.cpu()
        labels += label

    print(f"Eval accuracy: {100 * accuracy_score(labels, predictions):.2f}%")
