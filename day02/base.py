# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import poptorch


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
                    torchvision.transforms.Pad(4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((64, 64))])

    train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)

    opts = poptorch.Options()
    opts.enableExecutableCaching('../cache')

    model = torchvision.models.resnet50()
    print(model)

    train_dataloader = poptorch.DataLoader(opts,
                                        train_dataset,
                                        batch_size=16,
                                        shuffle=True)

    optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    model = ModelwithLoss(model, criterion)
    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)

    epochs = 5
    for epoch in range(1, epochs + 1):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description(f'[Epoch {epoch:02d}]')
        for data, labels in bar:
            output, loss = poptorch_model(data, labels)
            bar.set_postfix({"Loss": torch.mean(loss).item()})

