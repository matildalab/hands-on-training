# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popdist
import poptorch
import torch
import torchvision
import horovod.torch as hvd
from tqdm import tqdm

from utils import (
    parse_arguments, set_ipu_options,
    ModelWithNormalization, get_transform
)


def compute_accuracy(pred, labels):
    _, ind = torch.max(pred, 1)
    return torch.mean(torch.eq(ind, labels).to(pred.dtype)) * 100.0


class ModelwithLoss(torch.nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, x, labels=None):
        output = self.model(x)
        if labels is not None:
            loss = self.criterion(output, labels)
            accuracy = compute_accuracy(output, labels)
            return output, poptorch.identity_loss(loss, reduction='sum'), accuracy
        return output


if __name__ == '__main__':
    if popdist.isPopdistEnvSet():
        hvd.init()
    args = parse_arguments()
    opts = set_ipu_options(args)
    print(args)

    transform = get_transform(args.precision, args.eight_bit_io)
    train_dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=True, download=True, transform=transform)

    mode = poptorch.DataLoaderMode.Async if args.async_dataloader \
        else poptorch.DataLoaderMode.Sync

    train_dataloader = poptorch.DataLoader(opts,
                                           train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           mode=mode)

    model = torchvision.models.resnet101()
    model.layer3[8] = poptorch.BeginBlock(model.layer3[8], ipu_id=1)

    if args.precision[-3:] == '.16':
        optimizer = poptorch.optim.AdamW(model.parameters(), lr=0.001,
                                         loss_scaling=1000,
                                         accum_type=torch.float16,
                                         first_order_momentum_accum_type=torch.float16,
                                         second_order_momentum_accum_type=torch.float32)
    else:
        optimizer = poptorch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    input_dtype = torch.float16 if args.precision[:2] == '16' else torch.float32
    model_dtype = torch.float16 if args.precision[-2:] == '16' else torch.float32
    if args.eight_bit_io:
        model = ModelWithNormalization(model, dtype=input_dtype)
    model = ModelwithLoss(model, criterion).to(model_dtype)

    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        bar = tqdm(train_dataloader, total=len(train_dataloader), disable=popdist.getInstanceIndex() != 0)
        bar.set_description(f'[Epoch {epoch:02d}]')
        loss_values = []
        accuracy_values = []
        for data, labels in bar:
            outputs, losses, accuracies = poptorch_model(data, labels)
            loss = torch.mean(losses)
            accuracy = torch.mean(accuracies)
            if popdist.isPopdistEnvSet():
                loss, accuracy = hvd.grouped_allreduce([loss, accuracy], op=hvd.Average)
            loss_values.append(loss.item())
            accuracy_values.append(accuracy.item())
            bar.set_postfix({'loss': loss.item(), 'acc': accuracy.item()})
        epoch_loss = torch.as_tensor(loss_values).mean().item()
        epoch_accuracy = torch.as_tensor(accuracy_values).mean().item()
        if popdist.isPopdistEnvSet() and popdist.getInstanceIndex() == 0:
            print(f'\n[Epoch {epoch:02d}] loss: {epoch_loss:.3f}, acc: {epoch_accuracy:.1f}', end='')
        elif not popdist.isPopdistEnvSet():
            print(f'[Epoch {epoch:02d}] loss: {epoch_loss:.3f}, acc: {epoch_accuracy:.1f}')
