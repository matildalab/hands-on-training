# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import poptorch
import utils
from utils import NormalizeInputModel


class ModelwithLoss(nn.Module):
    def __init__(self, model, criterion, precision):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.precision = precision
    
    def forward(self, x, labels=None):
        if self.precision[:3] == '16.':
            x = x.half()
        else:
            x = x.float()
        output = self.model(x)
        if labels is not None:
            loss = self.criterion(output, labels)
            return output, poptorch.identity_loss(loss, reduction='sum')
        return output


if __name__ == '__main__':
    args = utils.parse_arguments()
    opts = utils.set_ipu_options(args)

    transform = utils.get_preprocessing_pipeline(args.precision, args.eight_bit_io)
    
    train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
    
    if args.async_dataloader:
        mode = poptorch.DataLoaderMode.Async
    else:
        mode = poptorch.DataLoaderMode.Sync
    
    train_dataloader = poptorch.DataLoader(opts,
                                        train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        mode=mode)

    model = torchvision.models.resnet101()
    model.layer3[8] = poptorch.BeginBlock(model.layer3[8], ipu_id=1)

    if args.precision[-3:] == '.16':
        optimizer = poptorch.optim.AdamW(model.parameters(), lr=0.001, 
                                        loss_scaling = 1000,
                                        accum_type = torch.float16,
                                        first_order_momentum_accum_type = torch.float16,
                                        second_order_momentum_accum_type = torch.float32)
    else:
        optimizer = poptorch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    if args.eight_bit_io:
        """
        Conduct normalization on IPU when eight-bit-io is enabled.
        """
        cast = "half" if args.precision[:3] == "16." else "full"
        model = NormalizeInputModel(
            model,
            utils.normalization_parameters["mean"],
            utils.normalization_parameters["std"],
            output_cast=cast
        )
    model = ModelwithLoss(model, criterion, args.precision)
    if args.precision[-3:] == '.16':
        model = model.half()  
    poptorch_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description(f'[Epoch {epoch:02d}]')
        for data, labels in bar:
            output, loss = poptorch_model(data, labels)
            bar.set_postfix({"Loss": torch.mean(loss).item()})

