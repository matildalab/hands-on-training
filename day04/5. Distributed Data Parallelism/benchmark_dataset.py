# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import poptorch
import torchvision
from tqdm import tqdm

from utils import (
    parse_arguments, set_ipu_options,
    get_transform
)

if __name__ == '__main__':
    args = parse_arguments()
    opts = set_ipu_options(args)
    print(args)

    transform = get_transform(args.precision, args.eight_bit_io)
    train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)

    mode = poptorch.DataLoaderMode.Async if args.async_dataloader \
        else poptorch.DataLoaderMode.Sync

    train_dataloader = poptorch.DataLoader(opts,
                                           train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           mode=mode)

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description(f'[Epoch {epoch:02d}]')
        for data, labels in bar:
            continue
