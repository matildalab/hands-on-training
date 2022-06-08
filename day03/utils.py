# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
from typing import List

import poptorch
import torch
from torchvision import transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN training in PopTorch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='epochs for training')
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16',
                        help="<input precision>.<model precision>: 16.16, 16.32, 32.32")
    parser.add_argument('--gradient-accumulation', default=1, type=int, help='gradient accumulation')
    parser.add_argument('--device-iterations', default=1, type=int, help='device iteration')
    parser.add_argument('--async-dataloader', action='store_true', help="use async io mode for dataloader")
    parser.add_argument('--eight-bit-io', action='store_true', help="set input io to eight bit")
    parser.add_argument('--replicas', default=1, type=int, help='replication factor for data parallel')

    args = parser.parse_args()

    return args


def set_ipu_options(args):
    opts = poptorch.Options()
    opts.Training.gradientAccumulation(args.gradient_accumulation)
    opts.deviceIterations(args.device_iterations)
    opts.replicationFactor(args.replicas)
    return opts


def get_transform(precision='16.16', eight_bit_io=False):
    preprocssing_steps = [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.PILToTensor()
    ]

    if not eight_bit_io:
        dtype = torch.float16 if precision[:2] == '16' else torch.float32
        preprocssing_steps.extend([CastTo(dtype), Normalize(dtype)])

    return transforms.Compose(preprocssing_steps)


class CastTo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, tensor):
        return tensor.to(self.dtype)


class Normalize(torch.nn.Module):
    def __init__(
            self, dtype: torch.dtype,
            mean: List[float] = [0.485, 0.456, 0.406],
            std: List[float] = [0.229, 0.224, 0.225]
    ):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=dtype)
        std = torch.as_tensor(std, dtype=dtype)
        self.mul = (1.0 / (255.0 * std)).view(3, 1, 1)
        self.sub = (mean / std).view(3, 1, 1)

    def forward(self, tensor):
        # same as ((tensor / 255) - mean) / std
        return tensor.mul(self.mul).sub(self.sub)


class ModelWithNormalization(torch.nn.Module):
    def __init__(
            self, model: torch.nn.Module, dtype: torch.dtype,
            mean: List[float] = [0.485, 0.456, 0.406],
            std: List[float] = [0.229, 0.224, 0.225]
    ):
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.normalize = Normalize(dtype, mean, std)

    def forward(self, img):
        # One-liner in Poplar SDK >= 2.6:
        # return self.model(self.normalize(img.to(self.dtype)))
        if self.dtype == torch.float32:
            img = img.float()
        elif self.dtype == torch.float16:
            img = img.half()
        else:
            raise Exception(f'Normalization dtype must be one of torch.float16 or torch.float32,'
                            f' but {self.dtype} is given')
        return self.model(self.normalize(img))
