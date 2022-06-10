# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse

import popdist
import popdist.poptorch
import poptorch
import torch
from torchvision import transforms

IMAGENET_STATISTICS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


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
    if not popdist.isPopdistEnvSet():
        parser.add_argument('--replicas', default=1, type=int, help='replication factor for data parallel')
    return parser.parse_args()


def set_ipu_options(args):
    if popdist.isPopdistEnvSet():
        opts = popdist.poptorch.Options()
        opts.randomSeed(2022)
        opts.showCompilationProgressBar(popdist.getInstanceIndex() == 0)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(args.replicas)
    opts.enableExecutableCaching('../../cache')
    opts.Training.gradientAccumulation(args.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Sum)
    opts.deviceIterations(args.device_iterations)
    return opts


def get_transform(precision='16.16', eight_bit_io=False):
    preprocssing_steps = [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.PILToTensor()
    ]

    if not eight_bit_io:
        dtype = torch.float16 if precision[:2] == '16' else torch.float32
        preprocssing_steps.extend([
            transforms.ConvertImageDtype(dtype),
            transforms.Normalize(**IMAGENET_STATISTICS)
        ])

    return transforms.Compose(preprocssing_steps)


class ModelWithNormalization(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype
        self.normalize = transforms.Normalize(**IMAGENET_STATISTICS)

    def forward(self, img):
        return self.model(self.normalize(img.to(self.dtype)))
