# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
import torch
from torchvision import transforms
import poptorch
from typing import List


normalization_parameters = {"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN training in PopTorch')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='epochs for training')
    parser.add_argument('--replication', default=1, type=int, help='replication factor for data parallel')
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--gradient-accumulation', default=1, type=int, help='gradient accumulation')
    parser.add_argument('--device-iteration', default=1, type=int, help='device iteration')
    parser.add_argument('--async-dataloader', action='store_true', default=False, help="use async io mode for dataloader")
    parser.add_argument('--eight-bit-io', action='store_true', default=False, help="set input io to eight bit")
    
    args = parser.parse_args()
    
    return args


def set_ipu_options(args):
    opts = poptorch.Options()
    opts.Training.gradientAccumulation(args.gradient_accumulation)
    opts.deviceIterations(args.device_iteration)
    opts.replicationFactor(args.replication)

    return opts


def get_preprocessing_pipeline(precision='16.16', eightbit=False):
    """
    Return optimized pipeline, which contains fused transformations.
    """
    pipeline_steps = []
    
    pipeline_steps.append(transforms.RandomHorizontalFlip())
    pipeline_steps.append(transforms.Resize((64, 64)))

    if eightbit:
        pipeline_steps.append(NormalizeToTensor.pil_to_tensor)
    else:
        pipeline_steps.append(NormalizeToTensor(mean=normalization_parameters["mean"], std=normalization_parameters["std"]))

    if eightbit:
        pipeline_steps.append(ToByte())
    elif precision[:3] == '16.':
        pipeline_steps.append(ToHalf())
    else:
        pipeline_steps.append(ToFloat())

    return transforms.Compose(pipeline_steps)


class ToHalf(torch.nn.Module):
    def forward(self, tensor):
        return tensor.half()


class ToFloat(torch.nn.Module):
    def forward(self, tensor):
        return tensor.float()


class ToByte(torch.nn.Module):
    def forward(self, tensor):
        return tensor.byte()


class NormalizeToTensor(torch.nn.Module):
    def __init__(self, mean, std):
        """
        Fuse ToTensor and Normalize operation.
        Expected input is a PIL image and the output is the normalized float tensor.
        """
        # fuse: division by 255 and the normalization
        # Convert division to multiply
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.mul = (1.0/(255.0 * std)).view(-1, 1, 1)
        self.sub = (mean / std).view(-1, 1, 1)
        super().__init__()

    def forward(self, img):
        img = self.pil_to_tensor(img).float()
        img.mul_(self.mul)
        img.sub_(self.sub)
        return img

    @staticmethod
    def pil_to_tensor(pic):
        if isinstance(pic, torch.Tensor):
            return pic
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        return img


class NormalizeInputModel(torch.nn.Module):
    """Wraps the model and convert the input tensor to the given type, and normalise it."""
    def __init__(self, model: torch.nn.Module, mean: List[float], std: List[float], output_cast=None):
        super().__init__()
        self.model = model
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.mul = (1.0/(255.0 * std)).view(-1, 1, 1)
        self.sub = (mean / std).view(-1, 1, 1)
        self.output_cast = output_cast
        if output_cast == "full":
            self.mul, self.sub = self.mul.float(), self.sub.float()
        elif output_cast == "half":
            self.mul, self.sub = self.mul.half(), self.sub.half()

    def forward(self, img):
        if self.output_cast == "half":
            img = img.half()
        elif self.output_cast == "full":
            img = img.float()
        img = img.mul(self.mul)
        img = img.sub(self.sub)
        return self.model(img)