import argparse
import poptorch


def parse_arguments():
    parser = argparse.ArgumentParser(description='CNN training in PopTorch')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='epochs for training')
    parser.add_argument('--pipeline-splits', type=str, nargs='+', default=[], help="List of the splitting layers")
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