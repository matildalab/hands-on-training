import horovod.torch as hvd
import popdist
import torch

if __name__ == '__main__':
    if popdist.isPopdistEnvSet():
        hvd.init()

        x = torch.tensor([1, 0], dtype=torch.float32)
        y = torch.tensor([0, 1], dtype=torch.float16)

        instanceIndex = popdist.getInstanceIndex()
        x *= instanceIndex
        y *= instanceIndex

        print(f'Before grouped allreduce: x = {x}, y = {y}')
        x, y = hvd.grouped_allreduce([x, y], op=hvd.Average)
        print(f'Before grouped allreduce: x = {x}, y = {y}')
    else:
        print('Please run this example with poprun')
