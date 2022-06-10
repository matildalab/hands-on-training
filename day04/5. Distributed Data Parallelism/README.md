# Chapter V - Distributed Data Parallelism
## Implementing Distributed Data Parallel Training
Let's modify the training code that we wrote in [day03](../../day03) to implement DDP (Distributed Data Paralellism).
### Modifying [day03/main.py](../../day03/main.py)
- Import `popdist` and `horovod.torch`.
```diff
+ import popdist
import poptorch
import torch
import torchvision
+ import horovod.torch as hvd
from tqdm import tqdm
```
- If `poprun` is invoked, initialize horovod.
```diff
if __name__ == '__main__':
    + if popdist.isPopdistEnvSet():
    +     hvd.init()
    args = parse_arguments()
    opts = set_ipu_options(args)
    print(args)
```
- If `poprun` is invoked, enable the progress bar only for the instance 0.
(Otherwise, the output will look messy.)
```diff
-        bar = tqdm(train_dataloader, total=len(train_dataloader))
+        bar = tqdm(train_dataloader, total=len(train_dataloader), disable=popdist.getInstanceIndex() != 0)
```
- When `poprun` is invoked, perform allreduce operator with horovod after each training iteration to synchronize different loss values over multiple instances.
```diff
for data, labels in bar:
    outputs, losses, accuracies = poptorch_model(data, labels)
    loss = torch.mean(losses)
    accuracy = torch.mean(accuracies)
+    if popdist.isPopdistEnvSet():
+        loss, accuracy = hvd.grouped_allreduce([loss, accuracy], op=hvd.Average)
    loss_values.append(loss.item())
    accuracy_values.append(accuracy.item())
    bar.set_postfix({'loss': loss.item(), 'acc': accuracy.item()})
```
- Modify the message format for better readability when `poprun` is invoked.
```diff
+if popdist.isPopdistEnvSet() and popdist.getInstanceIndex() == 0:
+    print(f'\n[Epoch {epoch:02d}] loss: {epoch_loss:.3f}, acc: {epoch_accuracy:.1f}', end='')
+elif not popdist.isPopdistEnvSet():
+"    "print(f'[Epoch {epoch:02d}] loss: {epoch_loss:.3f}, acc: {epoch_accuracy:.1f}')
```

### Modifying [day03/utils.py](../../day03/utils.py)
- Import `popdist` and `popdist.poptorch`.
```diff
import argparse

+import popdist
+import popdist.poptorch
import poptorch
import torch
```
- When `poprun` is invoked, replication factor needs to be set via `poprun`.
```diff
+    if not popdist.isPopdistEnvSet():
        parser.add_argument('--replicas', default=1, type=int, help='replication factor for data parallel')
```
- If `poprun` is invoked, use `popdist.poptorch.Options` instead of `poptorch.Options`.
In addition, you need to set a few other options.
  * `opts.randomSeed(2022)` - Required to initialize weights and dataloader with the same random seed.
  * `opts.showCompilationProgressBar(popdist.getInstanceIndex() == 0)` - Recommended for better readability.
  * `opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Sum)` - Note that this option defaults to `poptorch.ReductionType.Mean`, which can cause smoothing of gradients when using (distributed) data parallelism.
```diff
def set_ipu_options(args):
-    opts = poptorch.Options()
+    if popdist.isPopdistEnvSet():
+        opts = popdist.poptorch.Options()
+        opts.randomSeed(2022)
+        opts.showCompilationProgressBar(popdist.getInstanceIndex() == 0)
+    else:
+        opts = poptorch.Options()
+        opts.replicationFactor(args.replicas)
    opts.enableExecutableCaching('../cache')
    opts.Training.gradientAccumulation(args.gradient_accumulation)
+    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Sum)
    opts.deviceIterations(args.device_iterations)
-    opts.replicationFactor(args.replicas)
    return opts
```

That's it! We are ready to run DDP training.

## Execution Results
### Execution without DP
Recall that we have applied a number of techniques for efficient data loading.
```bash
python main.py \
  --batch-size 64 \
  --gradient-accumulation 8 \
  --device-iterations 4 \
  --eight-bit-io \
  --async-dataloader
```

<details><summary>Output</summary><p>

```
Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
[Epoch 01]:   0%|          | 0/24 [00:00<?, ?it/s][19:04:43.864] [poptorch:cpp] [warning] %3860 : Float(64, strides=[1], requires_grad=1, device=cpu), %ind : Long(64, strides=[1], requires_grad=0, device=cpu) = aten::max(%input, %4686, %4731) # main.py:16:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
Graph compilation: 100%|██████████| 100/100 [00:24<00:00]
[Epoch 01]: 100%|██████████| 24/24 [00:48<00:00,  2.03s/it, loss=1.45, acc=48.4]
[Epoch 01] loss: 1.835, acc: 32.3
[Epoch 02]: 100%|██████████| 24/24 [00:06<00:00,  3.45it/s, loss=1.14, acc=56.2]
[Epoch 02] loss: 1.390, acc: 49.9
[Epoch 03]: 100%|██████████| 24/24 [00:06<00:00,  3.45it/s, loss=1.22, acc=57.8] 
[Epoch 03] loss: 1.218, acc: 55.7
[Epoch 04]: 100%|██████████| 24/24 [00:06<00:00,  3.50it/s, loss=1.03, acc=65.6] 
[Epoch 04] loss: 1.067, acc: 61.9
[Epoch 05]: 100%|██████████| 24/24 [00:06<00:00,  3.53it/s, loss=0.863, acc=68.8]
[Epoch 05] loss: 0.948, acc: 66.9
```

</p></details>

### Basic Data Parallel (Replication)
```bash
python main.py \
  --batch-size 64 \
  --gradient-accumulation 8 \
  --device-iterations 4 \
  --eight-bit-io \
  --async-dataloader \
  --replicas 8
```

<details><summary>Output</summary><p>

```
Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=8)
Files already downloaded and verified
[Epoch 01]:   0%|          | 0/3 [00:00<?, ?it/s][19:06:09.556] [poptorch:cpp] [warning] %3860 : Float(64, strides=[1], requires_grad=1, device=cpu), %ind : Long(64, strides=[1], requires_grad=0, device=cpu) = aten::max(%input, %4737, %4759) # main.py:16:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
Graph compilation: 100%|██████████| 100/100 [07:52<00:00]
[Epoch 01]: 100%|██████████| 3/3 [08:36<00:00, 172.31s/it, loss=1.99, acc=27.5]
[Epoch 01] loss: 2.281, acc: 21.7
[Epoch 02]: 100%|██████████| 3/3 [00:06<00:00,  2.10s/it, loss=1.74, acc=35.7]
[Epoch 02] loss: 1.812, acc: 32.5
[Epoch 03]: 100%|██████████| 3/3 [00:06<00:00,  2.03s/it, loss=1.53, acc=41.6]
[Epoch 03] loss: 1.619, acc: 39.0
[Epoch 04]: 100%|██████████| 3/3 [00:06<00:00,  2.05s/it, loss=1.42, acc=50.6]
[Epoch 04] loss: 1.465, acc: 46.5
[Epoch 05]: 100%|██████████| 3/3 [00:06<00:00,  2.03s/it, loss=1.38, acc=50.6]
[Epoch 05] loss: 1.390, acc: 49.5
```

</p></details>

### Distributed Data Parallel
```bash
poprun \
  --num-replicas 8 \
  --num-instances 4 \
  --ipus-per-replica 2 \
python main.py \
  --batch-size 64 \
  --gradient-accumulation 8 \
  --device-iterations 4 \
  --eight-bit-io \
  --async-dataloader
```

<details><summary>Output</summary><p>

```
[1,1]<stdout>:Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16')
[1,3]<stdout>:Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16')
[1,0]<stdout>:Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16')
[1,2]<stdout>:Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16')
[1,3]<stdout>:Files already downloaded and verified
[1,1]<stdout>:Files already downloaded and verified
[1,0]<stdout>:Files already downloaded and verified
[1,2]<stdout>:Files already downloaded and verified
[Epoch 01]:   0%|          | 0/3 [00:00<?, ?it/s][1,2]<stderr>:[19:15:54.683] [poptorch:cpp] [warning] %3860 : Float(64, strides=[1], requires_grad=1, device=cpu), %ind : Long(64, strides=[1], requires_grad=0, device=cpu) = aten::max(%input, %4771, %4889) # main.py:16:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
[1,0]<stderr>:[19:16:22.488] [poptorch:cpp] [warning] %3860 : Float(64, strides=[1], requires_grad=1, device=cpu), %ind : Long(64, strides=[1], requires_grad=0, device=cpu) = aten::max(%input, %4732, %4748) # main.py:16:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
[1,1]<stderr>:[19:16:22.491] [poptorch:cpp] [warning] %3860 : Float(64, strides=[1], requires_grad=1, device=cpu), %ind : Long(64, strides=[1], requires_grad=0, device=cpu) = aten::max(%input, %4659, %4756) # main.py:16:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
[1,3]<stderr>:[19:16:22.494] [poptorch:cpp] [warning] %3860 : Float(64, strides=[1], requires_grad=1, device=cpu), %ind : Long(64, strides=[1], requires_grad=0, device=cpu) = aten::max(%input, %4706, %4755) # main.py:16:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
Graph compilation: 100%|██████████| 100/100 [00:27<00:00][1,0]<stderr>:[1,0]<stderr>:
[Epoch 01]: 100%|██████████| 3/3 [01:20<00:00, 26.95s/it, loss=2.11, acc=22.7][1,0]<stderr>:
[1,0]<stdout>:
[Epoch 02]: 100%|██████████| 3/3 [00:01<00:00,  2.12it/s, loss=1.67, acc=40][1,0]<stderr>:>:
[1,0]<stdout>:
[Epoch 03]: 100%|██████████| 3/3 [00:01<00:00,  1.81it/s, loss=1.52, acc=43.9][1,0]<stderr>:
[1,0]<stdout>:
[Epoch 04]: 100%|██████████| 3/3 [00:01<00:00,  1.81it/s, loss=1.5, acc=45.1][1,0]<stderr>::
[1,0]<stdout>:
[Epoch 05]: 100%|██████████| 3/3 [00:01<00:00,  1.83it/s, loss=1.29, acc=53.9][1,0]<stderr>:
[1,0]<stdout>:
[1,0]<stdout>:[Epoch 05] loss: 1.316, acc: 52.9
```

</p></details>

### Comparison


## What Is Happening Under The Hood?
