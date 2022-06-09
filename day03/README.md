# Memory and Performance Optimisation on the IPU

This tutorial contains basic optimization tips for using IPUs. Check out [MEMORY AND PERFORMANCE OPTIMISATION ON THE IPU](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/index.html#memory-and-performance-optimisation-on-the-ipu) for more advanced techniques and [IPU_hands-on_day03.pdf](https://github.com/matildalab/hands-on-training/blob/main/day03/IPU_hands-on_day03.pdf) for more detailed explanation of the codes.

## Half Precision

Several precision formats can be used during the model execution.
- Single precision : FP32 for both the data and the model
- Mixed precision : FP16 for the data and FP32 for the model
- Half precision : FP16 for both the data and the model

Using lower precision can reduce the memory usage dramatically. Poplar SDK contains many tools to support stable training with half precision, **so it's highly recommended to use half precision on the IPU**.
```bash
$ python main.py --precision 16.16 --gradient-accumulation 3
Namespace(async_dataloader=False, batch_size=32, device_iterations=1, eight_bit_io=False, epochs=5, gradient_accumulation=3, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:25<00:00]
[Epoch 01]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 520/520 [08:40<00:00,  1.00s/it, Loss=1.37]
[Epoch 01] loss = 1.711
[Epoch 02]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 520/520 [01:00<00:00,  8.64it/s, Loss=1.27]
[Epoch 02] loss = 1.271
[Epoch 03]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 520/520 [01:00<00:00,  8.63it/s, Loss=1.13]
[Epoch 03] loss = 1.023
[Epoch 04]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 520/520 [01:00<00:00,  8.64it/s, Loss=1.14]
[Epoch 04] loss = 0.873
[Epoch 05]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 520/520 [01:00<00:00,  8.65it/s, Loss=0.746]
[Epoch 05] loss = 0.750
```
More information can be found in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#half-float16-support).

## Gradient Accumulation

The gradient accumulation value must be at least `2 * ipus_per_replica - 1`.
In this case, model is pipelined over 2 IPUs:
```python
model = torchvision.models.resnet101()
model.layer3[8] = poptorch.BeginBlock(model.layer3[8], ipu_id=1)
```
So `ipus_per_replica=2`, meaning that gradient accumulation must be *at least 3*.\
Increasing gradient accumulation results in higher throughput by maximizing the portion of [main execution phase](https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html#pipeline-operation) during pipelining.
It also results in more samples per weight update without increase in memory usage.
```bash
$ python main.py --gradient-accumulation 8
Namespace(async_dataloader=False, batch_size=32, device_iterations=1, eight_bit_io=False, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:26<00:00]
[Epoch 01]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 195/195 [08:29<00:00,  2.61s/it, Loss=1.53]
[Epoch 01] loss = 1.817
[Epoch 02]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 195/195 [00:49<00:00,  3.93it/s, Loss=1.19]
[Epoch 02] loss = 1.392
[Epoch 03]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 195/195 [00:49<00:00,  3.92it/s, Loss=1.04]
[Epoch 03] loss = 1.156
[Epoch 04]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 195/195 [00:49<00:00,  3.92it/s, Loss=0.844]
[Epoch 04] loss = 1.022
[Epoch 05]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 195/195 [00:49<00:00,  3.93it/s, Loss=0.824]
[Epoch 05] loss = 0.864
```
Gradient accumulation can be set via poptorch option:
```python
opts = poptorch.Options()
opts.Training.gradientAccumulation(args.gradient_accumulation)
```
More information can be found in the [documentation]([documentaion](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#poptorch-options-training-gradientaccumulation)).

## Device Iterations

Increasing device iterations reduces the training time by reducing the number of host-device communications. Note that setting the device iteration number higher than the optimal might cause I/O overhead which can be resulted in inefficient training.
```bash
$ python main.py --gradient-accumulation 8 --device-iterations 4
Namespace(async_dataloader=False, batch_size=32, device_iterations=4, eight_bit_io=False, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:26<00:00]
[Epoch 01]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [08:27<00:00, 10.56s/it, Loss=1.46]
[Epoch 01] loss = 1.794
[Epoch 02]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:46<00:00,  1.02it/s, Loss=0.96]
[Epoch 02] loss = 1.344
[Epoch 03]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:46<00:00,  1.02it/s, Loss=0.856]
[Epoch 03] loss = 1.127
[Epoch 04]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:46<00:00,  1.02it/s, Loss=0.938]
[Epoch 04] loss = 0.950
[Epoch 05]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:46<00:00,  1.02it/s, Loss=0.804]
[Epoch 05] loss = 0.826
```
Device iterations can be set via poptorch option:
```python
opts = poptorch.Options()
opts.deviceIterations(args.device_iterations)
```
More information can be found in the [documentation]([documentaion](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#poptorch-options-deviceiterations)).

## 8-bit I/O

The data type of input data passed to the device from the host are normally `float16` or `float32`. 
Host-Device I/O can be optimized by casting the input data to `uint8` type.
In this case, normalization which results in casting the data to `float16`(or `float32`) should be conducted on the device.

```bash
$ python main.py --gradient-accumulation 8 --device-iterations 4 --eight-bit-io
Namespace(async_dataloader=False, batch_size=32, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:26<00:00]
[Epoch 01]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [07:56<00:00,  9.93s/it, Loss=1.42]
[Epoch 01] loss = 1.769
[Epoch 02]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:15<00:00,  3.06it/s, Loss=1.11]
[Epoch 02] loss = 1.345
[Epoch 03]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:15<00:00,  3.06it/s, Loss=1.18]
[Epoch 03] loss = 1.130
[Epoch 04]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:15<00:00,  3.06it/s, Loss=0.835]
[Epoch 04] loss = 0.989
[Epoch 05]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:15<00:00,  3.06it/s, Loss=0.888]
[Epoch 05] loss = 0.887
```

## Asynchronous Data Loading

Asynchronous mode can reduce host overhead by offloading the data loading process to a seperate thread. By this mode, the host can start loading the next batch while the IPU is running.
```bash
$ python main.py --gradient-accumulation 8 --device-iterations 4 --eight-bit-io --async-dataloader
Namespace(async_dataloader=True, batch_size=32, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [07:25<00:00]
[Epoch 01]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [07:47<00:00,  9.74s/it, Loss=1.52]
[Epoch 01] loss = 1.788
[Epoch 02]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:08<00:00,  5.62it/s, Loss=1.17]
[Epoch 02] loss = 1.333
[Epoch 03]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:08<00:00,  5.60it/s, Loss=1.07]
[Epoch 03] loss = 1.174
[Epoch 04]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:08<00:00,  5.73it/s, Loss=1.15]
[Epoch 04] loss = 1.033
[Epoch 05]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:08<00:00,  5.78it/s, Loss=0.842]
[Epoch 05] loss = 0.869
```
Asynchronous data loading can be set via argument of the `poptorch.DataLoader`.
```python
mode = poptorch.DataLoaderMode.Async if args.async_dataloader \
        else poptorch.DataLoaderMode.Sync

train_dataloader = poptorch.DataLoader(opts,
                                        train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        mode=mode)
```
More information can be found in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#poptorch-asynchronousdataaccessor).