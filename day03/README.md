# Memory and Performance Optimisation on the IPU

This tutorial contains basic optimization tips for using IPUs. Check out [MEMORY AND PERFORMANCE OPTIMISATION ON THE IPU](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/index.html#memory-and-performance-optimisation-on-the-ipu) for more advanced techniques and [IPU_hands-on_day03.pdf](./IPU_hands-on_day03.pdf) for more detailed performance results.
Note that this sample code is NOT heavily optimized in that several un-optimized implementations are used. This tutorial focuses on easy understanding of each techniques with minimum code changes.

## Half Precision

Several precision formats can be used during the model execution.
- Single precision : FP32 for both the data and the model
- Mixed precision : FP16 for the data and FP32 for the model
- Half precision : FP16 for both the data and the model

Using lower precision can reduce the memory usage dramatically. Poplar SDK contains many tools to support stable training with half precision, **so it's highly recommended to use half precision on the IPU**.

Command:
```bash
$ python main.py --batch-size 64 --precision 16.16 --gradient-accumulation 3
```
Output:
```bash
Namespace(async_dataloader=False, batch_size=64, device_iterations=1, eight_bit_io=False, epochs=5, gradient_accumulation=3, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|██████████| 100/100 [08:16<00:00]
[Epoch 01]: 100%|██████████| 260/260 [09:24<00:00,  2.17s/it, loss=1.51, acc=43.8]  
[Epoch 01] loss: 1.760, acc: 37.2
[Epoch 02]: 100%|██████████| 260/260 [00:52<00:00,  4.94it/s, loss=0.934, acc=65.6]
[Epoch 02] loss: 1.286, acc: 53.2
[Epoch 03]: 100%|██████████| 260/260 [00:52<00:00,  4.94it/s, loss=1.08, acc=64.1] 
[Epoch 03] loss: 1.060, acc: 62.6
[Epoch 04]: 100%|██████████| 260/260 [00:52<00:00,  4.95it/s, loss=0.747, acc=76.6]
[Epoch 04] loss: 0.881, acc: 69.0
[Epoch 05]: 100%|██████████| 260/260 [00:52<00:00,  4.95it/s, loss=0.641, acc=76.6]
[Epoch 05] loss: 0.744, acc: 74.0
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

Command:
```bash
$ python main.py --batch-size 64 --precision 16.16 --gradient-accumulation 8
```
Output:
```bash
Namespace(async_dataloader=False, batch_size=64, device_iterations=1, eight_bit_io=False, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|██████████| 100/100 [08:20<00:00]
[Epoch 01]: 100%|██████████| 97/97 [09:20<00:00,  5.78s/it, loss=1.56, acc=39.1]   
[Epoch 01] loss: 1.900, acc: 33.7
[Epoch 02]: 100%|██████████| 97/97 [00:46<00:00,  2.10it/s, loss=1.01, acc=67.2]
[Epoch 02] loss: 1.375, acc: 50.0
[Epoch 03]: 100%|██████████| 97/97 [00:46<00:00,  2.10it/s, loss=1.31, acc=54.7] 
[Epoch 03] loss: 1.177, acc: 58.0
[Epoch 04]: 100%|██████████| 97/97 [00:46<00:00,  2.11it/s, loss=0.935, acc=67.2]
[Epoch 04] loss: 1.056, acc: 61.9
[Epoch 05]: 100%|██████████| 97/97 [00:46<00:00,  2.09it/s, loss=0.905, acc=64.1]
[Epoch 05] loss: 0.921, acc: 67.3
```
Gradient accumulation can be set via poptorch option:
```python
opts = poptorch.Options()
opts.Training.gradientAccumulation(args.gradient_accumulation)
```
More information can be found in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#poptorch-options-training-gradientaccumulation).

## Device Iterations

Increasing device iterations reduces the training time by reducing the number of host-device communications. Note that setting the device iteration number higher than the optimal might cause I/O overhead which can be resulted in inefficient training.

Command:
```bash
$ python main.py --batch-size 64 --precision 16.16 --gradient-accumulation 8 --device-iteration 4
```
Output:
```bash
Namespace(async_dataloader=False, batch_size=64, device_iterations=4, eight_bit_io=False, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|██████████| 100/100 [08:10<00:00]
[Epoch 01]: 100%|██████████| 24/24 [09:11<00:00, 22.97s/it, loss=1.51, acc=45.3]  
[Epoch 01] loss: 1.759, acc: 36.5
[Epoch 02]: 100%|██████████| 24/24 [00:45<00:00,  1.88s/it, loss=1.11, acc=57.8]
[Epoch 02] loss: 1.395, acc: 48.3
[Epoch 03]: 100%|██████████| 24/24 [00:45<00:00,  1.88s/it, loss=0.841, acc=75] 
[Epoch 03] loss: 1.251, acc: 55.1
[Epoch 04]: 100%|██████████| 24/24 [00:45<00:00,  1.88s/it, loss=1.37, acc=54.7] 
[Epoch 04] loss: 1.086, acc: 61.8
[Epoch 05]: 100%|██████████| 24/24 [00:45<00:00,  1.88s/it, loss=0.744, acc=73.4]
[Epoch 05] loss: 0.863, acc: 68.9
```
Device iterations can be set via poptorch option:
```python
opts = poptorch.Options()
opts.deviceIterations(args.device_iterations)
```
More information can be found in the [documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html#poptorch-options-deviceiterations).

## 8-bit I/O

The data type of input data passed to the device from the host are normally `float16` or `float32`. 
Host-Device I/O can be optimized by casting the input data to `uint8` type.
In this case, normalization which results in casting the data to `float16`(or `float32`) should be conducted on the device.

Command:
```bash
$ python main.py --batch-size 64 --precision 16.16 --gradient-accumulation 8 --device-iteration 4 --eight-bit-io
```
Output:
```bash
Namespace(async_dataloader=False, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|██████████| 100/100 [08:07<00:00]
[Epoch 01]: 100%|██████████| 24/24 [08:35<00:00, 21.48s/it, loss=1.65, acc=34.4]  
[Epoch 01] loss: 1.848, acc: 32.6
[Epoch 02]: 100%|██████████| 24/24 [00:13<00:00,  1.77it/s, loss=1.33, acc=51.6]
[Epoch 02] loss: 1.334, acc: 51.8
[Epoch 03]: 100%|██████████| 24/24 [00:13<00:00,  1.77it/s, loss=1.44, acc=50]   
[Epoch 03] loss: 1.205, acc: 56.3
[Epoch 04]: 100%|██████████| 24/24 [00:13<00:00,  1.77it/s, loss=0.639, acc=81.2]
[Epoch 04] loss: 1.004, acc: 63.4
[Epoch 05]: 100%|██████████| 24/24 [00:13<00:00,  1.77it/s, loss=0.903, acc=71.9]
[Epoch 05] loss: 0.926, acc: 67.2
```

## Asynchronous Data Loading

Asynchronous mode can reduce host overhead by offloading the data loading process to a seperate thread. By this mode, the host can start loading the next batch while the IPU is running.

Command:
```bash
$ python main.py --batch-size 64 --precision 16.16 --gradient-accumulation 8 --device-iteration 4 --eight-bit-io --async-dataloader
```
Output:
```bash
Namespace(async_dataloader=True, batch_size=64, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=1)
Files already downloaded and verified
Graph compilation: 100%|██████████| 100/100 [08:10<00:00]
[Epoch 01]: 100%|██████████| 24/24 [08:32<00:00, 21.35s/it, loss=1.27, acc=50]    
[Epoch 01] loss: 1.758, acc: 34.2
[Epoch 02]: 100%|██████████| 24/24 [00:08<00:00,  2.75it/s, loss=1.4, acc=50]   
[Epoch 02] loss: 1.352, acc: 49.4
[Epoch 03]: 100%|██████████| 24/24 [00:08<00:00,  2.71it/s, loss=1.03, acc=65.6] 
[Epoch 03] loss: 1.200, acc: 56.9
[Epoch 04]: 100%|██████████| 24/24 [00:08<00:00,  2.78it/s, loss=0.954, acc=65.6]
[Epoch 04] loss: 1.039, acc: 64.2
[Epoch 05]: 100%|██████████| 24/24 [00:08<00:00,  2.76it/s, loss=1.1, acc=62.5]  
[Epoch 05] loss: 1.003, acc: 65.1
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

## Next Step
Accelerate even further with [distributed data parallelism](../day04).