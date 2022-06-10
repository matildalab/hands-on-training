# Chapter I - Overview
In [day03](../../day03), we have implemented model parallelism for ResNet-101 model over 2 IPUs for training.
In [day04](..), we are going to implement distributed data parallelism to leverage more IPUs for speeding up the training even further.

## Baseline
This is a summary for what happens during the training loop:
- Step 1: the host (or CPU) process loads and pre-processes inputs.
- Step 2: the inputs are sent to the IPUs
- Step 3: IPUs perform forward pass, backward pass and weight update
- Step 4: IPUs send the outputs and losses back to the host
- Step 5: The host performs post-processes the outputs and losses.

```marmaid
flowchart LR
    subgraph r0[Replica]
        subgraph model[Model]
            i0[IPU 0]
            i1[IPU 1]
        end
    r0desc{{Forward\nBackward\nWeight Update}}
    end
    subgraph h0[Host]
    p0([Process])
    h0desc{{Data Loading\nPre-processing\nPost-processing}}
    end
    p0-- infeed -->model
    model-- outfeed -->p0
```

##  Data Parallelism
You can boost the training with more IPUs using data parallelism.
In other words, you can run the step 3 with more than one replicas of the model.
```marmaid
flowchart LR
    subgraph h0[Host 0]
    p0([Process 0])
    end
    subgraph r0[Replica 0]
    i0[IPU 0]
    i1[IPU 1]
    end
    subgraph r1[Replica 1]
    i2[IPU 2]
    i3[IPU 3]
    end
    subgraph r2[Replica 2]
    i4[IPU 4]
    i5[IPU 5]
    end
    subgraph r3[Replica 3]
    i6[IPU 6]
    i7[IPU 7]
    end
    p0<-->r0
    p0<-->r1
    p0<-->r2
    p0<-->r3
```
You can set the number of replicas of the model via poptorch option:
```python
opts = poptorch.Options()
opts.replicationFactor(args.replicas)
```
For example, if you want to run the training with 4 replicas, this can be easily done with
[day03/main.py](../../day03/main.py) by specifying the option flag `--replicas 4`:
```bash
python main.py --gradient-accumulation 8 --device-iterations 4 --replicas 4 --eight-bit-io --async-dataloader
```
Output
```
Namespace(async_dataloader=True, batch_size=32, device_iterations=4, eight_bit_io=True, epochs=5, gradient_accumulation=8, precision='16.16', replicas=4)
Files already downloaded and verified
[Epoch 01]:   0%|                                                                                                                                  | 0/12 [00:00<?, ?it/s][23:52:27.477]
[poptorch:cpp] [warning] %3860 : Float(32, strides=[1], requires_grad=1, device=cpu), %ind : Long(32, strides=[1], requires_grad=0, device=cpu) = aten::max
(%input, %4812, %4830) # main.py:15:0: torch.int64 is not supported natively on IPU, loss of range/precision may occur. We will only warn on the first instance.
Graph compilation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [08:16<00:00]
[Epoch 01]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [08:55<00:00, 44.65s/it, loss=1.73, acc=41.4]
[Epoch 01] loss: 1.993, acc: 28.6
[Epoch 02]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.89it/s, loss=1.36, acc=53.9]
[Epoch 02] loss: 1.474, acc: 45.8
[Epoch 03]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.91it/s, loss=1.22, acc=59.4]
[Epoch 03] loss: 1.333, acc: 54.2
[Epoch 04]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.91it/s, loss=1.16, acc=57.8]
[Epoch 04] loss: 1.205, acc: 56.7
[Epoch 05]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:06<00:00,  1.94it/s, loss=1.13, acc=57]
[Epoch 05] loss: 1.149, acc: 57.7
```

## Distributed Data Parallelism
As you may have noticed, even though we have used 8 IPUs, the training time is about the same as when we used only 2 IPUs.
This is due to the extra bottleneck introduced in step 1 and step 2 since the host needs to prefetch more input data for the replicas.

In order to make the data parallelism more efficient, step 1 must be parallelized as well by using more host processes.
```marmaid
flowchart LR
    subgraph h0[Host 0]
    p0([Process 0])
    p1([Process 1])
    p2([Process 2])
    p3([Process 3])
    end
    subgraph r0[Replica 0]
    i0[IPU 0]
    i1[IPU 1]
    end
    subgraph r1[Replica 1]
    i2[IPU 2]
    i3[IPU 3]
    end
    subgraph r2[Replica 2]
    i4[IPU 4]
    i5[IPU 5]
    end
    subgraph r3[Replica 3]
    i6[IPU 6]
    i7[IPU 7]
    end
    p0<-->r0
    p1<-->r1
    p2<-->r2
    p3<-->r3
```

Furthermore, if you have more than one host servers configured with your IPU-POD, you may want to utilize multiple hosts.
```marmaid
flowchart LR
    subgraph h1[Host 1]
    p4([Process 4])
    p5([Process 5])
    p6([Process 6])
    p7([Process 7])
    end
    subgraph r4[Replica 4]
    i8[IPU 8]
    i9[IPU 9]
    end
    subgraph r5[Replica 5]
    i10[IPU 10]
    i11[IPU 11]
    end
    subgraph r6[Replica 6]
    i12[IPU 12]
    i13[IPU 13]
    end
    subgraph r7[Replica 7]
    i14[IPU 14]
    i15[IPU 15]
    end
    p4<-->r4
    p5<-->r5
    p6<-->r6
    p7<-->r7
    subgraph h0[Host 0]
    p0([Process 0])
    p1([Process 1])
    p2([Process 2])
    p3([Process 3])
    end
    subgraph r0[Replica 0]
    i0[IPU 0]
    i1[IPU 1]
    end
    subgraph r1[Replica 1]
    i2[IPU 2]
    i3[IPU 3]
    end
    subgraph r2[Replica 2]
    i4[IPU 4]
    i5[IPU 5]
    end
    subgraph r3[Replica 3]
    i6[IPU 6]
    i7[IPU 7]
    end
    p0<-->r0
    p1<-->r1
    p2<-->r2
    p3<-->r3
```

However, it could be cumbersome to implement the distributed data parallism as described above.
There are a number of constraints for this idea to work efficiently, for example:
- Each process need to load and pre-process different input data samples to its replicas.
- It requires reductions of gradients over multiple processes or multiple host servers.
- It requires to sync different loss values over multiple processes.

Fortunately, Poplar SDK comes with a set of tools to realize this idea without tremendous effort.

[continue to the next chapter](../2.%20PopRun%20and%20PopDist)
