# Data Parallelism

## Data Parallelism with Pytorch

In pytorch, you can leverage data parallelism through `poptorch.Options()`. You can specify the number of replicas with `replicationFactor()`.
```python
opts = poptorch.Options()
opts.replicationFactor(2)
```
With this code, 2 replicas will be generated on different IPUs. If the model is fit on 1 IPU, 2 IPUs will be used in total and if the model is pipelined over 2 IPUs, 4 IPUs will be used in total.

## Data Parallelism with Tensorflow

In tensorflow, you can leverage data parallelism through `ipu.config.IPUConfig()`. But, unlike pytorch you will specify the total number of IPUs to be used, so that the number of replicas will be calculated as `total number of IPUs/number of model pipelines`.
```python
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()
```
With this code, the total number of IPUs is fixed to 2, so if the model is fit on 1 IPU, it will generate 2 replicas and if the model is pipelined over 2 IPUs, it will be run only with 1 replica.

## Running the code

We will practice this tutorial with pytorch. Before running the code, be sure that the proper virtual environment is activated. We will use the same [environment](../../day01/2.%20Running%20Pytorch%20on%20IPU/README.md#L3) we created in day01.

Activate proper virtual environment first.
```bash
(venv_tf)$ deactivate
$ source ../../venv_pytorch/bin/activate
```

`base_pytorch.py` is a code to train `resnet50` with 1 replica.
```bash
(venv_pytorch)$ nohup python base_pytorch.py &
```

<details><summary>Output </summary><p>

```
Files already downloaded and verified
[Epoch 01]:   0%|          | 0/6250 [00:00<?, ?it/s2022-09-07T05:52:12.503278Z popart:devicex 252340.252340 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
2022-09-07T05:52:21.985298Z popart:devicex 252340.252340 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
Graph compilation: 100%|██████████| 100/100 [03:48<00:00]
[Epoch 01]: 100%|██████████| 6250/6250 [04:51<00:00, 21.43it/s, Loss=1.47]  
[Epoch 02]: 100%|██████████| 6250/6250 [00:56<00:00, 110.37it/s, Loss=2.55] 
[Epoch 03]: 100%|██████████| 6250/6250 [00:56<00:00, 110.54it/s, Loss=1.23] 
[Epoch 04]: 100%|██████████| 6250/6250 [00:56<00:00, 110.26it/s, Loss=0.934]
[Epoch 05]: 100%|██████████| 6250/6250 [00:56<00:00, 110.47it/s, Loss=1.42]
```

</p></details>

This time, let's run the training with 2 replicas. You just need to add one more line which you can see in `data_parallel_pytorch.py`.
```diff
+   opts.replicationFactor(2)
```
Run the code.
```bash
(venv_pytorch)$ nohup python data_parallel_pytorch.py &
```

<details><summary>Output </summary><p>

```
Files already downloaded and verified
[Epoch 01]:   0%|          | 0/3125 [00:00<?, ?it/s2022-09-07T05:41:12.761929Z popart:devicex 250725.250725 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
2022-09-07T05:41:26.216625Z popart:devicex 250725.250725 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
Graph compilation: 100%|██████████| 100/100 [04:02<00:00]
[Epoch 01]: 100%|██████████| 3125/3125 [05:00<00:00, 10.39it/s, Loss=2.28]  
[Epoch 02]: 100%|██████████| 3125/3125 [00:48<00:00, 63.86it/s, Loss=1.86] 
[Epoch 03]: 100%|██████████| 3125/3125 [00:50<00:00, 62.47it/s, Loss=1.99] 
[Epoch 04]: 100%|██████████| 3125/3125 [00:49<00:00, 62.70it/s, Loss=1.22] 
[Epoch 05]: 100%|██████████| 3125/3125 [00:49<00:00, 63.76it/s, Loss=0.809]
```

</p></details>

As you can see, the number of iterations became half when 2 replicas is used. So, you can boost your training with larger batch size.

While running the code, you can see 2 IPUs are operating through `gc-monitor` command.
```
+---------------+---------------------------------------------------------------------------------+
|  gc-monitor   |        Partition: pt-trainee01-4-ipus [active] has 4 reconfigurable IPUs        |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type  | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|  10.1.5.9   | 0029.0002.8211021  |        |    1.9.0     |  2.4.4   | M2000 | 0  |  3   |  DNC  |
|  10.1.5.9   | 0029.0002.8211021  |        |    1.9.0     |  2.4.4   | M2000 | 1  |  2   |  DNC  |
|  10.1.5.9   | 0029.0001.8211021  |        |    1.9.0     |  2.4.4   | M2000 | 2  |  1   |  DNC  |
|  10.1.5.9   | 0029.0001.8211021  |        |    1.9.0     |  2.4.4   | M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+-----------------+
|                                                                      Attached processes in partition pt-trainee01-4-ipus                                                                       |          IPU           |      Board      |
+--------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
|  PID   |                                                                             Command                                                                             |  Time  |    User    | ID |  Clock   |  Temp  |  Temp  | Power  |
+--------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
| 203893 |                                                                             python                                                                              |  11s   | trainee01  | 0  | 1330MHz  |  N/A   |  N/A   |  N/A   |
| 203893 |                                                                             python                                                                              |  11s   | trainee01  | 1  | 1330MHz  |  N/A   |        |        |
+--------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
```

[Continue to the next chapter](../3.%20Model%20Parallelism)