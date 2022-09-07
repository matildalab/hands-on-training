# Model Parallelism

## Model Parallelism with Pytorch

In pytorch, Model parallelism can be set in 2 ways. First, if you write the model architecture by yourself, you can use `poptorch.Block()`. This code loads `layer1` on IPU0, `layer2` on IPU1, `layer3` and `layer4` on IPU2 and `softmax` layer on IPU3.
```diff
class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 10)
        self.layer2 = torch.nn.Linear(10, 5)
        self.layer3 = torch.nn.Linear(5, 5)
        self.layer4 = torch.nn.Linear(5, 5)

        self.act = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Explicit layers on a certain IPU
        poptorch.Block.useAutoId()
+       with poptorch.Block(ipu_id=0):
            x = self.act(self.layer1(x))

+       with poptorch.Block(ipu_id=1):
            x = self.act(self.layer2(x))

+       with poptorch.Block(ipu_id=2):
            x = self.act(self.layer3(x))
            x = self.act(self.layer4(x))

+       with poptorch.Block(ipu_id=3):
            x = self.softmax(x)
        return x
```

When you import models from other libraries as in most cases, you can use `poptorch.BeginBlock()`. As the name indicates, you can specify a block and from that block and the rest of the model will be loaded on the specified IPU. In this code below, the whole model is loaded on IPU0 by default. And by using `poptorch.BeginBlock()`, from `layer4` to the end will be loaded on IPU1. You can check the model layers by `print` the model and you can decide at which block to be splited using Graphcore visualization tool called [popvision](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/introduction.html). Popvision will be introduced deeply in the next chapter.
```diff
    model = torchvision.models.resnet50()
+   model.layer4 = poptorch.BeginBlock(model.layer4, ipu_id=1)
```

## Model Parallelism with Tensorflow

In tensorflow, you can use 2 different APIs as in pytorch. In the first usecase, you can use `keras.ipu.PipelineStage()`
```diff
    input_layer = keras.layers.Input((28, 28))

+   with keras.ipu.PipelineStage(0):
        x = keras.layers.Dense(8)(input_layer)
        x = keras.layers.Dense(16)(x)

+   with keras.ipu.PipelineStage(1):
        x = keras.layers.Dense(16)(x)
        x = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=input_layer, outputs=x)
```
And you can use `get_pipeline_stage_assignment()` and `set_pipeline_stage_assignment()` when you import models.
```python
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():

    model = resnet.ResNet50(weights='imagenet')

    # Get the individual assignments - note that they are returned in post-order.
    assignments = model.get_pipeline_stage_assignment()

    # Iterate over them and set their pipeline stages.
    stage_id = 0
    for assignment in assignments:
        assignment.pipeline_stage = stage_id
        # Split the model on the `conv4_block2_add` layer.
        if assignment.layer.name.startswith("conv4_block2_add"):
            stage_id = 1

    # Set the assignments to the model.
    model.set_pipeline_stage_assignment(assignments)

    model.print_pipeline_stage_assignment_summary()
```

## Running the code

We will practice this tutorial with pytorch. Before running the code, be sure that the proper virtual environment is activated. We will use the same [environment](../../day01/2.%20Running%20Pytorch%20on%20IPU/README.md#L3) we created in day01.

From the code you made in the previous chapter, add these two lines. You will be obliged to use `gradientAccumulation` together which will be introduced in day03. Now as the model is splited once at `layer4`, the model is pipelined over 2 IPUs.
```diff   
    opts = poptorch.Options()
+   opts.Training.gradientAccumulation(3)
    opts.replicationFactor(2)

    model = torchvision.models.resnet50()
+   model.layer4 = poptorch.BeginBlock(model.layer4, ipu_id=1)
```

<details><summary>Output </summary><p>

```
Files already downloaded and verified
[Epoch 01]:   0%|          | 0/1041 [00:00<?, ?it/s2022-09-07T07:26:42.345814Z popart:devicex 177873.177873 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
2022-09-07T07:26:48.430879Z popart:devicex 177873.177873 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
Graph compilation: 100%|██████████| 100/100 [04:32<00:00]
[Epoch 01]: 100%|██████████| 1041/1041 [05:18<00:00,  3.27it/s, Loss=2.2]  
[Epoch 02]: 100%|██████████| 1041/1041 [00:37<00:00, 27.54it/s, Loss=2.04]
[Epoch 03]: 100%|██████████| 1041/1041 [00:36<00:00, 28.66it/s, Loss=1.76]
[Epoch 04]: 100%|██████████| 1041/1041 [00:35<00:00, 29.17it/s, Loss=1.49]
[Epoch 05]: 100%|██████████| 1041/1041 [00:35<00:00, 29.25it/s, Loss=1.86] 
```

</p></details>

While running the code, you can see 4 IPUs are operating through `gc-monitor` command.
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
| 177873 |                                                                             python                                                                              |  1m3s  | trainee01  | 0  | 1330MHz  |  N/A   |  N/A   |  N/A   |
| 177873 |                                                                             python                                                                              |  1m3s  | trainee01  | 1  | 1330MHz  |  N/A   |        |        |
| 177873 |                                                                             python                                                                              |  1m3s  | trainee01  | 2  | 1330MHz  |  N/A   |        |        |
| 177873 |                                                                             python                                                                              |  1m3s  | trainee01  | 3  | 1330MHz  |  N/A   |        |        |
+--------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
```

[Continue to the next chapter](../4.%20Popvision%20Graph%20Analyser)