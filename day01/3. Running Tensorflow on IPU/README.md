# Running Tensorflow on IPU

## How to setup

Let's create a virtual environment and activate it first
```bash
$ virtualenv ../../venv_tf -p python3
$ source ../../venv_tf/bin/activate
```
Instead of using the original tensorflow, we will use IPU-tensorflow included in the SDK. If you use Keras, you should install IPU-keras as well. They are all included in the SDK. Install `tensorflow`, `keras` and `ipu_tensorflow_addons`. `ipu_tensorflow_addons` includes some IPU-optimized layers and IPU-optimized optimizers. When installing `tensorflow` and `ipu_tensorflow_addons`, you have to specify cpu vendor and tensorflow version between tensorflow1 and tensorflow2.

You can check the cpu vendor with `cat /proc/cpuinfo` command.
```bash
(venv_tf)$ pip install [SDK-path]/tensorflow-2*amd*.whl
(venv_tf)$ pip install [SDK-path]/ipu_tensorflow_addons-2*.whl
(venv_tf)$ pip install [SDK-path]/keras*.whl
```

## IPU Porting

`tensorflow_gpu.py` is a gpu-running code of running a simple keras model with MNIST dataset. We will convert this code into ipu-running code with minimum code changes. If you run `tensorflow_gpu.py` on IPU-POD, it will run on CPU as there is no GPU device. The model looks like below.
```
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
```

Porting tensorflow is much simpler than pytorch as we use IPU-tensorflow from the beginning.

### IPUConfig

You have to initialize `ipu.config.IPUConfig()` first and you can set IPU configuration with it. For example, you can specify a number of IPU to be used with `config.auto_select_ipus`. After setting configuration you must call `configure_ipu_system()`, so that the configuration is actually applied to IPU.
```diff
+   from tensorflow.python import ipu

+   config = ipu.config.IPUConfig()
+   config.auto_select_ipus = 1
+   config.configure_ipu_system()
```

### IPUStrategy

Then you need to initialize `ipu.ipu_strategy.IPUStrategy()` which ensures the program targets a system with one or more IPUs. And you should use `strategy.scope()` context to ensure that everything within that context will be compiled for the IPU device. You should do this instead of using the `tf.device` context.
```diff
+   strategy = ipu.ipu_strategy.IPUStrategy()
+   with strategy.scope():
        model = create_model()

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

        model.fit(dataset, epochs=5)
```

### IPU Tensorflow Addons

As mentioned before, `ipu_tensorflow_addons` offers various IPU-optimized [layers](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/ipu_tensorflow_addons.html#keras-layers) and [optimizers](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/ipu_tensorflow_addons/ipu_tensorflow_addons.html#optimizers). They show faster execution and use less memory and also have some IPU-specific arguments for extreme optimization. So, if your model has any of these layers and optimizers, it is highly recommended to replace them with the one from `ipu_tensorflow_addons`.

For example, if you use `LSTM` layers in your model, you can simply replace it by importing it from `ipu_tensorflow_addons`.
```diff
-   from tensorflow.keras.layers import LSTM
+   from ipu_tensorflow_addons.keras.layers import LSTM
```

## Running the code

Now, we can run the tensorflow model on IPU. The completed code is available on `tensorflow_ipu.py`.
```bash
(venv_tf)$ nohup python tensorflow_ipu.py &
```
<details><summary>Output </summary><p>

```
2022-09-06 23:59:34.180032: I tensorflow/compiler/plugin/poplar/driver/poplar_platform.cc:43] Poplar version: 2.6.0 (e0ab3b4f12) Poplar package: a313c81b39
2022-09-06 23:59:35.817746: I tensorflow/compiler/plugin/poplar/driver/poplar_executor.cc:1618] TensorFlow device /device:IPU:0 attached to 1 IPU with Poplar device ID: 0
2022-09-06 23:59:36.460931: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/5
2022-09-06 23:59:36.793552: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:210] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
Compiling module a_inference_train_function_869__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.616:
[##################################################] 100% Compilation Finished [Elapsed: 00:00:14.4]
2022-09-06 23:59:51.436002: I tensorflow/compiler/jit/xla_compilation_cache.cc:376] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
1875/1875 [==============================] - 20s 2ms/step - loss: 0.2052 - accuracy: 0.9392
Epoch 2/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0931 - accuracy: 0.9733
Epoch 3/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0690 - accuracy: 0.9808
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0568 - accuracy: 0.9843
Epoch 5/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0473 - accuracy: 0.9877
```

</p></details>

You can see that graph is being compiled before actual training/inference begins. For the next time you run the model with same computational graph, you can cache the compiled executable. Then, it won't spend time to compile the graph again from the next time. You can use executable caching by running the program with an environment variable `TF_POPLAR_FLAGS='--executable_cache_path=./cache'`.
```bash
(venv_tf)$ TF_POPLAR_FLAGS='--executable_cache_path=./cache' nohup python tensorflow_ipu.py &
```

While running the code, you can check IPU utilization through `gc-monitor` command. You can see a program is running on one IPU.
```
+---------------+---------------------------------------------------------------------------------+
|  gc-monitor   |        Partition: pt-trainee01-4-ipus [active] has 4 reconfigurable IPUs        |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type  | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|  10.1.5.7   | 0069.0002.8210521  |        |    1.9.0     |  2.4.4   | M2000 | 0  |  3   |  DNC  |
|  10.1.5.7   | 0069.0002.8210521  |        |    1.9.0     |  2.4.4   | M2000 | 1  |  2   |  DNC  |
|  10.1.5.7   | 0069.0001.8210521  |        |    1.9.0     |  2.4.4   | M2000 | 2  |  1   |  DNC  |
|  10.1.5.7   | 0069.0001.8210521  |        |    1.9.0     |  2.4.4   | M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
+-----------------------------------------------------------------------------------------------------------------+------------------------+-----------------+
|                               Attached processes in partition pt-trainee01-4-ipus                               |          IPU           |      Board      |
+--------+----------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
|  PID   |                                     Command                                      |  Time  |    User    | ID |  Clock   |  Temp  |  Temp  | Power  |
+--------+----------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
| 205698 |                                      python                                      |  13s   | trainee01  | 0  | 1330MHz  |  N/A   |  N/A   |  N/A   |
+--------+----------------------------------------------------------------------------------+--------+------------+----+----------+--------+--------+--------+
```

For for details, please visit [IPU Tensorflow2 User Guide](https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/index.html)

[Continue to the next chapter](../../day02)