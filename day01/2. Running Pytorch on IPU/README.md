# Running Pytorch on IPU

## How to setup

To use pytorch, you have to install poptorch which makes pytorch use Poplar libraries at the backend.
Let's create a virtual environment and activate it first
```bash
$ virtualenv venv_pytorch -p python3
$ source venv_pytorch/bin/activate
```
Install poptorch.
```bash
(venv_pytorch)$ pip install [SDK-path]/poptorch*.whl
```
While installing poptorch, it will automatically install compatible torch as well. So, you don't have to install pytorch seperately.
```
Package             Version
------------------- -----------
dataclasses         0.8
importlib-resources 5.4.0
pip                 21.3.1
pkg_resources       0.0.0
poptorch            2.6.0+74275
setuptools          59.6.0
torch               1.10.0+cpu
tqdm                4.64.1
typing_extensions   4.1.1
wheel               0.37.1
zipp                3.6.0
```
Then, install packages required for this tutorial with `requirements.txt`.
```bash
(venv_pytorch)$ pip install -r requirements.txt
```

## IPU Porting

`pytorch_gpu.py` is a gpu-running code of running a simple convolution model with Fashion MNIST dataset. We will convert this code into ipu-running code with minimum code changes. If you run `pytorch_gpu.py` on IPU-POD, it will run on CPU as there is no GPU device. The model looks like below.
```
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(972, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))

        return x
```

### Model
First of all, you have to initialize `poptorch.Options()`. It will be used to initialize poptorch model and dataloader later. You can set various options with it which will make your implementation much faster and easier. Details of the options will be introduced in the later chapters.
```diff
+   import poptorch

+   opts = poptorch.Options()
```

You don't have to modify layers inside the model, but to make it use Poplar libraries, you have to wrap the model with `poptorch.trainingModel` or `poptorch.inferenceModel`. As this wrapped instance automatically target IPUs, you don't have to specify on which device you desire to load the model.
```diff
    model = ClassificationModel()
    model.train()
-   model = model.to(device)
+   training_model = poptorch.trainingModel(model,
+                                           options=opts,
+                                           optimizer=optimizer)
```
And especially for training, the model should return the loss along with original output. You can do this simply by adding the loss on outputs in forward path like below.
```diff
-   def forward(self, x):
+   def forward(self, x, labels=None)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
+       if labels is not None:
+           loss = self.criterion(output, labels)
+           return output, loss

    return x
```
But, as we normally use a model imported from 3rd party libraries, we will create a wrapper class so that we don't have to modify the code inside the imported libraries.
```diff
+   class ModelwithLoss(nn.Module):
+       def __init__(self, model, criterion):
+        super().__init__()
+        self.model = model
+        self.criterion = criterion
+
+       def forward(self, x, labels=None):
+           output = self.model(x)
+           if labels is not None:
+               loss = self.criterion(output, labels)
+               return output, loss
+           return output

    model = ClassificationModel()
    model.train()
+   model = ModelwithLoss(model, criterion)    
    training_model = poptorch.trainingModel(model,
                                            options=opts,
                                            optimizer=optimizer)
```
`poptorch.trainingModel` will automatically detect [torch loss](://pytorch.org/docs/stable/nn.html#loss-functions) instances as the loss. However, if you use some custom modified loss, you have to wrap it with `poptorch.identity_loss` to allow `poptorch.trainingModel` detect it.
```diff
    loss = self.criterion(output, labels)
-   return output, loss
+   return output, poptorch.identity_loss(loss, reduction='sum')
```

### Dataset
To leverage IPU specific optimizaiton techniques, you should use `poptorch.DataLoader` instead of `torch.utils.data.DataLoader`. Details of those techniques will be introduced in the later chapters.
```diff
    train_dataset = torchvision.datasets.FashionMNIST(
        "../datasets", transform=transform, download=True, train=True)
    test_dataset = torchvision.datasets.FashionMNIST(
        "../datasets", transform=transform, download=True, train=False)
-   train_dataloader = torch.utils.data.DataLoader(train_dataset,
+   train_dataloader = poptorch.DataLoader(opts,
                                           train_dataset,
                                           batch_size=16,
                                           shuffle=True)
```

### Optimizer
Similarly to dataset, you should use `poptorch.optim` for training.
```diff
-   optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
+   optimizer = poptorch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

### Training Loop
When `poptorch.trainingModel` is being called, it performs all the training processes internally. This is why we added the loss to the outputs. Also, you don't have to set device to load the inputs as we didn't with the model. So, the training loop becomes much simpler. 
```diff
for data, labels in bar:
-   data = data.to(device)
-   labels = labels.to(device)

-   optimizer.zero_grad()
-   output = model(data)
+   output, loss = training_model(data, labels)
-   loss = criterion(output, labels)
-   loss.backward()
-   optimizer.step()
```

## Running the code

Now, we can run the pytorch model on IPU. The completed code is available on `pytorch_ipu.py`.
```bash
(venv_pytorch)$ nohup python pytorch_ipu.py &
```
<details><summary>Output </summary><p>

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz
26422272it [00:15, 1758536.29it/s]                              
Extracting ../datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz to ../datasets/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz
29696it [00:00, 52913.35it/s]                           
Extracting ../datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ../datasets/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
4422656it [00:02, 1997843.16it/s]                             
Extracting ../datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ../datasets/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
6144it [00:00, 72184324.30it/s]         
Extracting ../datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ../datasets/FashionMNIST/raw

[Epoch 01]:   0%|          | 0/3750 [00:00<?, ?it/s2022-09-06T14:14:57.775829Z popart:devicex 175824.175824 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
2022-09-06T14:14:57.915304Z popart:devicex 175824.175824 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
Graph compilation: 100%|██████████| 100/100 [00:39<00:00]2022-09-06T14:15:37.677556Z popart:devicex 175824.175824 W: Specified directory not found. Creating "../cache" directory 
Graph compilation:  97%|█████████▋| 97/100 [00:39<00:01]
[Epoch 01]: 100%|██████████| 3750/3750 [00:51<00:00, 72.21it/s, Loss=1.12]  
[Epoch 02]: 100%|██████████| 3750/3750 [00:09<00:00, 376.43it/s, Loss=0.775]
[Epoch 03]: 100%|██████████| 3750/3750 [00:09<00:00, 375.37it/s, Loss=1.05] 
[Epoch 04]: 100%|██████████| 3750/3750 [00:10<00:00, 365.68it/s, Loss=0.897]
[Epoch 05]: 100%|██████████| 3750/3750 [00:10<00:00, 374.72it/s, Loss=1.02] 
Graph compilation:   0%|          | 0/100 [00:00<?]2022-09-06T14:16:31.380537Z popart:devicex 175824.175824 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
2022-09-06T14:16:31.414728Z popart:devicex 175824.175824 W: The `debug.retainDebugInformation` engine option was implicitly set to `true`. The default will change to `false` in a future release. Set it to `true` explicitly if you want to query debug information (for example, by calling `Session::getReport`).
Graph compilation: 100%|██████████| 100/100 [00:15<00:00]
Eval accuracy: 86.42%
```

</p></details>

You can see that graph is being compiled before actual training/inference begins. For the next time you run the model with same computational graph, you can cache the compiled executable. Then, it won't spend time to compile the graph again from the next time. You can use executable caching by running the program with an environment variable `POPTORCH_CACHE_DIR=./cache`.
```bash
(venv_pytorch)$ POPTORCH_CACHE_DIR=./cache nohup python pytorch_ipu.py &
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

For more details, please visit [IPU Pytorch User Guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html)