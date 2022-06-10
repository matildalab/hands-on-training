# Chapter IV - Distributed Data Loading
In this chapter, you will find out how PopRun and PopDist interacts with PopTorch's data loading.

## Step 1 - PyTorch Data Loader

Let's start with a simple native PyTorch code [step1.py](step1.py) that reads CIFAR10 dataset and pass it to data loader with batch size 32.

```python
import torch
import torchvision

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../../datasets', download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)
    for (data, label) in dataloader:
        print(f'* dataset size: {len(dataset)}\n'
              f'* dataloader size: {len(dataloader)}\n'
              f'* batch size: {data.shape[0]}')
        break
```

Let's check the result first.
```bash
python step1.py
```

Output
```
Files already downloaded and verified
* dataset size: 50000
* dataloader size: 1562
* batch size: 32
```

Since the dataset size is 50,000 and we have set `drop_last=True` for `torch.utils.data.DataLoader`,
the dataloader size is 1,562 as expected. (`50,000 = 32 * 1,562 + 16`)

What happens if we run this script with PopRun?

```bash
poprun --num-replicas 8 --num-instances 4 python step1.py
```

<details><summary>Output</summary><p>

```
[1,0]<stdout>:Files already downloaded and verified
[1,3]<stdout>:Files already downloaded and verified
[1,2]<stdout>:Files already downloaded and verified
[1,1]<stdout>:Files already downloaded and verified
[1,3]<stdout>:* dataset size: 50000
[1,3]<stdout>:* dataloader size: 1562
[1,3]<stdout>:* batch size: 32
[1,0]<stdout>:* dataset size: 50000
[1,0]<stdout>:* dataloader size: 1562
[1,0]<stdout>:* batch size: 32
[1,1]<stdout>:* dataset size: 50000
[1,1]<stdout>:* dataloader size: 1562
[1,1]<stdout>:* batch size: 32
[1,2]<stdout>:* dataset size: 50000
[1,2]<stdout>:* dataloader size: 1562
[1,2]<stdout>:* batch size: 32
```

</p></details>

Each instance has dataloader that iterates over entire dataset.

However, if you want to implement distributed data parallelism,
you need to make sure that each instance should feed different
input data batches to its replicas.

## Step 2 - PopTorch Data Loader
`poptorch.DataLoader` will automatically do this for you.
Let's make a few lines of changes as follows. (See [step2.py](step2.py)).
```diff
-import torch
+import poptorch
import torchvision

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../../datasets', download=True,
                                           transform=torchvision.transforms.ToTensor())
-    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)
+    options = poptorch.Options()
+    dataloader = poptorch.DataLoader(options, dataset, batch_size=32, drop_last=True)
    for (data, label) in dataloader:
    print(f'* dataset size: {len(dataset)}\n'
          f'* dataloader size: {len(dataloader)}\n'
          f'* batch size: {data.shape[0]}')
    break
```

Now let's see what happens when `torch.utils.data.DataLoader` is replaced by `poptorch.DataLoader`.
```bash
poprun --num-replicas 8 --num-instances 4 python step2.py
```

<details><summary>Output</summary><p>

```
[1,3]<stdout>:Files already downloaded and verified
[1,1]<stdout>:Files already downloaded and verified
[1,2]<stdout>:Files already downloaded and verified
[1,0]<stdout>:Files already downloaded and verified
[1,3]<stdout>:* dataset size: 50000
[1,3]<stdout>:* dataloader size: 390
[1,3]<stdout>:* batch size: 32
[1,1]<stdout>:* dataset size: 50000
[1,1]<stdout>:* dataloader size: 390
[1,1]<stdout>:* batch size: 32
[1,2]<stdout>:* dataset size: 50000
[1,2]<stdout>:* dataloader size: 390
[1,2]<stdout>:* batch size: 32
[1,0]<stdout>:* dataset size: 50000
[1,0]<stdout>:* dataloader size: 390
[1,0]<stdout>:* batch size: 32
```

</p></details>

`poptorch.DataLoader` has automatically distributed the dataset over the 4 instances.
(Note that `1,562 = 4 * 390 + 2` so 2 data batches are ignored.)

Note that each instance is associated with 2 replicas since we have specified `--num-replicas 8 --num-instances 4`.
However, we don't have the notion of replicas in this context. The last missing puzzle here is `popdist.poptorch.Options`.

# Step 3
The ultimate distributed pipeline that we want to implement is described in the table below.

<table>
    <tbody align="center">
        <tr>
            <td rowspan=9>1562 input batches</td>
            <td rowspan=2>390 batches for instance 0</td>
            <td>195 batches for replica 0</td>
        </tr>
        <tr>
            <td>195 batches for replica 1</td>
        </tr>
        <tr>
            <td rowspan=2>390 batches for instance 1</td>
            <td>195 batches for replica 2</td>
        </tr>
        <tr>
            <td>195 batches for replica 3</td>
        </tr>
        <tr>
            <td rowspan=2>390 batches for instance 2</td>
            <td>195 batches for replica 4</td>
        </tr>
        <tr>
            <td>195 batches for replica 5</td>
        </tr>
        <tr>
            <td rowspan=2>390 batches for instance 3</td>
            <td>195 batches for replica 6</td>
        </tr>
        <tr>
            <td>195 batches for replica 7</td>
        </tr>
        <tr>
            <td colspan=2>2 superflous batches</td>
        </tr>
    </tbody>
</table>

In fact, all you need to do is to use `popdist.poptorch.Options`. (See [step3.py](step3.py)).
```diff
-import poptorch
+import popdist.poptorch
import torchvision

if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root='../../datasets', download=True,
                                           transform=torchvision.transforms.ToTensor())
-    options = poptorch.Options()
+    options = popdist.poptorch.Options()
    dataloader = poptorch.DataLoader(options, dataset, batch_size=32, drop_last=True)
    for (data, label) in dataloader:
-        print(f'* dataset size: {len(dataset)}\n'
+        print(f'* number of local replicas: {options.replication_factor}\n'
+              f'* dataset size: {len(dataset)}\n'
              f'* dataloader size: {len(dataloader)}\n'
              f'* batch size: {data.shape[0]}')
        break
```

Now let's see what happens when `poptorch.Options` is replaced by `popdist.poptorch.Options`.
```bash
poprun --num-replicas 8 --num-instances 4 python step3.py
```

<details><summary>Output</summary><p>

```
[1,1]<stdout>:Files already downloaded and verified
[1,3]<stdout>:Files already downloaded and verified
[1,2]<stdout>:Files already downloaded and verified
[1,0]<stdout>:Files already downloaded and verified
[1,1]<stdout>:* number of local replicas: 2
[1,1]<stdout>:* dataset size: 50000
[1,1]<stdout>:* dataloader size: 195
[1,1]<stdout>:* batch size: 64
[1,3]<stdout>:* number of local replicas: 2
[1,3]<stdout>:* dataset size: 50000
[1,3]<stdout>:* dataloader size: 195
[1,3]<stdout>:* batch size: 64
[1,2]<stdout>:* number of local replicas: 2
[1,2]<stdout>:* dataset size: 50000
[1,2]<stdout>:* dataloader size: 195
[1,2]<stdout>:* batch size: 64
[1,0]<stdout>:* number of local replicas: 2
[1,0]<stdout>:* dataset size: 50000
[1,0]<stdout>:* dataloader size: 195
[1,0]<stdout>:* batch size: 64
```

</p></details>

`popdist.poptorch.Options` can automatically detect distributed forwarded from `poprun`.
This is why `options.replication_factor` was already set to `2`.
In fact, if you try to set replication factor by yourself,
```python
import popdist.poptorch
options = popdist.poptorch.Options()
options.replicationFactor(2)
```
you will get an error `RuntimeError: Cannot call replicationFactor with popdist.poptorch.Options`.

You should always set replication factor via `poprun` whenever you use `popdist.poptorch.Options`.

[continue to the next chapter](../5.%20Distributed%20Data%20Parallelism)
