# Chapter III - Reduction with Horovod
In this chapter, you will find out how to perform reduction operation over multiple instances.

## Using Horovod for Inter-instance Reduction
Horovod can be used to perform reduction operations for tensors residing over a number of instances.
The core API that we are going to use in this tutorial is `horovod.torch.grouped_allreduce`.
Let's take a look at [hello_horovod.py](hello_horovod.py)


First, import `horovod.torch`. If poprun is invoked, initialize it. 
```python
import horovod.torch as hvd
import popdist
import torch

if __name__ == '__main__':
    if popdist.isPopdistEnvSet():
        hvd.init()
        ...
    else:
        print('Please run this example with poprun')
```

If poprun is invoked, create two tensors `x` and `y` as follows
```python
        x = torch.tensor([1, 0], dtype=torch.float32)
        y = torch.tensor([0, 1], dtype=torch.float16)
        
        instanceIndex = popdist.getInstanceIndex()
        x *= instanceIndex
        y *= instanceIndex
```

Then print their values before and after performing reduction operator with horovod over the instances.
Note that there're three reduction ops supported by horovod:
* Average
* Sum
* Adasum
```python
        print(f'Before grouped allreduce: x = {x}, y = {y}')
        x, y = <strong>hvd.grouped_allreduce([x, y], op=hvd.Average)</strong>
        print(f'Before grouped allreduce: x = {x}, y = {y}')
```

Let's execute it to see the result.

```bash
poprun --num-replicas 8 --num-instances 4 python hello_horovod.py
```

<details><summary>Output</summary>
<p>

``` 
[1,0]<stdout>:Before grouped allreduce: x = tensor([0., 0.]), y = tensor([0., 0.], dtype=torch.float16)
[1,3]<stdout>:Before grouped allreduce: x = tensor([3., 0.]), y = tensor([0., 3.], dtype=torch.float16)
[1,1]<stdout>:Before grouped allreduce: x = tensor([1., 0.]), y = tensor([0., 1.], dtype=torch.float16)
[1,2]<stdout>:Before grouped allreduce: x = tensor([2., 0.]), y = tensor([0., 2.], dtype=torch.float16)
[1,3]<stdout>:Before grouped allreduce: x = tensor([1.5000, 0.0000]), y = tensor([0.0000, 1.5000], dtype=torch.float16)
[1,2]<stdout>:Before grouped allreduce: x = tensor([1.5000, 0.0000]), y = tensor([0.0000, 1.5000], dtype=torch.float16)
[1,1]<stdout>:Before grouped allreduce: x = tensor([1.5000, 0.0000]), y = tensor([0.0000, 1.5000], dtype=torch.float16)
[1,0]<stdout>:Before grouped allreduce: x = tensor([1.5000, 0.0000]), y = tensor([0.0000, 1.5000], dtype=torch.float16)
```
</p>
</details>

You can see that horovod successfully performed reduction op (Average in this case) over the 4 instances.

[continue to the next chapter](../4.%20Distributed%20Data%20Loading)
