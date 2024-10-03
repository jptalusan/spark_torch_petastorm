# How to use dataloader and what it does
https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

## Requirements
Spark>=3.4.0
## Handling larger than ram datasets
1. spark: What is [parquet](https://www.databricks.com/glossary/what-is-parquet) and why is it better?
2. petastorm


## [Petastorm](https://www.uber.com/blog/petastorm/)
Its own datafiles, [see this page](https://www.hopsworks.ai/post/guide-to-file-formats-for-machine-learning). can use with pyspark

https://petastorm.readthedocs.io/en/latest/readme_include.html

```python
import torch
from petastorm.pytorch import DataLoader

torch.manual_seed(1)
device = torch.device('cpu')
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def _transform_row(mnist_row):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return (transform(mnist_row['image']), mnist_row['digit'])


transform = TransformSpec(_transform_row, removed_fields=['idx'])

with DataLoader(make_reader('file:///localpath/mnist/train', num_epochs=10,
                            transform_spec=transform, seed=1, shuffle_rows=True), batch_size=64) as train_loader:
    train(model, device, train_loader, 10, optimizer, 1)
with DataLoader(make_reader('file:///localpath/mnist/test', num_epochs=10,
                            transform_spec=transform), batch_size=1000) as test_loader:
    test(model, device, test_loader)
```

Here are some examples for PetaStorm and Torch. For more information, see the readme of the petastorm github.

* https://docs.databricks.com/en/archive/machine-learning/petastorm.html
* https://github.com/uber/petastorm/blob/master/examples/mnist/pytorch_example.py
    * Contrast and compare it to a [run of the mill torch example for mnist](https://github.com/pytorch/examples/blob/main/mnist/main.py).
* https://blog.munhou.com/2020/01/16/Data-Pipeline-From-Pyspark-to-Pytorch/
    > Simple example but might be deprecated
* https://www.databricks.com/notebooks/simple-aws/petastorm-spark-converter-pytorch.html
    > Another very good example. jsut need to find the dataset, its in databricks.

## Spark and Torch
[How to run?](https://www.databricks.com/blog/2023/04/20/pytorch-databricks-introducing-spark-pytorch-distributor.html)
* https://docs.databricks.com/en/_extras/notebooks/source/deep-learning/torch-distributor-notebook.html
Note torch distributor is for single machine, multi-gpu.


# References
* https://discuss.pytorch.org/t/pytorch-multi-worker-dataloader-runs-in-parallel-with-training-code/103485/2
    > The DataLoader will use multiprocessing to create multiple workers, which will load and process each data sample and add the batch to a queue. It should thus not be blocking the training as long as the queue is filled with batches.
* https://stackoverflow.com/questions/57836355/how-to-split-and-load-huge-dataset-that-doesnt-fit-into-memory-into-pytorch-dat/57843010#57843010
    > Yes, the default behavior for the ImageFolder is to create a list of image paths and load the actual images only when needed. 
* https://stackoverflow.com/questions/68199072/pytorch-dataloader-for-reading-a-large-parquet-csv-file
    > Reading spark parquets using dask.
* https://stackoverflow.com/questions/60685684/how-to-load-large-multi-file-parquet-files-for-tensorflow-pytorch
    > For petastorm, you are right to use make_batch_reader(). Indeed, the error messages are not always helpful; but you can inspect the stack trace and investigate where in petastorm code it originates from.