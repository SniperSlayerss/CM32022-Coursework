# Structure

We have created one Python file per task plus a few extra files for the dataset and the model (tasks D1-7). 
You need to implement each task in the correspondent task file. You can create additional files, but you must make
sure that they are on this folder and that your code runs fine (e.g. no circular imports or files outside 
this folder). Importantly, **please do not rename the files**.

Inside each task file you will find a `prepare_test()` function (please do not rename it). 
This is our interface to automatically test your code.
For tasks D1-D7 this entails creating an instance of your model(s), loading the weights that you provide from your 
training, and return the model with the loaded weights. This means that for tasks D1-D7 you need to save the 
weights of your trained model inside the `./models` folder and upload them together with your code. 
The expected model file names are specified in each task file. 

You can save your model's weights with this line of code:

```python
torch.save(model.state_dict(), './models/dx.pth')
```

The details about each `prepare_test()` are specified in each task file.
For example, for task D1 we have:

```python
def prepare_test():
    # TODO: Create an instance of your model here. Load the pre-trained weights and return your model.
    #  Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 100), where B is the batch size
    #  and 100 is the number of classes. The output of your model must be the prediction of your classifier,
    #  providing a score for each class, for each image in input
    f = new_backbone()
    model = MyClassifier(f, ...)

    # do not edit from here downwards
    weights_path = 'models/d1.pth'
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model
```

The test code is private, but we provide a `test_check.py` where you can test whether your model is correctly loaded,
takes the expected input and returns the expected output. For example, for task D1 we have:

```python

def d1_check():
    from task_d1 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 100)
    print('D1 check passed')

if __name__ == '__main__':
    d1_check()
```

### Tip

If you write your code in a modular way, each task will require only an incremental amount of new code.

### Which model weights should I submit?

You should track the performance of your model and submit the weights of the best performing epoch.
**The important thing is that you upload only one file of weights per task**, (or however many specified for each task),
**named as specified in each task file**.

# Data preparation

First you will need to write your dataset class to load the CIFAR100 dataset. Implement your class extending 
the `torch.utils.data.Dataset` class in the file `dataset.py`. 
Once you have implemented it, you can import your class in each task file to create the 
training and testing data loaders (using the `torch.utils.data.DataLoader` class)

[Here](https://www.cs.toronto.edu/~kriz/cifar.html) you can read more about the dataset. 
We have prepared smaller training and testing splits which you will find in the `./data` folder. 
You can load each pickle file with the following:

```python
import pickle
with open(pickle_path, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
```

The training set contains 10,000 images, while the test set contains 2,500 images.

Note that for task D4 you must implement the triplet sampling yourself and that you cannot use any 
external library for this.

## Important notes

- Arrays and tensors should have shape (C, H, W), where C is the number of channels, H is the height of the image and 
  W is the width.
    - Remember that `plt.imshow` requires images in the (H, W, C) shape
- Everything in the pickled dictionary is a byte, including keys (which are unencoded strings). Remember this when 
  you need to access the dictionary and print things.
- Images must be normalised between 0 and 1
- You might need to adjust the batch size of the data loader depending on the machine and the task. You can use 
  different data loaders for different tasks, reusing the same dataset object.
- You cannot use other splits from the original files, i.e. you must use the provided pickles.
    -  Your model will be evaluated with a held-out testing split

# Backbone model

We have prepared the backbone model for you, which is [MobileNetV3](https://arxiv.org/abs/1905.02244). 
This is implemented in the `backbone.py` file. To instantiate the backbone model in your task files, simply do:

```python
from backbone import new_backbone

f = new_backbone()
```

## Important notes

- You can only use this model from the provided code. You cannot use another implementation of the model or 
  other pre-trained weights.
- It might take a while to download the weights. Weights are cached automatically, so you will only download 
  the weights once.
- The model requires at least 2 samples during training.
- This is a tiny CNN designed for efficiency (it can run on consumer CPUs). We chose this model to make training fast
  but naturally the size of the model will bring some limitations. Keep this in mind when running your experiments 
  and writing your report.

# Task D4

This is the most difficult task, so here are a couple of tips:

- Triplet mining strategy makes a difference, but it can be slow if implemented with `for` loops. Use 
  [broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) to speed things up.
- We recommend a more sophisticated optimiser (e.g. AdamW) with a small-ish learning rate (e.g. 1e-4) 

# Task D5

For this task we prepared different train-test split, which you will find in `./data/zero_shot`.
These have the exact structure as the other splits, so you won't have to write a new dataset class.
You will find three different pickles for the support sets:

- `support_1.pkl`
- `support_5.pkl`
- `support_10.pkl`

Which correspond to sets with support size 1, 5 and 10, respectively. These pickles too are structured like
the others, so you can use the same dataset class for the support sets as well. 

# Running times

## On the lab machines

We tested our implementation on the lab machines, and for most tasks we could run a whole training/testing epoch in 
35 seconds with a small batch size 32.

## Colab

We recommend using [Colab](https://colab.google/) with GPU acceleration to speed computation massively.
