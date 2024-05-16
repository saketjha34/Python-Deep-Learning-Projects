
# Importing necessary libraries and modules for image processing and deep learning tasks
# !pip install torchsummary --q
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis

from PIL import Image  # For handling image files
import matplotlib  # For creating plots
import matplotlib.pyplot as plt  # MATLAB-like interface for plotting
import matplotlib.patches as mpatches  # For creating graphical shapes
import warnings

import torch  # PyTorch deep learning framework
from torch import nn  # Neural network module
from torch import optim  # Optimization module
import torch.optim as optim  # Optimization algorithms
import torch.nn.functional as F  # Functional module with activation and loss functions
from torchvision import transforms, models  # Torchvision for datasets, models, and transformations
from torch.autograd import Variable  # For automatic differentiation
from torchvision.datasets import ImageFolder  # For loading image datasets
import torch.utils.data as data  # Utilities for handling data
from torch.utils.data import DataLoader, TensorDataset  # For batch data loading and creating datasets from tensors
import os
from torchsummary import summary  # For printing a summary of PyTorch model architecture

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# !pip install opendatasets --q
import opendatasets as od
dataset_url = "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals"
od.download(dataset_url)

# data_dir = '/kaggle/input/animal-image-dataset-90-different-animals/animals/animals'
# class_names = os.listdir(data_dir)
# class_names[:5] , len(class_names)

"""## 2 . Data Augmentation and Image Folder
We can create training and validation datasets using the `ImageFolder` class from `torchvision`. In addition to the `ToTensor` transform, we'll also apply some other transforms to the images. There are a few important changes we'll make while creating PyTorch datasets for training and validation:

1. **Use test set for validation**: Instead of setting aside a fraction (e.g. 10%) of the data from the training set for validation, we'll simply use the test set as our validation set. This just gives a little more data to train with. In general, once you have picked the best model architecture & hypeparameters using a fixed validation set, it is a good idea to retrain the same model on the entire dataset just to give it a small final boost in performance.
2. **Channel-wise data normalization**: We will normalize the image tensors by subtracting the mean and dividing by the standard deviation across each channel. As a result, the mean of the data across each channel is 0, and standard deviation is 1. Normalizing the data prevents the values from any one channel from disproportionately affecting the losses and gradients while training, simply by having a higher or wider range of values that others.

<img src="https://i.imgur.com/LYxXBVg.png" width="360">


3. **Randomized data augmentations**: We will apply randomly chosen transformations while loading images from the training dataset. Specifically, we will pad each image by 4 pixels, and then take a random crop of size 32 x 32 pixels, and then flip the image horizontally with a 50% probability. Since the transformation will be applied randomly and dynamically each time a particular image is loaded, the model sees slightly different images in each epoch of training, which allows it generalize better.

![data-augmentation](https://imgaug.readthedocs.io/en/latest/_images/cropandpad_percent.jpg)
"""

average  = torch.Tensor([0,0,0])
standard_dev = torch.Tensor([0,0,0])
images = ImageFolder(data_dir, transforms.ToTensor())
for image in images:
    average += image[0].mean([1,2])
    standard_dev += image[0].std([1,2])
average / len(images) , standard_dev / len(images)

"""

> Mean , Standard Deviation -- [R,G,B]

(tensor([0.5087, 0.5006, 0.4405]), tensor([0.2128, 0.2084, 0.2120]))

"""

stats = ((0.5087, 0.5006, 0.4405),(0.2128, 0.2084, 0.2120))

"""## 2.1 Prepare DataLoader

Now we've got a dataset ready to go.

The next step is to prepare it with a [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) or `DataLoader` for short.

The `DataLoader` does what you think it might do.

It helps load data into a model.

For training and for inference.

It turns a large `Dataset` into a Python iterable of smaller chunks.

These smaller chunks are called **batches** or **mini-batches** and can be set by the `batch_size` parameter.

Why do this?

Because it's more computationally efficient.

In an ideal world you could do the forward pass and backward pass across all of your data at once.

But once you start using really large datasets, unless you've got infinite computing power, it's easier to break them up into batches.

It also gives your model more opportunities to improve.

With **mini-batches** (small portions of the data), gradient descent is performed more often per epoch (once per mini-batch rather than once per epoch).

What's a good batch size?

[32 is a good place to start](https://twitter.com/ylecun/status/989610208497360896?s=20&t=N96J_jotN--PYuJk2WcjMw) for a fair amount of problems.

But since this is a value you can set (a **hyperparameter**) you can try all different kinds of values, though generally powers of 2 are used most often (e.g. 32, 64, 128, 256, 512).

Let's create `DataLoader`'s for our training and test sets.
"""

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.v2.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(*stats,inplace = True),
        ]),

}


dataset = ImageFolder(
    root= data_dir,
    transform=data_transforms['train'],
)

img, label = dataset[0]
print(img.shape, label)
img

"""### 2.2 Visualizing our data"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

img, label = dataset[567]
print(img.shape)
plt.imshow(img.permute(1,2,0))
plt.title(f'{dataset.classes[label]} : {label}')
plt.show()

random_seed = 42
torch.manual_seed(random_seed);
from torch.utils.data import random_split
val_size = 500
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

class_names = train_ds.dataset.classes
class_names[:10]

rows, cols = 5,5
def plot_images(rows,cols):
    # torch.manual_seed(42)
    fig = plt.figure(figsize=(12, 12))
    for i in range(1,rows*cols+1):
        random_idx = torch.randint(0 , len(train_ds), size = [1]).item()
        image, target = train_ds[random_idx]
        fig.add_subplot(rows,cols,i)
        plt.title(f'{class_names[target]} : {target}' )
        plt.axis(False);
        warnings.filterwarnings("ignore")
        plt.imshow(image.permute(1,2,0), cmap='gray')
plot_images(rows=rows,cols = cols)

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_dl = DataLoader(
    train_ds,
    shuffle = True,
    batch_size=BATCH_SIZE,

)

val_dl = DataLoader(
    val_ds,
    batch_size = BATCH_SIZE ,
    shuffle = False,

)

for images, labels in train_dl:
    print('images.shape:', images.shape)
    print(labels)
    break

"""## 3. Model : Building a Model | Transfer Learning | ResNet50
The second approach is to use an already pre-trained model. There are a lot of these models out there, so make sure to do a little research. How many layers to reuse and how many to retrain depends on the problem.

Keras, for example, provides numerous pre-trained models that can be used for transfer learning, prediction, feature extraction and fine-tuning. You can find these models, and also some brief tutorials on how to use them, here. There are also many research institutions that release trained models.

## ResNet-50 Architecture

The original ResNet architecture was ResNet-34, which comprised 34 weighted layers. It provided a novel way to add more convolutional layers to a CNN, without running into the vanishing gradient problem, using the concept of shortcut connections. A shortcut connection “skips over” some layers, converting a regular network to a residual network.

The regular network was based on the VGG neural networks (VGG-16 and VGG-19)—each convolutional network had a 3×3 filter. However, a ResNet has fewer filters and is less complex than a VGGNet. A 34-layer ResNet can achieve a performance of 3.6 billion FLOPs, and a smaller 18-layer ResNet can achieve 1.8 billion FLOPs, which is significantly faster than a VGG-19 Network with 19.6 billion FLOPs (read more in the ResNet paper, He et, al, 2015).

The ResNet architecture follows two basic design rules. First, the number of filters in each layer is the same depending on the size of the output feature map. Second, if the feature map’s size is halved, it has double the number of filters to maintain the time complexity of each layer.

## Special characteristics of ResNet-50

ResNet-50 has an architecture based on the model depicted above, but with one important difference. The 50-layer ResNet uses a bottleneck design for the building block. A bottleneck residual block uses 1×1 convolutions, known as a “bottleneck”, which reduces the number of parameters and matrix multiplications. This enables much faster training of each layer. It uses a stack of three layers rather than two layers.

The 50-layer ResNet architecture includes the following elements, as shown in the table below:

* A 7×7 kernel convolution alongside 64 other kernels with a 2-sized stride.
* A max pooling layer with a 2-sized stride.
* 9 more layers—3×3,64 kernel convolution, another with 1×1,64 kernels, and a third with 1×1,256 kernels. These 3 layers are repeated 3 times.
* 12 more layers with 1×1,128 kernels, 3×3,128 kernels, and 1×1,512 kernels, iterated 4 times.
* 18 more layers with 1×1,256 cores, and 2 cores 3×3,256 and 1×1,1024, iterated 6 times.
* 9 more layers with 1×1,512 cores, 3×3,512 cores, and 1×1,2048 cores iterated 3 times.
(up to this point the network has 50 layers)

* Average pooling, followed by a fully connected layer with 1000 nodes, using the softmax activation function.

<img src = 'https://miro.medium.com/v2/resize:fit:900/1*-JY0Nx8HXlNyJNijEyCWUg.jpeg'>
"""

# !pip install timm --q
import timm
from pprint import pprint
model_names = timm.list_models()
pprint(model_names[:10])

model = timm.create_model('resnet50',
                          pretrained=True,
                          num_classes = len(class_names)).to(device)

# Print the summary of the model
summary(model, (3, 224, 224))

"""### 3.1 Setup loss, optimizer and evaluation metrics


> **Note:** Rather than importing and using our own accuracy function or evaluation metric(s), you could import various evaluation metrics from the [TorchMetrics package](https://torchmetrics.readthedocs.io/en/latest/).
"""

# Define the loss function (criterion)
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""### 4 Functionizing training and test loops

So far we've been writing train and test loops over and over.

Let's write them again but this time we'll put them in functions so they can be called again and again.

And because we're using device-agnostic code now, we'll be sure to call `.to(device)` on our feature (`X`) and target (`y`) tensors.

For the training loop we'll create a function called `train_step()` which takes in a model, a `DataLoader` a loss function and an optimizer.

The testing loop will be similar but it'll be called `test_step()` and it'll take in a model, a `DataLoader`, a loss function and an evaluation function.

"""

def train(model, train_loader, optimizer):
    # Set the model to training mode
    model.train()

    # Iterate through batches of data in the train_loader
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target tensors to the appropriate device (GPU if available, otherwise CPU)
        data, target = data.to(device), target.to(device)

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # Calculate the batch loss
        loss =  criterion(output, target)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

def evaluate(model, test_loader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for test loss and correct predictions
    test_loss = 0
    correct = 0

    # Turn off gradients during evaluation
    with torch.no_grad():
        # Iterate through batches of data in the test_loader
        for data, target in test_loader:
            # Move data and target tensors to the appropriate device (GPU if available, otherwise CPU)
            data, target = data.to(device), target.to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # Compute the total test loss
            test_loss += criterion(output, target).item()

            # Get the index of the highest probability prediction
            pred = output.max(1, keepdim=True)[1]

            # Count the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate the average test loss
    test_loss /= len(test_loader.dataset)

    # Calculate the test accuracy
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

"""### 5 Creating a training loop and training a model on batches of data

Beautiful!

Looks like we've got all of the pieces of the puzzle ready to go, a timer, a loss function, an optimizer, a model and most importantly, some data.

Let's now create a training loop and a testing loop to train and evaluate our model.

We'll be using the same steps as the previous notebook(s), though since our data is now in batch form, we'll add another loop to loop through our data batches.

Our data batches are contained within our `DataLoader`s, `train_dataloader` and `test_dataloader` for the training and test data splits respectively.

A batch is `BATCH_SIZE` samples of `X` (features) and `y` (labels), since we're using `BATCH_SIZE=32`, our batches have 32 samples of images and targets.

And since we're computing on batches of data, our loss and evaluation metrics will be calculated **per batch** rather than across the whole dataset.

This means we'll have to divide our loss and accuracy values by the number of batches in each dataset's respective dataloader.

Let's step through it:
1. Loop through epochs.
2. Loop through training batches, perform training steps, calculate the train loss *per batch*.
3. Loop through testing batches, perform testing steps, calculate the test loss *per batch*.
4. Print out what's happening.
5. Time it all (for fun).

A fair few steps but...

...if in doubt, code it out.
"""

import time  # Importing time module for time tracking
import copy  # Importing copy module for deep copying model weights

def train_model(model, train_loader, val_loader, optimizer, num_epochs=30):
    # Lists to store training and validation accuracies and losses for each epoch
    acc_t = []  # Training accuracy
    acc_v = []  # Validation accuracy
    loss_t = []  # Training loss
    loss_v = []  # Validation loss

    best_acc = 0.0  # Best validation accuracy initialized to 0.0
    best_model_wts = copy.deepcopy(model.state_dict())  # Deep copy of model weights as the best model

    # Loop through each epoch
    for epoch in range(1, num_epochs + 1):
        since = time.time()  # Record the start time of the epoch

        # Train the model for one epoch
        train(model, train_loader, optimizer)

        # Evaluate the model on the training and validation sets to get loss and accuracy
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        # Update the best validation accuracy and best model weights if the current validation accuracy is higher
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # Append current epoch's accuracy and loss values to the lists
        acc_t.append(train_acc)
        acc_v.append(val_acc)
        loss_t.append(train_loss)
        loss_v.append(val_loss)

        time_elapsed = time.time() - since  # Calculate time elapsed for the epoch

        # Print epoch statistics
        print('-------------- EPOCH {} ----------------'.format(epoch))
        print('Train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('Val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    # Plot accuracy graph
    plt.plot(range(len(acc_t)), acc_t, 'b', range(len(acc_v)), acc_v, 'r')
    blue_patch = mpatches.Patch(color='blue', label='Train Accuracy')
    red_patch = mpatches.Patch(color='red', label='Validation Accuracy')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    # Plot loss graph
    plt.plot(range(len(loss_t)), loss_t, 'b', range(len(loss_v)), loss_v, 'r')
    blue_patch = mpatches.Patch(color='blue', label='Train Loss')
    red_patch = mpatches.Patch(color='red', label='Validation Loss')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # Return the model with the best weights
    return model

# Defining the number of epochs for training
EPOCH = 15

# Calling the train_model function to train the model
# The function takes the current model, training data loader (train_loader),
# validation data loader (val_loader), optimizer, and number of epochs as inputs
model = train_model(model, train_dl, val_dl, optimizer, EPOCH)
# The trained model with the best validation accuracy is returned and assigned to the variable 'model'

"""## 6. Evaluate Model"""

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the training data to get the training accuracy
# The evaluate function returns both loss and accuracy, but we're only interested in accuracy here,
# hence using '_' to ignore the loss value
_, train_acc = evaluate(model, train_dl)

# Evaluate the model on the validation data to get the validation accuracy
_, val_acc = evaluate(model, val_dl)

# Print the saved model's training accuracy
print('Train Accuracy: {:.4f}'.format(train_acc))

# Print the saved model's validation accuracy
print('Validation Accuracy: {:.4f}'.format(val_acc))

"""## 7. Making a confusion matrix for further prediction evaluation

There are many [different evaluation metrics](https://www.learnpytorch.io/02_pytorch_classification/#9-more-classification-evaluation-metrics) we can use for classification problems.

One of the most visual is a [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).

A confusion matrix shows you where your classification model got confused between predicitons and true labels.

To make a confusion matrix, we'll go through three steps:
1. Make predictions with our trained model, `model_2` (a confusion matrix compares predictions to true labels).
2. Make a confusion matrix using [`torchmetrics.ConfusionMatrix`](https://torchmetrics.readthedocs.io/en/latest/references/modules.html?highlight=confusion#confusionmatrix).
3. Plot the confusion matrix using [`mlxtend.plotting.plot_confusion_matrix()`](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/).

Let's start by making predictions with our trained model.
"""

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

import random
random.seed(69)
test_samples = []
test_labels = []
for sample, label in random.sample(list(val_ds), k=16):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs= make_predictions(model=model,
                             data=test_samples)

pred_classes = pred_probs.argmax(dim=1)
pred_classes

test_labels, pred_classes

# Plot predictions
plt.figure(figsize=(15, 15))
nrows = 4
ncols = 4
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.permute(1,2,0), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]]

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"

  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);

def get_preds(model , dataloader , device = device):
    from tqdm.auto import tqdm
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):

            X, y = X.to(device), y.to(device)

            y_logit = model(X)

            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor
val_preds = get_preds(model , val_dl)
val_preds[:10]

val_targets = [label for image , label  in val_ds]

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(val_targets, val_preds)
print(f'val accuarcy : {accuracy*100}%')

"""## 8. Classification Report"""

from sklearn.metrics import classification_report
report = classification_report(val_targets, val_preds)
print(report)

"""## 9. Save and load best performing model

Let's finish this section off by saving and loading in our best performing model.

Recall from [notebook 01](https://www.learnpytorch.io/01_pytorch_workflow/#5-saving-and-loading-a-pytorch-model) we can save and load a PyTorch model using a combination of:
* `torch.save` - a function to save a whole PyTorch model or a model's `state_dict()`.
* `torch.load` - a function to load in a saved PyTorch object.
* `torch.nn.Module.load_state_dict()` - a function to load a saved `state_dict()` into an existing model instance.

You can see more of these three in the [PyTorch saving and loading models documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

For now, let's save our `model_2`'s `state_dict()` then load it back in and evaluate it to make sure the save and load went correctly.
"""

from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("pytorch_saved_model")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "AnimalImageClassificationResnet50.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)

loaded_model = model

# Load in the saved state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model = loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()

# Evaluate the model on the training data to get the training accuracy
# The evaluate function returns both loss and accuracy, but we're only interested in accuracy here,
# hence using '_' to ignore the loss value
_, train_acc = evaluate(loaded_model, train_dl)

# Evaluate the model on the validation data to get the validation accuracy
_, val_acc = evaluate(loaded_model, val_dl)

# Print the saved model's training accuracy
print('Train Accuracy: {:.4f}'.format(train_acc))

# Print the saved model's validation accuracy
print('Validation Accuracy: {:.4f}'.format(val_acc))

