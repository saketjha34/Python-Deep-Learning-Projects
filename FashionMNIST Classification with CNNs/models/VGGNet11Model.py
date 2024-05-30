import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import os
import time
from pathlib import Path
import copy
from torchvision import datasets
from torchvision.transforms import v2


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        v2.Grayscale(),
        ]),

    'test': transforms.Compose([
        transforms.ToTensor(),
        v2.Grayscale(),
        ]),
}

train_data = datasets.FashionMNIST(
    train = True,
    root = 'dataset/',
    download = True,
    transform = data_transforms['train'],
    target_transform=None,
)

test_data = datasets.FashionMNIST(
    root="dataset/",
    train=False,
    download=True,
    transform=data_transforms['test'],
    target_transform=None,
)

class_names = train_data.classes


NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 128
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = True,
)

test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = False
)


# Architecture : [64 , 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
class VGGNet11(nn.Module):
    def __init__(self, in_channels =3 , num_classes = 10):
        super(VGGNet11,self).__init__()

        self.ConvBlock1 = self._create_block(in_channels=in_channels , out_channels=64 )

        self.ConvBlock2 = self._create_block(in_channels=64 , out_channels=128, pool = True)

        self.ConvBlock3 = self._create_block(in_channels=128 , out_channels=256)
        self.ConvBlock4 = self._create_block(in_channels=256 , out_channels=256)

        self.ConvBlock5 = self._create_block(in_channels=256 , out_channels=512)
        self.ConvBlock6 = self._create_block(in_channels=512 , out_channels=512, pool=True)

        self.ConvBlock7 = self._create_block(in_channels=512 , out_channels=512)
        self.ConvBlock8 = self._create_block(in_channels=512 , out_channels=512)
        
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 , out_features=4096),
            nn.ReLU(inplace=True ),
            nn.Dropout(0.4),
            nn.Linear(4096 , 4096),
            nn.ReLU(inplace=True ),
            nn.Dropout(0.4),
            nn.Linear(4096 , num_classes)

        )

    def forward(self , x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.ConvBlock8(x)
        x = self.AvgPool(x)
        x = self.FC(x)
        return x

    def _create_block(self,in_channels , out_channels , pool = False):
        layer = []
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=(3,3) ,stride=1 , padding=1 , bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layer.append(block)

        if  pool == True :layer.append(nn.MaxPool2d(kernel_size=(2,2) , stride=2))
        return nn.Sequential(*layer)

model = VGGNet11(in_channels = 1 , num_classes = len(class_names)).to(device)


def train(model:torch.nn.Module,
          train_loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          criterion:torch.nn.Module,
          device = device):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss =  criterion(output, target)

        loss.backward()

        optimizer.step()


def evaluate(model:torch.nn.Module,
              test_loader:torch.utils.data.DataLoader,
              criterion:torch.nn.Module,
              device = device):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += criterion(output, target).item()

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


import time
import copy
results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
def train_model(model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                val_loader:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer,
                criterion:torch.nn.Module ,
                num_epochs:int,
                device:torch.device = device):

    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict()) 

    for epoch in range(1, num_epochs + 1):
        since = time.time()  

        train(model, train_loader, optimizer , criterion)

        train_loss, train_acc = evaluate(model, train_loader , criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(val_loss)
        results["test_acc"].append(val_acc)

        time_elapsed = time.time() - since  

        print('-------------- EPOCH {} ----------------'.format(epoch))
        print('Train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('Val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    model.load_state_dict(best_model_wts)

    return model


epoch = 40
learning_rate = 3e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate , betas = (0.5 , 0.999))

model = train_model(model, train_dataloader, test_dataloader, optimizer, criterion ,epoch)


def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    loss = results['train_loss']
    test_loss = results['test_loss']

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

plot_loss_curves(results)

model.eval()

_, train_acc = evaluate(model, train_dataloader , criterion)
_, test_acc = evaluate(model, test_dataloader, criterion)

print('Train Accuracy: {:.4f}%'.format(train_acc))
print('Test Accuracy: {:.4f}%'.format(test_acc))


MODEL_PATH = Path("pytorch saved model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "FashionMNIST.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)


loaded_model = model
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model = loaded_model.to(device)

loaded_model.eval()
_, train_acc = evaluate(loaded_model, train_dataloader , criterion)
_, test_acc = evaluate(loaded_model, test_dataloader, criterion)

print('Loaded Model Train Accuracy: {:.4f}%'.format(train_acc))
print('Loaded Model Test Accuracy: {:.4f}%'.format(test_acc))