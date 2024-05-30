import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torch.utils.data import DataLoader
import opendatasets as od
import os
from pathlib import Path
from torchvision.transforms import v2
import random
import torch
import torch.nn as nn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_url = "https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data"
od.download(dataset_url)

data_dir = '/kaggle/working/fracture-multi-region-x-ray-data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/'
os.listdir(data_dir)

train_dir = data_dir + 'train'
test_dir = data_dir + 'test'
val_dir = data_dir + 'val'

class_names = os.listdir(train_dir)
class_names


def identify_and_remove_corrupted_images(dataset_dir:str)->list:
    corrupted_images = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            try:
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    img.load()
            except (IOError, OSError) as e:
                # print(f"Corrupted image detected: {image_path}")
                corrupted_images.append(image_path)

    return corrupted_images

train_dataset_dir = train_dir
val_dataset_dir = val_dir


corrupted_images_train = identify_and_remove_corrupted_images(train_dir)
corrupted_images_val = identify_and_remove_corrupted_images(val_dir)
corrupted_images_test = identify_and_remove_corrupted_images(test_dir)



data_transforms = {
    'train': transforms.Compose([
        v2.Resize(size = (224,224)),
        v2.ToTensor(),
        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=7.5),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        v2.Grayscale(),

    ]),

    'val': transforms.Compose([
        v2.Resize(size = (224,224)),
        v2.ToTensor(),
        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=7.5),
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        v2.Grayscale(),

    ]) ,

     'test': transforms.Compose([
        v2.Resize(size = (224,224)),
        v2.ToTensor(),
        v2.Grayscale(),
    ])

}

train_data = ImageFolder(root=train_dir,
                            transform=data_transforms['train'],
                            target_transform=None,
                            is_valid_file=None)

val_data = ImageFolder(root=val_dataset_dir,
                          transform=data_transforms['val'],
                          target_transform=None,
                          is_valid_file=None)

test_data = ImageFolder(root=test_dir,
                          transform=data_transforms['test'],
                          target_transform=None,
                          is_valid_file=None)


train_data.samples = [(img, target) for img, target in train_data.samples if img not in corrupted_images_train]
val_data.samples = [(img, target) for img, target in val_data.samples if img not in corrupted_images_val]
test_data.samples = [(img, target) for img, target in test_data.samples if img not in corrupted_images_test]


BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = True,
)

val_dataloader = DataLoader(
    dataset = val_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = False
)

test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = False
)



class ResNet18(nn.Module):
    def __init__(self , in_channels = 3,  num_classes = 10):
        super(ResNet18 , self).__init__()

        self.num_classes = num_classes

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=64 , kernel_size=(7 , 7) , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(2,2),
        )

        self.ResBlock1 = self._create_residualblock(in_channels=64 , out_channels=64 , stride=1 , padding=1)
        self.ResBlock2 = self._create_residualblock(in_channels=64 , out_channels=64 , stride=1 , padding=1)

        self.ConvBlock2 = self._create_basicblock(in_channels=64 , out_channels=128 , kernel=3 , padding=1 , stride=1)
        self.ResBlock3 = self._create_residualblock(in_channels=128 , out_channels=128 , stride=1 , padding=1)

        self.ConvBlock3 = self._create_basicblock(in_channels=128 , out_channels=256 , kernel=3 , padding=1 , stride=1)
        self.ResBlock4 = self._create_residualblock(in_channels=256 , out_channels=256 , stride=1 , padding=1)

        self.ConvBlock4 = self._create_basicblock(in_channels=256 , out_channels=512 , kernel=3 , padding=1 , stride=1)
        self.ResBlock5 = self._create_residualblock(in_channels=512 , out_channels=512 , stride=1 , padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=7 , stride=7)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512 , out_features=self.num_classes),
        )

    def forward(self, x):
        out = self.ConvBlock1(x)
        out = self.ResBlock1(out) + out
        out = self.ResBlock2(out) + out
        out = self.ConvBlock2(out)
        out = self.ResBlock3(out) + out
        out = self.ConvBlock3(out)
        out = self.ConvBlock4(out)
        out = self.ResBlock5(out) + out
        out = self.avgpool(out)
        out = self.FC(out)
        return out

    def _create_residualblock(self,in_channels , out_channels , stride , padding ):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=(3,3) , padding=padding , stride=stride , bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels , kernel_size=(3,3) , padding=padding , stride=stride , bias= False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True),
            )


    def _create_basicblock(self,in_channels , out_channels , kernel,  stride , padding ):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel , padding=stride , stride =padding , bias =False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2 , inplace=True),
                nn.MaxPool2d(kernel_size=2 , stride=2),
            )

model = ResNet18(in_channels=1,num_classes = len(class_names)).to(device)


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



results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

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

epoch = 25
learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model = train_model(model, train_dataloader, val_dataloader, optimizer, criterion ,epoch)


def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

plot_loss_curves(results)


model.eval()

_, train_acc = evaluate(model, train_dataloader , criterion)
_, val_acc = evaluate(model, val_dataloader , criterion)
_, test_acc = evaluate(model, test_dataloader, criterion)

print('Train Accuracy: {:.4f}%'.format(train_acc))
print('Validation Accuracy: {:.4f}%'.format(val_acc))
print('Test Accuracy: {:.4f}%'.format(test_acc))



MODEL_PATH = Path("pytorch_saved_model/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "BoneXRayFractureClassificationResNet18.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)

loaded_model = model
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model = loaded_model.to(device)

loaded_model.eval()

_, train_acc = evaluate(loaded_model, train_dataloader , criterion)
_, val_acc = evaluate(loaded_model, val_dataloader , criterion)
_, test_acc = evaluate(loaded_model, test_dataloader, criterion)

print('Train Accuracy: {:.4f}%'.format(train_acc))
print('Validation Accuracy: {:.4f}%'.format(val_acc))
print('Test Accuracy: {:.4f}%'.format(test_acc))