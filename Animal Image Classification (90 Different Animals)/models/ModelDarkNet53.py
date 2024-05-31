# pip install opendatasets timm 
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches  
from pathlib import Path
import torch  
import time 
import timm
import copy  
from torch import nn 
from torch import optim  
import torch.optim as optim  
import torch.nn.functional as F 
from torchvision import transforms
from torchvision.datasets import ImageFolder  
import torch.utils.data 
from torch.utils.data import DataLoader
import os
from torchvision.transforms import v2
from torch.utils.data import random_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import opendatasets as od
dataset_url = "https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals"
od.download(dataset_url)

data_dir = '/animal-image-dataset-90-different-animals/animals/animals'
class_names = os.listdir(data_dir)
class_names[:5] , len(class_names)

stats = ((0.5087, 0.5006, 0.4405),(0.2128, 0.2084, 0.2120))

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        v2.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(*stats,inplace = True),
        ]),
}

dataset = ImageFolder(
    root= data_dir,
    transform=data_transforms['train'],
    target_transform=None
)


random_seed = 42
torch.manual_seed(random_seed);
val_size = 500
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
class_names = train_ds.dataset.classes


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


class DarkNet53(nn.Module):
    def __init__(self , num_classes = 1000):
        super(DarkNet53, self).__init__()
        self.model = timm.create_model('darknet53', pretrained=True, num_classes = num_classes)

    def forward(self,x):
        return self.model(x)
    
model = DarkNet53(num_classes=90).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001 , betas = (0.5 , 0.999))


def train(model, train_loader, optimizer):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss =  criterion(output, target)

        loss.backward()

        optimizer.step()


def evaluate(model, test_loader):
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


def train_model(model, train_loader, val_loader, optimizer, num_epochs=30):

    acc_t = []  
    acc_v = []  
    loss_t = []  
    loss_v = [] 

    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict())  

    for epoch in range(1, num_epochs + 1):
        since = time.time()  

        train(model, train_loader, optimizer)

        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        acc_t.append(train_acc)
        acc_v.append(val_acc)
        loss_t.append(train_loss)
        loss_v.append(val_loss)

        time_elapsed = time.time() - since 

        print('-------------- EPOCH {} ----------------'.format(epoch))
        print('Train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('Val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    plt.plot(range(len(acc_t)), acc_t, 'b', range(len(acc_v)), acc_v, 'r')
    blue_patch = mpatches.Patch(color='blue', label='Train Accuracy')
    red_patch = mpatches.Patch(color='red', label='Validation Accuracy')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    plt.plot(range(len(loss_t)), loss_t, 'b', range(len(loss_v)), loss_v, 'r')
    blue_patch = mpatches.Patch(color='blue', label='Train Loss')
    red_patch = mpatches.Patch(color='red', label='Validation Loss')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    model.load_state_dict(best_model_wts)
    return model


EPOCH = 15
model = train_model(model, train_dl, val_dl, optimizer, EPOCH)



model.eval()
_, train_acc = evaluate(model, train_dl)
_, val_acc = evaluate(model, val_dl)

print('Train Accuracy: {:.4f}'.format(train_acc))
print('Validation Accuracy: {:.4f}'.format(val_acc))




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
val_targets = [label for image , label  in val_ds]

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(val_targets, val_preds)
print(f'val accuarcy : {accuracy*100}%')


from sklearn.metrics import classification_report
report = classification_report(val_targets, val_preds)
print(report)

MODEL_PATH = Path("pytorch saved model")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True )

MODEL_NAME = "AnimalImageClassificationDarkNet53.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), 
           f=MODEL_SAVE_PATH)


loaded_model = model
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model = loaded_model.to(device)


loaded_model.eval()
_, train_acc = evaluate(loaded_model, train_dl)
_, val_acc = evaluate(loaded_model, val_dl)

print('Train Accuracy: {:.4f}'.format(train_acc))
print('Validation Accuracy: {:.4f}'.format(val_acc))