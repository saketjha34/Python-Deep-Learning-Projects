import torch
from PIL import Image
import os
import time  
import copy  
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def identify_and_remove_corrupted_images(dataset_dir:str) -> list:
    '''
    Takes in dataset directory and returns list of corrupted images in that directory
    '''
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

def plot_random_images(dataset:torch.utils.data.dataset.Dataset ,
                          class_names : list[str] = None,
                          n : int = 3,
                          seed : int = None) -> None:
    if seed:
        random.seed(seed)

    fig = plt.figure(figsize=(12, 12))
    for i in range(1,n*n+1):

        random_idx = torch.randint(0 , len(dataset), size = [1]).item()
        image, target = dataset[random_idx]
        fig.add_subplot(n,n,i)
        plt.imshow(image.permute(1,2,0))
        plt.axis(False);

        if class_names:
            title = f"class: {class_names[target]} : {target}"
        else:
            title = None
        plt.title(title)


def train(model : torch.nn.Module,
          train_loader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          criterion : torch.nn.Module,
          device :torch.device = device):

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


def plot_loss_curves(results: dict[str, list[float]]) -> None:
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

def make_predictions(model: torch.nn.Module,
                     data: list, 
                     device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            
            sample = torch.unsqueeze(sample, dim=0).to(device) 

            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) 
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

def generate_random_sampleset(dataset:torch.utils.data.dataset.Dataset,
                               k : int) -> list:

    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(dataset), k=k):
        test_samples.append(sample)
        test_labels.append(label)

    return test_samples , test_labels


def plot_predicted_images(samples : list ,
                          test_targets : torch.Tensor ,
                          test_preds : torch.Tensor, 
                          class_names : list,
                          rows : int = 4,
                          cols : int = 4):
    
    plt.figure(figsize=(15, 15))

    for i, sample in enumerate(samples):
      
      plt.subplot(rows, cols, i+1)
      plt.imshow(sample.permute(1,2,0), cmap="gray")

      pred_label = class_names[test_preds[i]]
      truth_label = class_names[test_targets[i]]

      title_text = f"Pred: {pred_label} | Truth: {truth_label}"

      if pred_label == truth_label:
          plt.title(title_text, fontsize=10, c="g") 
      else:
          plt.title(title_text, fontsize=10, c="r") 
      plt.axis(False);


def get_preds(  model :torch.nn.Module,
                dataloader : torch.utils.datasets.Dataloader,
                device: torch.device = device ):

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


def plot_confusion_matrix(title: str,
                          targets:torch.Tensor, 
                          preds : torch.Tensor) -> None:

    cf = confusion_matrix(targets, preds)

    plt.figure(figsize=(6,4))
    sns.heatmap(cf , annot=True , cmap = 'Blues')

    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'Confusion Matrix : {title}') 
    plt.show()


def plot_classification_report(title: str ,
                               targets:torch.Tensor, 
                               preds : torch.Tensor) -> None:
    
    print('------------------------------------------------------------')
    report = classification_report(targets, preds)
    print(f'Classification Report : {title}')
    print(report)
    print('------------------------------------------------------------')
    print()
                        

