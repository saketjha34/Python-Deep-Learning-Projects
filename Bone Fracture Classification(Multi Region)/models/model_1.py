import numpy as np 
import pandas as pd  
from PIL import Image  
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches  
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
from torchinfo import summary  
import opendatasets as od
import torchvision.transforms as transforms  
import os 
from PIL import Image  

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_url = "https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data"
od.download(dataset_url)

data_dir = '/fracture-multi-region-x-ray-data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/'

train_dir = data_dir + 'train'
test_dir = data_dir + 'val'
val_dir = data_dir + 'test'

class_names = os.listdir(train_dir)
class_names


BATCH_SIZE = 32

def identify_and_remove_corrupted_images(dataset_dir):
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

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),                          
        transforms.CenterCrop(400),                      
        transforms.RandomHorizontalFlip(),               
        transforms.Grayscale(num_output_channels=1)]),  

    'val': transforms.Compose([
        transforms.ToTensor(),                          
        transforms.CenterCrop(400),                      
        transforms.Grayscale(num_output_channels=1)])   
}


train_dataset = ImageFolder(root=train_dataset_dir,
                            transform=data_transforms['train'],
                            target_transform=None,
                            is_valid_file=None)
val_dataset = ImageFolder(root=val_dataset_dir,
                          transform=data_transforms['val'],
                          target_transform=None,
                          is_valid_file=None)


train_dataset.samples = [(img, target) for img, target in train_dataset.samples if img not in corrupted_images_train]
val_dataset.samples = [(img, target) for img, target in val_dataset.samples if img not in corrupted_images_val]

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)



class AlexNet(nn.Module):
    def __init__(self, n_classes = 2):
        super(AlexNet, self).__init__()


        self.Conv_1 = nn.Sequential(
          nn.Conv2d(in_channels = 1, out_channels = 96, kernel_size = 11, stride = 4, padding = 0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 3, stride = 2),
          nn.BatchNorm2d(96))

        self.Conv_2 = nn.Sequential(
          nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 3, stride = 2),
          nn.BatchNorm2d(256))

        self.Conv_3 = nn.Sequential(
          nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
          nn.ReLU())

        self.Conv_4 = nn.Sequential(
          nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
          nn.ReLU())

        # 5th conv layer
        self.Conv_5 = nn.Sequential(
          nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 3, stride = 2))

        # 1st fully connected layer
        self.FC1 = nn.Sequential(
          nn.Flatten(),
          nn.Dropout(0.5),
          nn.Linear(256*11*11, 4096),
          nn.ReLU())

        # 2nd fully connected layer
        self.FC2 = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(4096, 4096),
          nn.ReLU())

        # 3rd fully connected layer --> output layer
        self.FC3 = nn.Sequential(
          nn.Linear(4096, n_classes))

    def forward(self, x):   # AlexNet forward propagation function
        # Propagate input through the layers of the network
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Conv_3(x)
        x = self.Conv_4(x)
        x = self.Conv_5(x)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)

        return F.log_softmax(x)

# Instantiate the AlexNet model and move it to the appropriate device (GPU if available, otherwise CPU)
model = AlexNet().to(DEVICE)

# Print the architecture of the model
model

# Print the summary of the model
summary(model, (1, 400, 400))


# Define the loss function (criterion)
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)



def train(model, train_loader, optimizer):
    # Set the model to training mode
    model.train()

    # Iterate through batches of data in the train_loader
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data and target tensors to the appropriate device (GPU if available, otherwise CPU)
        data, target = data.to(DEVICE), target.to(DEVICE)

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
            data, target = data.to(DEVICE), target.to(DEVICE)

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
model = train_model(model, train_loader, val_loader, optimizer, EPOCH)
# The trained model with the best validation accuracy is returned and assigned to the variable 'model'

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the training data to get the training accuracy
# The evaluate function returns both loss and accuracy, but we're only interested in accuracy here,
# hence using '_' to ignore the loss value
_, train_acc = evaluate(model, train_loader)

# Evaluate the model on the validation data to get the validation accuracy
_, val_acc = evaluate(model, val_loader)

# Print the saved model's training accuracy
print('Train Accuracy: {:.4f}'.format(train_acc))

# Print the saved model's validation accuracy
print('Validation Accuracy: {:.4f}'.format(val_acc))


from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("pytorch_saved_model")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "BoneFracture.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)

loaded_model = AlexNet(2)

# Load in the saved state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model = loaded_model.to(DEVICE)

# Set the model to evaluation mode
loaded_model.eval()

# Evaluate the model on the training data to get the training accuracy
# The evaluate function returns both loss and accuracy, but we're only interested in accuracy here,
# hence using '_' to ignore the loss value
_, train_acc = evaluate(loaded_model, train_loader)

# Evaluate the model on the validation data to get the validation accuracy
_, val_acc = evaluate(loaded_model, val_loader)

# Print the saved model's training accuracy
print('Train Accuracy: {:.4f}'.format(train_acc))

# Print the saved model's validation accuracy
print('Validation Accuracy: {:.4f}'.format(val_acc))

# Importing the necessary function from scikit-learn library
from sklearn.metrics import classification_report

# Function for making predictions and generating classification report
def prediction(model, data_loader):
    # Set the model to evaluation mode
    model.eval()

    # Initialize empty tensors to store predictions and labels
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    # Turn off gradient calculation for inference
    with torch.no_grad():
        # Iterate over the batches in the data loader
        for i, (data, label) in enumerate(data_loader):
            # Move data and label tensors to the appropriate device (GPU if available, otherwise CPU)
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            # Forward pass to get model predictions
            outputs = model(data)

            # Get the predicted class labels
            _, preds = torch.max(outputs, 1)

            # Append batch predictions and labels to the predlist and lbllist tensors
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, label.view(-1).cpu()])

    # Generate and print the classification report using scikit-learn's classification_report function
    print(classification_report(lbllist.numpy(), predlist.numpy()))

    # Return None as there is no specific output returned by this function
    return

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = DEVICE):
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
for sample, label in random.sample(list(val_dataset), k=16):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs= make_predictions(model=loaded_model,
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
  plt.imshow(sample.squeeze(), cmap="gray")

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




def get_preds(model , dataloader):
    from tqdm.auto import tqdm
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):

            X, y = X.to(DEVICE), y.to(DEVICE)

            y_logit = model(X)

            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor
val_preds = get_preds(loaded_model , val_loader)
val_preds[:10]


# Call the prediction function for making predictions and generating classification report
prediction(model, val_loader)

