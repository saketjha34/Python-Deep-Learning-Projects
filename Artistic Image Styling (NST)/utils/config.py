import os
import torch
import torch.nn as nn
from torchvision.transforms import transforms , v2

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
EPOCHS = 4000
LEARNING_RATE = 1e-3
WEIGHTS = (2,0.01)
IMG_SIZE = 720
IMG_CHANNEL = 3
ORIGINAL_IMG_PATH = 'path/to/your/input/image.jpg'
STYLE_IMG_PATH = 'path/to/styling/image.jpg'
    
image_transforms = transforms.Compose([
        v2.Resize(size = (IMG_SIZE,IMG_SIZE)),
        v2.CenterCrop((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
    ])

