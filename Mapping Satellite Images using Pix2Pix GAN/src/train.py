import torch  
from pathlib import Path
from torch import nn 
from torch import optim  
import torch.optim as optim  
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from dataset import Pix2PixDataset
import opendatasets as od
from utils.config import get_default_device,to_device
from utils.config import DeviceDataLoader
from utils.config import IMG_SIZE,BATCH_SIZE,LEARNING_RATE,EPOCHS,L1_LAMBDA,NUM_WORKERS
from utils.model import Generator, Discriminator
from utils.utils import train_model

device = get_default_device()
device
image_transforms = transforms.Compose([
    v2.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
])

od.download("https://www.kaggle.com/datasets/skjha69/map-dataset")
data_dir = "map-dataset/maps/"
train_dir = data_dir + "train/"
val_dir = data_dir + "val/"

train_dataset = Pix2PixDataset(root_dir=train_dir, 
                               transform=image_transforms , 
                               split_size = 600)
val_dataset = Pix2PixDataset(root_dir=val_dir, 
                             transform=image_transforms , 
                             split_size = 600)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS,
)

val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

generator = Generator()
discriminator = Discriminator()
discriminator = to_device(discriminator, device)
generator = to_device(generator, device)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
opt_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
opt_generator = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
loss_fn = nn.BCEWithLogitsLoss()
l1_loss_fn = nn.L1Loss()

model_history = train_model(train_loader = train_loader, 
                            discriminator= discriminator , 
                            generator=generator , 
                            opt_discriminator =opt_discriminator ,
                            opt_generator=opt_generator ,
                            l1_loss_fn=l1_loss_fn , 
                            loss_fn= loss_fn , 
                            g_scaler=g_scaler,
                            d_scaler= d_scaler,
                            L1_LAMBDA = L1_LAMBDA,
                            epochs = EPOCHS,
                            device= device)

MODEL_PATH = Path("pytorch_saved_model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "MapGeneratorPix2Pix.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=generator.state_dict(), f=MODEL_SAVE_PATH)
print(f"Saving model to: {MODEL_SAVE_PATH}")