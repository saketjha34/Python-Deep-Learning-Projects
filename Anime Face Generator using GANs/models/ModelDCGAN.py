import os
import torch   
from torch import nn 
import opendatasets as od
from pathlib import Path
from torchvision import transforms,v2
from torch.utils.data import DataLoader
from utils.utils import save_image_grid
from torchvision.datasets import ImageFolder 
from utils.utils  import train_model , plot_loss_curve_grid
from utils.config import DeviceDataLoader , to_device, get_default_device 
from  utils.config import IMAGE_SIZE, IMG_CHANNELS, NUM_WORKERS, BATCH_SIZE , NOISE_DIM, LEARNING_RATE, EPOCHS

device = get_default_device()
device

dataset_url = "https://www.kaggle.com/splcher/animefacedataset"
od.download(dataset_url)
data_dir = '/kaggle/working/animefacedataset'
dataset_dir = '/kaggle/working/animefacedataset/'

data_transforms = {
    'train': transforms.Compose([
        v2.Resize(size = (IMAGE_SIZE,IMAGE_SIZE)),
        v2.ToTensor(),
        v2.CenterCrop(IMAGE_SIZE)
    
    ]), }

train_data = ImageFolder(root=dataset_dir,
                            transform=data_transforms['train'],
                            target_transform=None,
                            is_valid_file=None)

train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = True,
)
train_dataloader = DeviceDataLoader(train_dataloader, device)


class Discriminator(nn.Module):
    def __init__(self , in_channels):
        super(Discriminator,self).__init__()

        self.Network = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            self._create_block(in_channels=64 , out_channels=128 , kernel_size=4,stride=2, padding=1),
            self._create_block(in_channels=128 , out_channels=256 , kernel_size=4,stride=2, padding=1),
            self._create_block(in_channels=256 , out_channels=512 , kernel_size=4,stride=2, padding=1),
    
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Flatten(),
            nn.Sigmoid(),
        
        )

    def _create_block(self,in_channels, out_channels , kernel_size , padding ,stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,padding=padding, stride=stride , bias= False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self,x):
        return self.Network(x)
    
class Generator(nn.Module):
    def __init__(self , noise_channels , img_channels):
        super(Generator , self).__init__()

        self.Network = nn.Sequential(

            self._create_block(in_channels=noise_channels, out_channels=512, kernel_size=4, padding=0 , stride=1),
            self._create_block(in_channels=512, out_channels=256, kernel_size=4, padding=1 , stride=2),   
            self._create_block(in_channels=256, out_channels=128, kernel_size=4, padding=1 , stride=2),   
            self._create_block(in_channels=128, out_channels=64, kernel_size=4, padding=1 , stride=2), 

            nn.ConvTranspose2d(in_channels=64, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def _create_block(self,in_channels, out_channels , kernel_size , padding ,stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding,bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.Network(x)
    
generator = Generator(noise_channels = NOISE_DIM , img_channels = IMG_CHANNELS)
discriminator = Discriminator(in_channels = IMG_CHANNELS)
discriminator = to_device(discriminator, device)
generator = to_device(generator, device)

sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)

fixed_latent = torch.randn([BATCH_SIZE , NOISE_DIM , 1, 1] ,device = device)
save_image_grid(0, generator,fixed_latent)

loss_fn = nn.BCELoss()
opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_generator = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
fixed_latent = torch.randn([BATCH_SIZE , NOISE_DIM , 1, 1] ,device = device)

results = train_model(discriminator,
        generator=generator,
        train_dl=train_dataloader,
        epochs=EPOCHS,
        fixed_latent=fixed_latent,
        opt_generator = opt_generator,
        opt_discriminator=opt_discriminator,
        loss_fn = loss_fn,
        batch_size=BATCH_SIZE,
        noise_dim = NOISE_DIM,
        save_samples = save_image_grid,
        device = device,
        start_idx=1)

MODEL_PATH = Path("pytorch saved models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "AnimeFaceGANs.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=generator.state_dict(),f=MODEL_SAVE_PATH)

plot_loss_curve_grid(results=results)