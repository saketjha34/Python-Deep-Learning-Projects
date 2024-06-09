import torch.nn as nn

# input -> batch_size x 3 x 64 x 64 , output -> batch_size x 1 x 1 x 1
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
    
# input -> batch_size x fixed_noise x 1 x 1 , output -> batch_size x 3 x 64 x 64
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



# <-------------------------------------------------------------------------------------------------------------------------------------->

# input -> batch_size x 3 x 256 x 256 , output -> batch_size x 1 x 1 x 1
class Discriminator(nn.Module):
    def __init__(self , in_channels):
        super(Discriminator,self).__init__()

        self.Network = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            self._create_block(in_channels=64 , out_channels=128 , kernel_size=4,stride=2, padding=1),
            self._create_block(in_channels=128 , out_channels=256 , kernel_size=4,stride=2, padding=1),
            self._create_block(in_channels=256 , out_channels=512 , kernel_size=4,stride=2, padding=1),
            self._create_block(in_channels=512 , out_channels=1024 , kernel_size=4,stride=2, padding=1),
    
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1,1)),
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

# input -> batch_size x fixed_noise x 1 x 1 , output -> batch_size x 3 x 256 x 256
class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels):
        super(Generator, self).__init__()

        self.Network = nn.Sequential(
            self._create_block(in_channels=noise_channels, out_channels=1024, kernel_size=4, padding=0, stride=1),  
            self._create_block(in_channels=1024, out_channels=512, kernel_size=4, padding=1, stride=2),  
            self._create_block(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2),  
            self._create_block(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2),  
            self._create_block(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2),  
            self._create_block(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2),  
            nn.ConvTranspose2d(in_channels=32, out_channels=img_channels, kernel_size=4, stride=2, padding=1),  
            nn.Conv2d(in_channels=img_channels, out_channels=img_channels, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(in_channels=img_channels, out_channels=img_channels, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(in_channels=img_channels, out_channels=img_channels, kernel_size=3, stride=1, padding=1),  
            nn.Tanh(),
        )

    def forward(self, x):
        return self.Network(x)

    def _create_block(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
