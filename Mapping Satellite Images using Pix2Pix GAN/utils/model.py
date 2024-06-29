import torch.nn as nn
import torch
class DownConvLayer(nn.Module):
    def __init__(self , in_channels , out_channels):
        super(DownConvLayer, self).__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels , kernel_size=4 , stride = 2 , padding = 1 , padding_mode="reflect" , bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True)
        )
    def forward(self , x):
        return self.Layer(x)

class UpConvLayer(nn.Module):
    def __init__(self , in_channels , out_channels , use_dropout = False):
        super(UpConvLayer, self).__init__()
        self.use_dropout = use_dropout
        self.Layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels , kernel_size=4 , stride = 2 , padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.Dropout = nn.Dropout(p = 0.5)
    def forward(self , x):
        x = self.Layer(x)
        return self.Dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels = 3):
        super(Generator,self).__init__()

        self.DownConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2 , inplace=True),
        )

        self.DownConvLayer2 = DownConvLayer(in_channels=64 ,out_channels=128)
        self.DownConvLayer3 = DownConvLayer(in_channels=128 ,out_channels=256)
        self.DownConvLayer4 = DownConvLayer(in_channels=256 ,out_channels=512)
        self.DownConvLayer5 = DownConvLayer(in_channels=512 ,out_channels=512)
        self.DownConvLayer6 = DownConvLayer(in_channels=512 ,out_channels=512)
        self.DownConvLayer7 = DownConvLayer(in_channels=512 ,out_channels=512)

        self.BottleNeck = nn.Sequential(
            nn.Conv2d(in_channels=512 , out_channels= 512,kernel_size= 4, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )

        self.UpConvLayer1 = UpConvLayer(in_channels=512 , out_channels=512 ,use_dropout=True)
        self.UpConvLayer2 = UpConvLayer(in_channels=1024 , out_channels=512 ,use_dropout=True)
        self.UpConvLayer3 = UpConvLayer(in_channels=1024, out_channels=512 ,use_dropout=True)
        self.UpConvLayer4 = UpConvLayer(in_channels=1024 , out_channels=512 ,use_dropout=False)
        self.UpConvLayer5 = UpConvLayer(in_channels=1024 , out_channels=256 ,use_dropout=False)
        self.UpConvLayer6 = UpConvLayer(in_channels=512 , out_channels=128 ,use_dropout=False)
        self.UpConvLayer7 = UpConvLayer(in_channels=256 , out_channels=64 ,use_dropout=False)

        self.FinalConvLayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self,x):
        down1 = self.DownConvLayer1(x)
        down2 = self.DownConvLayer2(down1)
        down3 = self.DownConvLayer3(down2)
        down4 = self.DownConvLayer4(down3)
        down5 = self.DownConvLayer5(down4)
        down6 = self.DownConvLayer6(down5)
        down7 = self.DownConvLayer7(down6)
        bottleneck = self.BottleNeck(down7)
        up1 = self.UpConvLayer1(bottleneck)
        up2 = self.UpConvLayer2(torch.cat([up1 ,down7],dim=1))
        up3 = self.UpConvLayer3(torch.cat([up2 ,down6],dim=1))
        up4 = self.UpConvLayer4(torch.cat([up3 ,down5],dim=1))
        up5 = self.UpConvLayer5(torch.cat([up4 ,down4],dim=1))
        up6 = self.UpConvLayer6(torch.cat([up5 ,down3],dim=1))
        up7 = self.UpConvLayer7(torch.cat([up6 ,down2],dim=1))
        return self.FinalConvLayer(torch.cat([up7 ,down1],dim=1))
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvBlock,self).__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=stride,padding=1,padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True),
        )
    def forward(self, x ):
        return self.Layer(x)
    
class Discriminator(nn.Module):
    def __init__(self , in_channels = 3):
        super(Discriminator,self).__init__()

        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2 , out_channels=64 , kernel_size=4 , stride=2 , padding=1 , padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.ConvLayer2 = ConvBlock(in_channels=64 , out_channels=128)
        self.ConvLayer3 = ConvBlock(in_channels=128 , out_channels=256)
        self.ConvLayer4 = ConvBlock(in_channels=256 , out_channels=512 , stride=1)
        self.ConvLayer5 = nn.Sequential(
            nn.Conv2d(in_channels=512 , out_channels=1 , kernel_size=4 , stride=1 , padding=1 , padding_mode="reflect")
        )

    def forward(self, x, y):
        x = torch.cat([x ,y], dim=1)
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = self.ConvLayer5(x)
        return x