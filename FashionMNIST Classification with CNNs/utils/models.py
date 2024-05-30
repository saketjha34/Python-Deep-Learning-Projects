import torch.nn as nn

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