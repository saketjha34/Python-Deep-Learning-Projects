import torch.nn as nn

# Model No.1 ResNet18 image size : N x 1 x 224 x 224
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

# Model No.2 AlexNet -> image size : N x 1 x 224 x 224
class AlexNet(nn.Module):
    def __init__(self,in_channels = 3 ,num_classes = 1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=96 , kernel_size=(11,11) , padding=1 , stride=4, bias = False),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)  ,stride=2),
            nn.BatchNorm2d(96)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=96 , out_channels=256 , kernel_size=(7,7) , padding=3 , stride=1, bias = False),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3)  ,stride=2),
            nn.BatchNorm2d(256)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=256 , out_channels=384 , kernel_size=(5,5) , padding=2 , stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.BatchNorm2d(384)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=384 , out_channels= 384 , kernel_size=(3,3) , padding=1 , stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(in_channels=384 , out_channels=256 , kernel_size=(3,3) , padding=1 , stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3)  ,stride=2),
        )

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=256*6*6 , out_features=4096),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Dropout(0.4),
            nn.Linear(in_features=4096 , out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096 , out_features=self.num_classes)
        )

    def forward(self,x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.FC(x)
        return x