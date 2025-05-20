from torchvision.transforms import transforms , v2

import yaml

class StyleTransferConfig:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.EPOCHS = cfg['EPOCHS']
        self.LEARNING_RATE = cfg['LEARNING_RATE']
        self.WEIGHTS = tuple(cfg['WEIGHTS'])
        self.IMG_SIZE = cfg['IMG_SIZE']
        self.IMG_CHANNEL = cfg['IMG_CHANNEL']
        self.IMG_NAME = cfg['IMG_NAME']
        self.ORIGINAL_IMG_PATH = cfg['ORIGINAL_IMG_PATH']
        self.STYLE_IMG_PATH = cfg['STYLE_IMG_PATH']

StyleTransferconfig = StyleTransferConfig()

image_transforms = transforms.Compose([
        v2.Resize(size = (StyleTransferconfig.IMG_SIZE,StyleTransferconfig.IMG_SIZE)),
        v2.CenterCrop((StyleTransferconfig.IMG_SIZE,StyleTransferconfig.IMG_SIZE)),
        transforms.ToTensor(),
    ])