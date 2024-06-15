import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import v2, transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMG_SIZE = 720
EPOCHS = 6000
LEARNING_RATE = 0.001
WEIGHTS = (20,0.1)
original_img_path = '/kaggle/input/stylingimages/person3.jpg'
style_img_Path = '/kaggle/input/stylingimages/style8.jpg'
image_transforms = transforms.Compose([
        v2.Resize(size = (IMG_SIZE,IMG_SIZE)),
        v2.CenterCrop((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
    ])


def load_image(img_path: str, 
               image_transforms: transforms.Compose) -> torch.Tensor:
    image = Image.open(img_path)
    return image_transforms(image).unsqueeze(0).to(device)

class ImageStyler(nn.Module):
    def __init__(self):
        super(ImageStyler ,self,).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained = True ).features[:29]

    def forward(self,x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


class ContentLoss(nn.Module):
    def __init__(self,):
        super(ContentLoss, self).__init__()
    
    def forward(self, input , target):
        loss = nn.functional.mse_loss(input, target)
        return loss


class StyleLoss(nn.Module):
    def __init__(self,):
        super(StyleLoss, self).__init__()

    def forward(self, input ,target_feature):
        G = self._gram_matrix(input)
        A = self._gram_matrix(target_feature).detach()
        loss = nn.functional.mse_loss(G, A)
        return loss
    
    def _gram_matrix(self,input):
        batch_size, num_feature_maps, height, width = input.size()
        features = input.view(batch_size * num_feature_maps, height * width)
        G = torch.mm(features, features.t())
        return G

def train_model(model : torch.nn.Module,
                generated_img : torch.Tensor,
                original_img : torch.Tensor,
                style_img : torch.Tensor,
                optimizer : torch.optim.Optimizer,
                content_loss_fn : torch.nn.Module,
                style_loss_fn : torch.nn.Module,
                content_weight : float,
                style_weight : float,
                epochs : int) -> dict[str, list[float]]:
    
    for epoch in tqdm(range(epochs)):

        generated_img_features = model(generated_img)
        original_img_features = model(original_img)
        style_img_features = model(style_img)

        content_loss = 0
        style_loss = 0
        total_loss = 0

        for gen_feature, orig_feature, style_feature in zip(generated_img_features,
                                                            original_img_features,
                                                            style_img_features):

            content_loss += content_loss_fn(gen_feature , orig_feature)
            style_loss += style_loss_fn(gen_feature , style_feature)

        total_loss += content_weight * content_loss + style_weight * style_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:

            print('------------------ EPOCH {} -------------------'.format(epoch))
            print('Content Loss: {:.6f}, Style Loss: {:.6f}'.format(content_loss, style_loss))
            print('Total Loss: {:.6f}'.format(total_loss))
            print()
            save_image(generated_img , 'generated.png')


model = ImageStyler().to(device).eval()
original_img = load_image(img_path=original_img_path , image_transforms=image_transforms)
style_img = load_image(img_path=style_img_Path , image_transforms=image_transforms)
generated_img = original_img.clone().requires_grad_(True)
optimizer = optim.Adam([generated_img], lr=LEARNING_RATE)
content_loss_fn = ContentLoss()
style_loss_fn = StyleLoss()


train_model(model = model ,
            generated_img = generated_img ,
            original_img = original_img ,
            style_img = style_img,
            optimizer = optimizer ,
            content_loss_fn = content_loss_fn ,
            style_loss_fn = style_loss_fn ,
            content_weight = WEIGHTS[0],
            style_weight = WEIGHTS[1],
            epochs = EPOCHS)