import torch
import torch.optim as optim
from utils.utils import load_image, plot_images_grid, image_to_tensor
from torchvision.utils import save_image
import torchvision.transforms as transforms
from utils.config import StyleTransferConfig
from torchvision.transforms import v2, transforms
from src.model import ImageStyler, ContentLoss, StyleLoss
from src.train import train_model
from src.plot_loss_curve import plot_loss_curve_grid, plot_loss_curves

StyleTransferConfig = StyleTransferConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


IMG_CHANNEL = StyleTransferConfig.IMG_CHANNEL
IMG_NAME = StyleTransferConfig.IMG_NAME
IMG_SIZE = StyleTransferConfig.IMG_SIZE
EPOCHS = StyleTransferConfig.EPOCHS
LEARNING_RATE = StyleTransferConfig.LEARNING_RATE
WEIGHTS = StyleTransferConfig.WEIGHTS
ORIGINAL_IMG_PATH = StyleTransferConfig.ORIGINAL_IMG_PATH
STYLE_IMG_PATH = StyleTransferConfig.STYLE_IMG_PATH


image_transforms = transforms.Compose([
        v2.Resize(size = (IMG_SIZE,IMG_SIZE)),
        v2.CenterCrop((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
    ])


if __name__ == "__main__":
    
    model = ImageStyler().to(device).eval()
    original_img = load_image(img_path=ORIGINAL_IMG_PATH , image_transforms=image_transforms)
    style_img = load_image(img_path=STYLE_IMG_PATH , image_transforms=image_transforms)
    generated_img = original_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=LEARNING_RATE)
    content_loss_fn = ContentLoss()
    style_loss_fn = StyleLoss()

    results = train_model(model = model,
                          generated_img = generated_img,
                          original_img = original_img,
                          style_img = style_img,
                          optimizer = optimizer,
                          content_loss_fn = content_loss_fn,
                          style_loss_fn = style_loss_fn,
                          weights = WEIGHTS,
                          save_image= save_image,
                          epochs = EPOCHS)
    
    plot_loss_curves(results=results)
    plot_loss_curve_grid(results=results)
    plot_images_grid(original_img_path=ORIGINAL_IMG_PATH,
                     generated_img_path=f"output/generated_{IMG_NAME}.png",
                     style_img_path=STYLE_IMG_PATH,
                     image_to_tensor=image_to_tensor,
                     image_transforms=image_transforms)
