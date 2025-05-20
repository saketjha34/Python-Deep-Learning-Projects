import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm.notebook import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_image(img_path: str, 
               image_transforms: transforms.Compose) -> torch.Tensor:
    """
    Loads an image from the specified file path, applies the given transformations,
    and returns it as a tensor suitable for input to a PyTorch model.

    Args:
        img_path (str): The file path to the image to be loaded.
        image_transforms (transforms.Compose): A composition of transformations to be applied to the image.

    Returns:
        torch.Tensor: The transformed image as a tensor, with an added batch dimension, and moved to the specified device.

    Example:
        image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = load_image('path/to/image.jpg', image_transforms)
    """
    image = Image.open(img_path)
    return image_transforms(image).unsqueeze(0).to(device)


def save_generated_image(index: int,
                         generated_img: torch.Tensor,
                         show: bool = True) -> None:
    """
    Saves a generated image tensor to a specified directory with an indexed filename and optionally displays it.

    Args:
        index (int): The index number to be included in the generated image filename.
        generated_img (torch.Tensor): The generated image tensor to be saved.
        show (bool, optional): If True, displays the saved image using matplotlib. Defaults to True.

    Returns:
        None
    """
    sample_dir = 'Generated Images'
    os.makedirs(sample_dir, exist_ok=True)
    gen_img_name = 'generated-image-{0:0=4d}.png'.format(index)
    save_image(generated_img, os.path.join(sample_dir, gen_img_name))
    if show:
        plt.imshow(generated_img.detach().cpu().squeeze(0).permute(1, 2, 0))
        plt.title(f'{gen_img_name}')
        plt.axis(False)
        plt.show()

def image_to_tensor(img_path: str, image_transforms: transforms.Compose) -> torch.Tensor:
    """
    Converts an image file to a PyTorch tensor after applying the specified transformations.

    Args:
        img_path (str): The file path to the image to be converted.
        image_transforms (transforms.Compose): A composition of image transformations to be applied.
            This should be a torchvision.transforms.Compose object containing various transformations
            such as resizing, cropping, normalization, etc.

    Returns:
        torch.Tensor: The transformed image as a PyTorch tensor.

    Example:
        from torchvision import transforms

        # Define the image transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert image to tensor
        tensor = image_to_tensor('path/to/image.jpg', transform)
    """
    image = Image.open(img_path)
    return image_transforms(image)


def images_to_video(image_folder : str, 
                    video_path : str,
                    target_duration : int = 10, 
                    target_fps : int = 60) -> None:
    """
    Convert a folder of images to a video slowed down to the target duration.
    
    Args:
    - image_folder (str): Path to the folder containing images.
    - video_path (str): Path where the output video will be saved.
    - target_duration (int): Desired duration of the output video in seconds.
    - target_fps (int): Frames per second for the video.
    
    Returns:
    - None
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  

    first_image_path = os.path.join(image_folder, images[0])
    first_image = Image.open(first_image_path)
    width, height = first_image.size

    total_frames_needed = target_fps * target_duration

    repeats_per_frame = total_frames_needed // len(images)
    if total_frames_needed % len(images) != 0:
        repeats_per_frame += 1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    video = cv2.VideoWriter(video_path, fourcc, target_fps, (width, height))

    transform = transforms.ToTensor()

    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        
        image_tensor = transform(image)

        image_np = image_tensor.permute(1, 2, 0).numpy() * 255
        image_np = image_np.astype('uint8')

        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        for _ in range(repeats_per_frame):
            video.write(image_np)


    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved at {video_path}")


def plot_images_grid(original_img_path: str,
                     generated_img_path: str,
                     style_img_path: str,
                     image_to_tensor : callable , 
                     image_transforms: transforms.Compose) -> None:
    """
    Plots the original, generated, and style images in a 1x3 grid structure.

    Args:
        original_img_path (str): The file path to the original image.
        generated_img_path (str): The file path to the generated image.
        style_img_path (str): The file path to the style image.
        image_transforms (transforms.Compose): A composition of image transformations to be applied.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    original_img_tensor = image_to_tensor(original_img_path, image_transforms)
    generated_img_tensor = image_to_tensor(generated_img_path, image_transforms)
    style_img_tensor = image_to_tensor(style_img_path, image_transforms)

    to_pil = ToPILImage()

    # Plot Original Image
    axs[0].imshow(to_pil(original_img_tensor))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Plot Style Image
    axs[1].imshow(to_pil(style_img_tensor))
    axs[1].set_title('Style Image')
    axs[1].axis('off')
    
        # Plot Generated Image
    axs[2].imshow(to_pil(generated_img_tensor))
    axs[2].set_title('Generated Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
