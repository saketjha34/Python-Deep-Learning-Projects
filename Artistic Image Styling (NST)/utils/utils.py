import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.gridspec import GridSpec
from torchvision.utils import save_image
from tqdm.notebook import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_model(model : torch.nn.Module,
                generated_img : torch.Tensor,
                original_img : torch.Tensor,
                style_img : torch.Tensor,
                optimizer : torch.optim.Optimizer,
                content_loss_fn : torch.nn.Module,
                style_loss_fn : torch.nn.Module,
                weights : tuple[float , float],
                save_generated_image: callable,
                epochs : int) -> dict[str, list[float]]:
    """
    Trains a neural network model for style transfer, optimizing the generated image to match the style of a given style image while preserving the content of an original image.

    Parameters:
    model (torch.nn.Module): The neural network model used to extract features from images.
    generated_img (torch.Tensor): The image being generated and optimized.
    original_img (torch.Tensor): The original image whose content is to be preserved.
    style_img (torch.Tensor): The style image whose style is to be applied to the generated image.
    optimizer (torch.optim.Optimizer): The optimizer used to update the generated image.
    content_loss_fn (callable): The function used to compute the content loss between the generated image and the original image.
    weights (tuple[float, float]): A tuple containing two weights: the content weight and the style weight.
    style_loss_fn (callable): The function used to compute the style loss between the generated image and the style image.
    epochs (int): The number of epochs for training.

    Returns:
    dict[str, list[float]]: A dictionary containing lists of content loss, style loss, and total loss recorded at each epoch.
    """
    
    results = {
        'content_loss': [],
        'style_loss': [],
        'total_loss': [],
    }

    for epoch in tqdm(range(epochs)):

        content_weight = weights[0]
        style_weight = weights[1]

        generated_img_features = model(generated_img)
        original_img_features = model(original_img)
        style_img_features = model(style_img)

        content_loss = 0
        style_loss = 0
        total_loss = 0

        for gen_feature, orig_feature, style_feature in zip(generated_img_features, 
                                                            original_img_features, 
                                                            style_img_features):

            content_loss += content_loss_fn(gen_feature, orig_feature)
            style_loss += style_loss_fn(gen_feature, style_feature)

        total_loss += content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            results['content_loss'].append(content_loss)
            results['style_loss'].append(style_loss)
            results['total_loss'].append(total_loss)

            print('------------------ EPOCH {} -------------------'.format(epoch))
            print('Content Loss: {:.6f}, Style Loss: {:.6f}'.format(content_loss, style_loss))
            print('Total Loss: {:.6f}'.format(total_loss))
            print()

            save_generated_image(epoch+1, generated_img)

    return results

def plot_loss_curves(results: dict[str, list[float]]) -> None:
    """
    Plots the style loss and total loss curves over epochs based on the provided results dictionary.

    Args:
        results (dict[str, list[float]]): A dictionary containing the loss values.
            Expected keys are 'style_loss' and 'total_loss', with corresponding lists of loss values for each epoch.

    Returns:
        None

    Example:
        results = {
            'style_loss': [0.8, 0.6, 0.4, 0.2],
            'total_loss': [1.0, 0.7, 0.5, 0.3]
        }
        plot_loss_curves(results)
    """
    style_loss = results['style_loss']
    total_loss = results['total_loss']

    epochs = range(len(style_loss))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, style_loss, label='Style Loss')
    plt.plot(epochs, total_loss, label='Total Loss')
    plt.title('Loss: Style vs Total')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


def plot_loss_curve_grid(results: dict[str, list[float]]) -> None:
    """
    Plots the content loss, style loss, and total loss curves over epochs using a grid layout.

    Args:
        results (dict[str, list[float]]): A dictionary containing the loss values.
            Expected keys are 'content_loss', 'style_loss', and 'total_loss', with corresponding lists of loss values for each epoch.

    Returns:
        None

    Example:
        results = {
            'content_loss': [1.0, 0.8, 0.6, 0.4],
            'style_loss': [0.9, 0.7, 0.5, 0.3],
            'total_loss': [1.9, 1.5, 1.1, 0.7]
        }
        plot_loss_curve_grid(results)
    """
    content_loss = results['content_loss']
    style_loss = results['style_loss']
    total_loss = results['total_loss']

    epochs = range(len(content_loss))

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, content_loss, marker='o', color='b', label='Content Loss')
    ax1.set_title('Content Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, style_loss, marker='s', color='r', label='Style Loss')
    ax2.set_title('Style Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, total_loss, marker='o', color='g', label='Total Loss')
    ax3.set_title('Total Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


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
