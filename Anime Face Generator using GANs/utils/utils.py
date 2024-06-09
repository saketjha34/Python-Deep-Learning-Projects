import os
import cv2
import torch
import random
from PIL import Image
import torch.nn as nn
import torch.functional as F
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
from torchvision.utils import make_grid , save_image

def plot_random_images(dataset:torch.utils.data.dataset.Dataset ,
                       class_names : list[str] = None,
                       n : int = 3,
                       seed : int = None) -> None:
    """
    Plots a grid of random images from a given dataset.

    Parameters:
    dataset (torch.utils.data.dataset.Dataset): The dataset from which to sample images.
    class_names (list[str], optional): A list of class names corresponding to the target labels.
                                       Defaults to None.
    n (int, optional): The number of images to plot along each axis (resulting in an n x n grid).
                       Defaults to 3.
    seed (int, optional): A seed for the random number generator to ensure reproducibility.
                          Defaults to None.

    Returns:
    None: This function does not return any value. It displays a grid of images.
    """
    if seed:
        random.seed(seed)

    fig = plt.figure(figsize=(12, 12))
    for i in range(1,n*n+1):
        random_idx = torch.randint(0 , len(dataset), size = [1]).item()
        image, target = dataset[random_idx]
        fig.add_subplot(n,n,i)
        plt.imshow(image.permute(1,2,0))
        plt.axis(False);
        if class_names:
            title = f"class: {class_names[target]} : {target}"
        else:
            title = None
        plt.title(title)
        

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


def make_image_grid(images:torch.Tensor, 
                    nmax=None) -> None:
    """
    Display a grid of images.

    Parameters:
    images (torch.Tensor): A batch of images of shape (batch_size, 3, 256, 256).
    nmax (int, optional): Maximum number of images to display. If None, display all images in the batch. Defaults to None.

    Returns:
    None: This function does not return any value. It displays a grid of images.
    """
    if nmax is not None:
        images = images[:nmax]
    
    # Create the grid of images
    grid_img = make_grid(images, nrow=8, padding=2, normalize=True)
    
    # Plot the grid
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(grid_img.permute(1, 2, 0))
    ax.set_xticks([]) 
    ax.set_yticks([])
    plt.show()


def show_batch(dataloader: torch.utils.data.DataLoader, 
               make_image_grid: callable, 
               nmax: int = 64) -> None:
    """
    Display a batch of images from a DataLoader using a specified function to create an image grid.

    Parameters:
    dataloader (torch.utils.data.DataLoader): The DataLoader to fetch the batch of images from.
    make_image_grid (callable): The function used to create and display the image grid.
    nmax (int, optional): The maximum number of images to display. Defaults to 64.

    Returns:
    None: This function does not return any value. It displays a batch of images using the provided image grid function.
    """
    for images, _ in dataloader:
        make_image_grid(images.cpu(), nmax)
        break


def save_image_grid(index : int,
                 generator:torch.nn.Module,
                 latent_tensors : torch.Tensor,
                 show=True) -> None:
    """
    Generate and save a grid of images using a generator model, and optionally display the grid.

    Parameters:
    index (int): The index used to name the saved image file.
    generator (torch.nn.Module): The generator model to produce images from latent tensors.
    latent_tensors (torch.Tensor) size = [batch_size,noise_dim,1,1]: The latent tensors used as input to the generator.
    show (bool, optional): Whether to display the generated image grid. Defaults to True.

    Returns:
    None: This function does not return any value. It saves and optionally displays a grid of generated images.
    """

    sample_dir = 'Generated Images'
    os.makedirs(sample_dir, exist_ok=True)
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

       

def train_discriminator(discriminator : torch.nn.Module, 
                        generator : torch.nn.Module, 
                        real_images : torch.Tensor, 
                        opt_discriminator :torch.optim.Optimizer ,
                        loss_fn :torch.nn.Module,
                        batch_size : int ,
                        noise_dim : int, 
                        device : torch.device) -> tuple[float, float , float]:
    """
    Train the discriminator model on both real and generated (fake) images.

    Parameters:
    discriminator (torch.nn.Module): The discriminator model.
    generator (torch.nn.Module): The generator model.
    real_images (torch.Tensor) , size = [batch_size,img_channels,img_size,img_size]: A batch of real images. -> Ex.[32,3,64,64]
    opt_discriminator (torch.optim.Optimizer): The optimizer for the discriminator.
    loss_fn (torch.nn.Module): The loss function used for training.
    batch_size (int): The size of the batch for training.
    noise_dim (int): The dimension of the noise tensor used to generate fake images.
    device (torch.device): The device (CPU or GPU) on which computations are performed.

    Returns:
    tuple[float, float, float]: A tuple containing the total loss, the average score for real images,
                                and the average score for fake images.
    """

    opt_discriminator.zero_grad()

    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = loss_fn(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    latent = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    fake_images = generator(latent)

    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = loss_fn(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    loss = real_loss + fake_loss
    loss.backward()
    opt_discriminator.step()
    return loss.item(), real_score, fake_score


def train_generator(generator : torch.nn.Module ,
                    discriminator : torch.nn.Module, 
                    opt_generator :torch.optim.Optimizer ,
                    loss_fn : torch.nn.Module, 
                    batch_size : int , 
                    noise_dim : int,
                    device : torch.device) -> float:
    """
    Train the generator model to produce images that can fool the discriminator.

    Parameters:
    generator (torch.nn.Module): The generator model.
    discriminator (torch.nn.Module): The discriminator model.
    opt_generator (torch.optim.Optimizer): The optimizer for the generator.
    loss_fn (torch.nn.Module): The loss function used for training.
    batch_size (int): The size of the batch for training.
    noise_dim (int): The dimension of the noise vector used to generate fake images.
    device (torch.device): The device (CPU or GPU) on which computations are performed.

    Returns:
    float: The loss value for the generator after the optimization step.
    """
    opt_generator.zero_grad()

    latent = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    fake_images = generator(latent)

    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = loss_fn(preds, targets)
 
    loss.backward()
    opt_generator.step()
    
    return loss.item()


def train_model(discriminator : nn.Module,
                generator : nn.Module,
                train_dl : torch.utils.data.DataLoader,
                fixed_latent : torch.Tensor,
                opt_generator : torch.optim.Optimizer,
                opt_discriminator : torch.optim.Optimizer ,
                loss_fn: nn.Module , 
                epochs : int, 
                batch_size  : int,
                noise_dim : int, 
                device : torch.device,
                save_samples : callable = None ,
                start_idx : int = 1) -> dict[str , list[float]]:
    """
    Train the GAN models (discriminator and generator) over multiple epochs.

    Parameters:
    discriminator (nn.Module): The discriminator model.
    generator (nn.Module): The generator model.
    train_dl (torch.utils.data.DataLoader): The DataLoader for the training dataset.
    fixed_latent (torch.Tensor): A fixed set of latent vectors for generating samples during training.
    opt_generator (torch.optim.Optimizer): The optimizer for the generator.
    opt_discriminator (torch.optim.Optimizer): The optimizer for the discriminator.
    loss_fn (nn.Module): The loss function used for training both generator and discriminator.
    epochs (int): The number of epochs to train for.
    batch_size (int): The batch size for training.
    noise_dim (int): The dimension of the noise vector for the generator.
    device (torch.device): The device to run the training on (CPU or GPU).
    save_samples (callable, optional): A function to save generated samples. Defaults to None.
    start_idx (int, optional): The starting index for saving generated images. Defaults to 1.

    Returns:
    dict[str, list[float]]: A dictionary containing the loss and scores for the generator and discriminator.
    """
    torch.cuda.empty_cache()
    
    results = {
        'loss_generator' : [],
        'loss_discriminator' : [],
        'real_scores' : [],
        'fake_scores' : [],
    }
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            real_images.to(device)
            loss_d, real_score, fake_score = train_discriminator(real_images=real_images,
                                                                 discriminator=discriminator,
                                                                 generator=generator,
                                                                 loss_fn=loss_fn,
                                                                 opt_discriminator=opt_discriminator,
                                                                 batch_size=batch_size,
                                                                 noise_dim=noise_dim,
                                                                 device=device)
            loss_g = train_generator(generator=generator,
                                     discriminator=discriminator,
                                     loss_fn=loss_fn,
                                     batch_size=batch_size,
                                     noise_dim=noise_dim,
                                     opt_generator=opt_generator,
                                     device=device)
            
        # Record losses & scores
        results['loss_generator'].append(loss_g)
        results['loss_discriminator'].append(loss_g)
        results['real_scores'].append(real_score)
        results['fake_scores'].append(fake_score)

        print('-------------- EPOCH {} ----------------'.format(epoch+1))
        print('Generator Loss: {:.4f}, Real Score: {:.2f}%'.format(loss_g, real_score))
        print('Discriminator Loss: {:.4f}, Fake Score: {:.2f}%'.format(loss_d, fake_score))
        print()
    
        if save_samples is not None:
            save_samples(epoch+start_idx,  generator,fixed_latent, show=False)
    
    return results


def plot_loss_curves(results: dict[str, list[float]]) -> None:
    """
    Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {'loss_generator' : [],
            'loss_discriminator' : [],
            'real_scores' : [],
            'fake_scores' : [],}
    """
    loss_generator = results['loss_generator']
    loss_discriminator = results['loss_discriminator']

    real_scores = results['real_scores']
    fake_scores = results['fake_scores']

    epochs = range(len(results['loss_generator']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_generator, label='Generator Loss')
    plt.plot(epochs, loss_discriminator, label='Discriminator Loss')
    plt.title('Loss : Generator vs Discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, real_scores, label='Real Score')
    plt.plot(epochs, fake_scores, label='Fake Score')
    plt.title('Scores : Real vs Fake')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)

def plot_loss_curve_grid(results: dict[str, list[float]]) -> None:
    """
    Plots training curves of a results dictionary.
    Args:
        results (dict): Dictionary containing lists of values.
            Example:
            {'loss_generator': [float],
             'loss_discriminator': [float],
             'real_scores': [float],
             'fake_scores': [float]}
    Plots:
        - Generator Loss over epochs.
        - Discriminator Loss over epochs.
        - Real Score over epochs.
        - Fake Score over epochs.
    """

    loss_generator = results['loss_generator']
    loss_discriminator = results['loss_discriminator']
    real_scores = results['real_scores']
    fake_scores = results['fake_scores']

    epochs = range(len(loss_generator))

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, loss_generator, marker='o', color='b', label='Generator Loss')
    ax1.set_title('Generator Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, loss_discriminator, marker='s', color='r', label='Discriminator Loss')
    ax2.set_title('Discriminator Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, real_scores, marker='o', color='g', label='Real Score')
    ax3.set_title('Real Score')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, fake_scores, marker='s', color='purple', label='Fake Score')
    ax4.set_title('Fake Score')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

def images_to_video(image_folder:str, 
                    video_path:str,
                    target_duration : int=10, 
                    target_fps : int=60) -> None:
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
