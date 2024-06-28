import os
import torch  
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt  
from matplotlib.gridspec import GridSpec
from torchvision.utils import save_image


def plot_random_images(dataset: torch.utils.data.Dataset, n: int = 3, seed: int = None) -> None:
    """
    Plots a grid of n x n pairs of random images and their corresponding target images from the dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset from which to draw the images.
        n (int, optional): The size of the grid (number of rows and columns). Default is 3.
        seed (int, optional): Random seed for reproducibility. Default is None.
    """
    if seed:
        random.seed(seed)

    fig, axes = plt.subplots(n, n*2, figsize=(12, 12))
    
    for i in range(n):
        for j in range(n):
            random_idx = torch.randint(0, len(dataset), size=[1]).item()
            input_image, target_image = dataset[random_idx]
            
            ax_input = axes[i, j*2]
            ax_target = axes[i, j*2 + 1]
            
            ax_input.imshow(input_image.permute(1, 2, 0))
            ax_input.set_title(f'Image set {i*n + j + 1} - Input')
            ax_input.axis('off')
            
            ax_target.imshow(target_image.permute(1, 2, 0))
            ax_target.set_title(f'Image set {i*n + j + 1} - Target')
            ax_target.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_image_grid(input_images: torch.Tensor, target_images: torch.Tensor, generated_images: torch.Tensor = None, nmax: int = None) -> None:
    """
    Display a grid of input, target, and optionally generated images in a 1x3 format for each set.

    Parameters:
    input_images (torch.Tensor): A batch of input images of shape (batch_size, 3, 256, 256).
    target_images (torch.Tensor): A batch of target images of shape (batch_size, 3, 256, 256).
    generated_images (torch.Tensor, optional): A batch of generated images of shape (batch_size, 3, 256, 256). Defaults to None.
    nmax (int, optional): Maximum number of images to display. If None, display all images in the batch. Defaults to None.

    Returns:
    None: This function does not return any value. It displays a grid of images.
    """
    if nmax is not None:
        input_images = input_images[:nmax]
        target_images = target_images[:nmax]
        if generated_images is not None:
            generated_images = generated_images[:nmax]

    num_images = input_images.shape[0]
    ncols = 3 if generated_images is not None else 2

    fig, axes = plt.subplots(num_images, ncols, figsize=(10, 4 * num_images))

    # Set column titles
    col_titles = ['Input', 'Target']
    if generated_images is not None:
        col_titles.append('Generated')
    
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=20)

    # Plot images
    for i in range(num_images):
        axes[i, 0].imshow(input_images[i].permute(1, 2, 0))
        axes[i, 0].axis('off')

        axes[i, 1].imshow(target_images[i].permute(1, 2, 0))
        axes[i, 1].axis('off')

        if generated_images is not None:
            axes[i, 2].imshow(generated_images[i].permute(1, 2, 0))
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def show_batch(dataloader: torch.utils.data.DataLoader, 
               plot_image_grid: callable, 
               nmax: int = 4) -> None:
    """
    Display a batch of images from a DataLoader using a specified function to create an image grid.

    Parameters:
    dataloader (torch.utils.data.DataLoader): The DataLoader to fetch the batch of images from.
    plot_image_grid (callable): The function used to create and display the image grid.
    nmax (int, optional): The maximum number of images to display. Defaults to 64.

    Returns:
    None: This function does not return any value. It displays a batch of images using the provided image grid function.
    """
    for images, targets in dataloader:
        plot_image_grid(images.cpu(),targets.cpu(),targets.cpu())
        break



def make_image_grid(input_images: torch.Tensor,
                    target_images: torch.Tensor,
                    generated_images: torch.Tensor ,
                    nmax: int = None) -> torch.Tensor:
    """
    Create a grid of input, target, and generated images.

    This function concatenates input, target, and optionally generated images into a single tensor
    that represents a grid. Each set of images is concatenated along the height dimension, and the
    resulting grid tensor can be used for saving or displaying.

    Parameters:
    input_images (torch.Tensor): A batch of input images with shape (batch_size, channels, height, width).
    target_images (torch.Tensor): A batch of target images with shape (batch_size, channels, height, width).
    generated_images (torch.Tensor): A batch of generated images with shape (batch_size, channels, height, width).
    nmax (int, optional): Maximum number of images to include in the grid. If None, all images are included. Defaults to None.

    Returns:
    torch.Tensor: A tensor representing the image grid.
    """
    all_images = torch.cat((input_images, target_images, generated_images), dim=0)
    nrow = input_images.size(0)
    grid = torch.cat([all_images[i::nrow] for i in range(nrow)], dim=2)
    return grid

def save_image_grid(index: int, 
                    input_images:torch.Tensor,
                    target_images:torch.Tensor,
                    generated_images:torch.Tensor, 
                    make_image_grid:callable ,
                    evaluation_dir:str) -> None:
    """
    Save a grid of input, target, and generated images to a file.

    This function creates a directory for saving images, generates a grid of images using the
    specified `make_image_grid` function, and saves the resulting image grid to a file.

    Parameters:
    index (int): Index for the filename to save the image grid.
    input_images (torch.Tensor): A batch of input images with shape (batch_size, channels, height, width).
    target_images (torch.Tensor): A batch of target images with shape (batch_size, channels, height, width).
    generated_images (torch.Tensor): A batch of generated images with shape (batch_size, channels, height, width).
    make_image_grid (callable): Function to create the image grid.
    show (bool, optional): Flag to indicate whether to display the image grid after saving (not implemented). Defaults to True.

    Returns:
    None: This function does not return any value. It saves the image grid to a file.
    """
    image_fname = 'generated-images-{0:0=4d}.png'.format(index)
    input_images = input_images.cpu()
    target_images = target_images.cpu()
    generated_images = generated_images.cpu()
    grid = make_image_grid(input_images, target_images, generated_images)
    save_image(grid, os.path.join(evaluation_dir, image_fname))


def train_generator_discriminator(train_loader :torch.utils.data.DataLoader, 
                                  discriminator:torch.nn.Module , 
                                  generator:torch.nn.Module , 
                                  opt_discriminator :torch.optim.Optimizer ,
                                  opt_generator:torch.optim.Optimizer ,
                                  l1_loss_fn:torch.nn.Module , 
                                  loss_fn:torch.nn.Module , 
                                  g_scaler:torch.cuda.amp.GradScaler,
                                  d_scaler:torch.cuda.amp.GradScaler,
                                  L1_LAMBDA:int,
                                  device:torch.device):


    loop = train_loader

    for idx, (x, y) in enumerate(tqdm(loop)):
        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast():
            y_fake = generator(x)
            D_real = discriminator(x, y)
            D_real_loss = loss_fn(D_real, torch.ones_like(D_real))
            D_fake = discriminator(x, y_fake.detach())
            D_fake_loss = loss_fn(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        discriminator.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_discriminator)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            D_fake = discriminator(x, y_fake)
            G_fake_loss = loss_fn(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss_fn(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_generator.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_generator)
        g_scaler.update()


    return D_real_loss.item(), D_fake_loss.item() ,D_loss.item() , G_fake_loss.item(), L1.item(), G_loss.item()


def train_model(train_loader :torch.utils.data.DataLoader, 
                discriminator:torch.nn.Module , 
                generator:torch.nn.Module , 
                opt_discriminator :torch.optim.Optimizer ,
                opt_generator:torch.optim.Optimizer ,
                l1_loss_fn:torch.nn.Module , 
                loss_fn:torch.nn.Module , 
                g_scaler:torch.cuda.amp.GradScaler,
                d_scaler:torch.cuda.amp.GradScaler,
                L1_LAMBDA:int,
                epochs:int,
                device:torch.device):

    results = {
        'D_real_loss' : [],
        'D_fake_loss' : [],
        'D_loss' : [],
        'G_fake_loss' : [],
        'L1' : [],
        'G_loss' : [],
    }

    for epoch in range(epochs):

        D_real_loss, D_fake_loss ,D_loss,G_fake_loss, L1, G_loss  = train_generator_discriminator(train_loader , 
                                                                                                 discriminator, 
                                                                                                 generator,
                                                                                                 opt_discriminator,
                                                                                                 opt_generator, 
                                                                                                 l1_loss_fn,
                                                                                                 loss_fn,
                                                                                                 g_scaler,
                                                                                                 d_scaler , 
                                                                                                 L1_LAMBDA,
                                                                                                 device)
                                                                                            
        
        results["D_real_loss"].append(D_real_loss)
        results["D_fake_loss"].append(D_fake_loss)
        results["D_loss"].append(D_loss)
        results["G_fake_loss"].append(G_fake_loss)
        results["L1"].append(L1)
        results["G_loss"].append(G_loss)

        if epoch % 50 == 0:
            print('-------------------------- EPOCH {} -------------------------'.format(epoch+1))
            print('Generator L1 Loss: {:.6f}, Generator Fake Loss: {:.6f}'.format(L1, G_fake_loss))
            print('Discriminator Real Loss: {:.6f}, Discriminator Fake Loss: {:.6f}'.format(D_real_loss, D_fake_loss))
            print('Discriminator Total Loss: {:.6f}, Generator Total Loss: {:.6f}'.format(D_loss, G_loss))
            print()

    return results


def eval_model(train_loader: torch.utils.data.DataLoader,
               generator: torch.nn.Module,
               make_image_grid: callable,
               save_image_grid: callable,
               evaluation_dir: str,
               evaluated_val_dir: str,
               device: torch.device) -> None:
    """
    Evaluates the generator model on the provided data loader and saves generated images.

    Parameters:
    - train_loader (torch.utils.data.DataLoader): DataLoader providing input and target images.
    - generator (torch.nn.Module): The generator model to evaluate.
    - make_image_grid (callable): Function to create a grid of images for visualization.
    - save_image_grid (callable): Function to save a grid of images to disk.
    - evaluation_dir (str): Directory where evaluation image grids will be saved.
    - evaluated_val_dir (str): Directory where individual generated images will be saved.
    - device (torch.device): Device on which to perform the evaluation (e.g., 'cpu' or 'cuda').

    Returns:
    - None
    """
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(evaluated_val_dir, exist_ok=True)
    
    for idx, (input_image, target_image) in enumerate(tqdm(train_loader)):
        input_image = input_image.to(device)
        target_image = target_image.to(device)
        gen_img_name = f'generated-image-{idx:04d}.png'

        with torch.no_grad():
            generated_image = generator(input_image).to(device)
            if (idx + 1) % 20 == 0:
                save_image_grid(idx, input_image, target_image, generated_image, make_image_grid, evaluation_dir)
            save_image(generated_image, os.path.join(evaluated_val_dir, gen_img_name))


def plot_loss_curve_grid(results: dict[str, list[float]]) -> None:

    D_real_loss = results['D_real_loss']
    D_fake_loss = results['D_fake_loss']
    D_loss = results['D_loss']
    
    G_fake_loss = results['G_fake_loss']
    L1 = results['L1']
    G_loss = results['G_loss']

    epochs = range(len(D_real_loss))

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, D_real_loss,  color='b', label='D_real_loss')
    ax1.set_title('Discriminator Real Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, D_fake_loss, color='r', label='D_fake_loss')
    ax2.set_title('Discriminator Fake Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, D_loss, color='g', label='D_loss')
    ax3.set_title('Discriminator Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, G_fake_loss,  color='c', label='G_fake_loss')
    ax4.set_title('Generator Fake Loss')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(epochs, L1, color='m', label='L1')
    ax5.set_title('L1 Loss')
    ax5.set_xlabel('Epochs')
    ax5.set_ylabel('Loss')
    ax5.legend()
    ax5.grid(True)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(epochs, G_loss, color='y', label='G_loss')
    ax6.set_title('Generator Loss')
    ax6.set_xlabel('Epochs')
    ax6.set_ylabel('Loss')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()


def plot_loss_curves(results: dict[str, list[float]]) -> None:

    D_real_loss = results['D_real_loss']
    D_fake_loss = results['D_fake_loss']
    D_loss = results['D_loss']
    
    G_fake_loss = results['G_fake_loss']
    L1 = results['L1']
    G_loss = results['G_loss']
    
    epochs = range(len(D_real_loss))

    plt.figure(figsize=(8, 16))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, D_real_loss, label='D_real_loss', color='b')
    plt.plot(epochs, D_fake_loss, label='D_fake_loss', color='r')
    plt.title('D_real_loss vs D_fake_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs, D_loss, label='D_loss', color='g')
    plt.plot(epochs, G_loss, label='G_loss', color='y')
    plt.title('D_loss vs G_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs, G_fake_loss, label='G_fake_loss', color='c')
    plt.plot(epochs, L1, label='L1', color='m')
    plt.title('G_fake_loss vs L1')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()