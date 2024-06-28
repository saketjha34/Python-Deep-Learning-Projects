import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class Pix2PixDataset(Dataset):
    """
    Dataset class for Pix2Pix image translation task.

    Args:
        root_dir (str): Path to the root directory containing image pairs.
        split_size (int, optional): Size of each split image (default is 600).
        transform (transforms.Compose, optional): Optional transformations to apply to the images (default is None).

    Attributes:
        split_size (int): Size of each split image.
        root_dir (str): Root directory containing image pairs.
        transform (transforms.Compose): Transformation pipeline for preprocessing images.
        image_paths (list): List of paths to all images in the dataset.

    Methods:
        __len__():
            Returns the total number of images in the dataset.

        __getitem__(idx):
            Fetches and returns a tuple of input and target images at the given index.

    """
    def __init__(self, root_dir:str, split_size:int = 600, transform:transforms.Compose=None):
        self.split_size = split_size
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.jpg') or fname.endswith('.png')]

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches and returns the input and target images at the given index.

        Args:
            idx (int): Index of the image pair to retrieve.

        Returns:
            tuple: A tuple containing input and target images.
                - input_image (Tensor or PIL.Image): Input image cropped from the left side.
                - target_image (Tensor or PIL.Image): Target image cropped from the right side.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        input_image = image.crop((0, 0, self.split_size, self.split_size))  
        target_image = image.crop((self.split_size, 0, self.split_size*2, self.split_size)) 
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image