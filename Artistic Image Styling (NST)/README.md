# Artistic Image Styling (Neural Style Transfer)

This repository contains a project that implements Neural Style Transfer (NST) using deep learning techniques. 
Neural Style Transfer is a technique that takes two images a content image and a style image and blends them together so that the output image looks like the content image but "painted" in the style of the style image.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Deployment](#deployment)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Neural Style Transfer (NST) is a deep learning technique that applies the artistic style of one image (the style image) to another image (the content image). The approach utilizes convolutional neural networks (CNNs) to separate and recombine content and style from different images, creating a visually appealing and stylistically unique output.

### Paper Details

The foundational paper for NST, "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, presents a method of using CNNs to achieve artistic style transfer. The technique involves optimizing an image to simultaneously match the content representation of a content image and the style representation of a style image.

### Loss Functions

The NST algorithm optimizes a combination of two loss functions:
- **Content Loss**: This measures the difference in content between the generated image and the content image. It is typically computed using the mean squared error between feature maps from a certain layer of the CNN.
- **Style Loss**: This measures the difference in style between the generated image and the style image. It is calculated using the Gram matrix of feature maps from multiple layers of the CNN to capture the correlations between different filter responses.

In this implementation, we use the VGG19 network architecture without batch normalization. VGG19 is a deep convolutional network that consists of 19 layers, including 16 convolutional layers and 3 fully connected layers. For the purpose of NST, only the convolutional layers are used because they capture hierarchical image features that are essential for content and style representation.
The VGG19 model pre-trained on the ImageNet dataset is employed to extract feature representations from the content and style images. Specifically, feature maps from layers like `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` (for style) and `conv4_2` (for content) are utilized to compute the respective loss functions.


## Project Structure

```
.
├── images
│   ├── test_images.jpg
├── Generated Images
│   ├── Generated Images during training
├── output
│   ├── generated images on test_images.jpg
├── styles
│   ├── styling_images.jpg
├── results
│   ├── output1.png               ## Image Grid (Input_Image, Style_Image, Generated_Image)
├── src
│   ├── __init__.py               ## Module Initialization
│   ├── model.py                  ## Pytorch Code for Model Implementation
│   ├── plot_loss_curve.py        ## loss_curve ploting Implementation
│   ├── train.py                  ## Train Model
├── utils
│   ├── __init__.py               ## Module Initialization
│   ├── utils.py                  ## Utility Functions
│   ├── config.py                 ## Model Config
├── main.py                       ## Implementation of Main Logic 
├── ArtisticImageStyling.ipynb    ## Notebook Version 
├── ModelResults.ipynb            ## Results Obtained 
├── config.yaml                   ## config.yaml 
├── demo.gif                      ## gif file of output 
├── README.md
└── requirements.txt
```

## Examples

### Example-1
![Example 1](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Artistic%20Image%20Styling%20(NST)/results/output1.png)

### Example-2
![Example 2](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Artistic%20Image%20Styling%20(NST)/results/output2.png)

### Example-3
![Example 3](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Artistic%20Image%20Styling%20(NST)/results/output3.png)

To view more examples, visit the `results/` directory. Alternatively, you can run the following command to view all the style implementations on the dataset:
```bash
python ModelResults.ipynb
```

## Dataset

The entire dataset, including test images to be styled and various artworks to be used as styles, has been uploaded as a Kaggle dataset. You can access it via the following link:
[Link to Kaggle dataset](https://www.kaggle.com/datasets/skjha69/artistic-images-for-neural-style-transfer).

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
cd Python-Deep-Learning-Projects/Artistic%20Image%20Styling%20(NST)
```

## Usage

To Style your own Custom Image and perform Neural Style Transfer, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
    cd Python-Deep-Learning-Projects/Artistic Image Styling(NST)
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    venv/Scripts/activate 
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure `config.yaml` located in the root Directory:

    Place your content image into the `images/` directory and select the style image to perform NST. Then replace the content image path with yours and do the same for the style image in the `config.yaml` file:
    ```yaml
    EPOCHS: 1000
    LEARNING_RATE: 0.001
    WEIGHTS: [2, 0.01]
    IMG_SIZE: 720
    IMG_CHANNEL: 3
    IMG_NAME : "your_image_name"
    ORIGINAL_IMG_PATH: "images/your_image_name.png"
    STYLE_IMG_PATH: "styles/style (5).jpg"
    ```

5. Train the model from the `root` directory:
    Once the settings and image paths have been configured, run:
    ```bash
    python main.py
    ``` 

For more detailed instructions, refer to the `utils/utils.py` , `utils/config.py` and `src/model.py` directory, which contains all the necessary scripts for data preprocessing, training, and evaluation.


## Model Architecture

### Neural Style Transfer (NST) Using VGG19 Architecture

Neural Style Transfer (NST) leverages a pre-trained convolutional neural network to blend the content of one image with the style of another. Here’s an overview of the VGG19 architecture and the implementation of NST in PyTorch.

#### VGG19 Architecture
- **Layers**: 
  - 19 layers: 16 convolutional layers and 3 fully connected layers.
  - Small receptive fields of 3x3 convolutional filters.
  - ReLU activations after each convolution layer.
- **Characteristics**:
  - Depth and simplicity make it effective for feature extraction.
  - Pre-trained on large datasets like ImageNet for robust feature representations.

#### NST Implementation Steps

1. **Loading VGG19 Model**:
   - Utilize a pre-trained VGG19 model for feature extraction.

2. **Selecting Layers for Content and Style Representation**:
   - **Content Representation**:
     - Typically extracted from a deeper layer (e.g., `conv4_2`) to capture high-level features.
   - **Style Representation**:
     - Extracted from multiple layers (e.g., `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`) to capture both local and global style patterns.

3. **Initializing the Input Image**:
   - The input image can be initialized randomly or as a copy of the content image.

4. **Defining Loss Functions**:
   - **Content Loss**:
     - Measures the difference in feature representations between the generated image and the content image.
   - **Style Loss**:
     - Measures the difference in texture and color patterns by comparing the Gram matrices of feature maps between the generated and style images.

5. **Optimization**:
   - Iteratively optimize the input image to minimize a weighted sum of the content and style losses using backpropagation.
   - Maintain the structure and semantic content of the content image.
   - Impose the artistic style of the style image onto the generated image.

#### Summary
- **Objective**:
  - Create a new image that combines the content of one image with the style of another.
- **Outcome**:
  - The generated image blends the high-level features of the content image with the texture and color patterns of the style image, resulting in a visually pleasing artistic image.

By following these steps, the NST process effectively blends content and style, leveraging the powerful feature extraction capabilities of the VGG19 architecture in PyTorch.


## Deployment

```python
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

```

## References

- **VGG19 Architecture**:
  - Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
  - [VGG19 on PyTorch Hub](https://pytorch.org/hub/pytorch_vision_vgg/)

- **Neural Style Transfer**:
  - Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. "A Neural Algorithm of Artistic Style." [arXiv:1508.06576](https://arxiv.org/abs/1508.06576)
    
- **Kaggle Dataset**:
  - Artistic Images For Neural Style Transfer "A Comprehensive Collection for Neural Style Transfer Research and Applications." [kaggle dataset](https://www.kaggle.com/datasets/skjha69/artistic-images-for-neural-style-transfer)


## Contributing

Contributions are welcome! If you have any improvements or new models to add, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for more details.

## Contact

For any questions or suggestions, please open an issue or contact me at saketjha0324@gmail.com. or [Linkedin](https://www.linkedin.com/in/saketjha34/)


Happy coding!
