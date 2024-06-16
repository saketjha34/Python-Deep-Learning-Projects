# Artistic Image Styling (Neural Style Transfer)

This repository contains a project that implements Neural Style Transfer (NST) using deep learning techniques. 
Neural Style Transfer is a technique that takes two images—a content image and a style image—and blends them together so that the output image looks like the content image but "painted" in the style of the style image.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [DCGAN](#dcgan)
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

<!DOCTYPE html>
<html lang="en">
<body>
<iframe width="560" height="315" src="https://www.youtube.com/embed/muw9ZHTmwOA" > video</iframe>
</body>
</html>



## Project Structure

```
.
├── images
│   ├── test_images.jpg
├── Generated Images
│   ├── Generated Images during training
├── output
│   ├── generated images on test_images.jpg
├── results
│   ├── images of styles implemnted on the dataset
├── styles
│   ├── styling_images.jpg
├── src
│   ├── train.py
├── utils
│   ├── utils.py
│   ├── config.py
│   ├── model.py
├── ArtisticImageStyling.ipynb
├── ModelResults.ipynb
├── NST PAPER.pdf
├── output_video.mp4
├── README.md
└── requirements.txt
```

## Examples

### Example1
![Example 1](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Artistic%20Image%20Styling%20(NST)/results/output1.png)

### Example2
![Example 2](https://github.com/saketjha34/Python-Deep-Learning-Projects/blob/main/Artistic%20Image%20Styling%20(NST)/results/output2.png)

### Example3
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

1. Clone the repository:
    ```bash
    git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
    cd Python-Deep-Learning-Projects/Artistic%20Image%20Styling%20(NST)
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv imagestyling
    source imagestyling/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage


To train the model on your own custom image and perform neural style transfer, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/saketjha34/Python-Deep-Learning-Projects.git
    cd Python-Deep-Learning-Projects/Artistic%20Image%20Styling%20(NST)
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv imagestyling
    source imagestyling/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure `config.py` located in the `utils/` directory:
    ```bash
    cd utils/
    ```
    Place your content image into the `images/` directory and select the style image to perform NST. Then replace the content image path with yours and do the same for the style image in the `config.py` file:
    ```python
    EPOCHS = 6000
    LEARNING_RATE = 1e-3
    WEIGHTS = (2, 0.01)
    IMG_SIZE = 720
    IMG_CHANNEL = 3
    ORIGINAL_IMG_PATH = 'path/to/your/input/image.jpg'
    STYLE_IMG_PATH = 'path/to/your/style/image.jpg'
    ```

5. Train the model from the `src/` directory:
    ```bash
    cd ../src/
    ```
    Once the settings and image paths have been configured, run:
    ```bash
    python train.py
    ``` 

For more detailed instructions, refer to the `utils/utils.py` , `utils/config.py` and `utils/model.py` directory, which contains all the necessary scripts for data preprocessing, training, and evaluation.

## Results

During training, 50 images were generated and saved in the `Generated Images` folder. These images represent the progression and improvements of the generator model over time. Additionally, a compiled video of these generated images, showcasing the development of the anime faces throughout the training process, can be found [here](output_video.mp4).

```bash
cd Generated Images/generated-images-0012.png
```

## Model Architecture

The model is built from scratch using convolutional neural networks like ResNet18 and AlexNet. The architecture details are available in the `utils/models.py` directory.

Generator
The generator in a DCGAN takes random noise as input and transforms it through a series of transposed convolutional layers, batch normalization, and ReLU activations to produce a synthetic image. This network aims to create realistic-looking anime faces that can deceive the discriminator.

Discriminator
The discriminator is a convolutional neural network that classifies images as real or fake. It consists of convolutional layers, batch normalization, and Leaky ReLU activations. Its goal is to distinguish between real anime faces and those generated by the generator, thereby improving the generator's output through adversarial training.

For more details, refer to the original paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

### DCGAN

#### Overview
The paper *"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"* by Alec Radford, Luke Metz, and Soumith Chintala introduced the DCGAN architecture. This architecture aims to improve the stability of GAN training and to leverage deep convolutional networks for both the generator and discriminator models.

#### Generator Model
The generator network uses a series of transposed convolutional layers to transform a vector of random noise into a synthetic image. Key components include:
- **Transposed Convolutions**: Used to upsample the input, converting low-resolution feature maps to high-resolution images.
- **Batch Normalization**: Applied after each transposed convolution to stabilize training by normalizing layer inputs.
- **ReLU Activations**: Employed in all layers except the output layer, where a Tanh activation is used to ensure the output is in the range \([-1, 1]\).

#### Discriminator Model
The discriminator is a convolutional neural network designed to classify images as real or fake. Key components include:
- **Convolutional Layers**: Extract features from input images through a series of convolutions.
- **Batch Normalization**: Applied after each convolution to stabilize training and improve convergence.
- **Leaky ReLU Activations**: Used in all layers to allow a small gradient when the unit is not active, mitigating the vanishing gradient problem.

#### Training Guidelines
The training process for DCGANs involves alternately updating the generator and discriminator:
1. **Discriminator Training**: Update the discriminator by maximizing the probability of assigning the correct label to both real images from the dataset and fake images from the generator.
2. **Generator Training**: Update the generator by minimizing the probability of the discriminator correctly classifying its outputs as fake. This involves backpropagating the discriminator's error through the generator's weights.

Key training practices include:
- **Using Batch Normalization**: Stabilizes training by normalizing the inputs to each layer.
- **Leaky ReLU in Discriminator**: Prevents dying ReLU problem by allowing a small gradient flow when the neuron is not active.
- **Tanh Activation in Generator Output**: Ensures the output image pixels are in the range \([-1, 1]\), matching the preprocessing of input images.

For more details, refer to the original paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

## Deployment

The trained image generator is saved as `.pth` file in the `pytorch saved models` directory. These files can be used for further deployment purposes. You can load the models in PyTorch using the following code:

```python
import torch
import torch.nn as nn
from  utils.config import IMAGE_SIZE, IMG_CHANNELS, NUM_WORKERS, BATCH_SIZE , NOISE_DIM, LEARNING_RATE, EPOCHS

class Generator(nn.Module):
    def __init__(self , noise_channels , img_channels):
        super(Generator , self).__init__()

        self.Network = nn.Sequential(

            self._create_block(in_channels=noise_channels, out_channels=512, kernel_size=4, padding=0 , stride=1),
            self._create_block(in_channels=512, out_channels=256, kernel_size=4, padding=1 , stride=2),   
            self._create_block(in_channels=256, out_channels=128, kernel_size=4, padding=1 , stride=2),   
            self._create_block(in_channels=128, out_channels=64, kernel_size=4, padding=1 , stride=2), 

            nn.ConvTranspose2d(in_channels=64, out_channels=img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def _create_block(self,in_channels, out_channels , kernel_size , padding ,stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride, padding=padding,bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.Network(x)

# Load Generator model
generator = Generator( noise_channels = NOISE_DIM , img_channels = IMG_CHANNELS )
generator.load_state_dict(torch.load('pytorch saved models/AnimeFaceDCGANs.pth'))
```

## References

- Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." arXiv preprint arXiv:1511.06434 (2015). [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Kaggle Dataset [link](https://www.kaggle.com/datasets/splcher/animefacedataset)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

For any questions or suggestions, please open an issue or contact me @ saketjha0324@gmail.com. or [Linkedin](https://www.linkedin.com/in/saketjha34/)

---

Happy coding!
